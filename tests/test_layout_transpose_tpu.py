"""TPU correctness for tpu-inference's layout-transpose kernels.

These move data without computing on it, so the bar is the same one the KV-cache
family sets: **bitwise identical** to `jnp.transpose`, not close to it. Anything
else would mean elements landed in the wrong place, not that they rounded.

Shapes, axes and dtypes are upstream's own, from
`tests/kernels/transpose_test.py`, including the cases that exist to catch
specific failures: a 128x2048x256 array too large to load into VMEM whole, tile
sizes that do not divide their axis, and a bf16 384x128x512 transpose that needs
more scoped VMEM than XLA's 32 MiB default.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_layout_transpose_tpu.py -q
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "memory" / "layout_transpose"
MLA = ROOT / "kernels" / "attention" / "mla_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="the layout-transpose kernels are Mosaic TPU kernels",
)

#: Upstream's `test_xpose_full` parameterisation.
FULL_CASES = [
    ((1024, 1024), (1, 0)),
    ((32, 64, 128), (2, 0, 1)),
    ((8, 16, 32, 64), (3, 2, 1, 0)),
    ((128, 256), (1, 0)),
    ((16, 32, 64), (2, 0, 1)),
]

#: Upstream's `test_xpose_pipeline` parameterisation.  The 128x2048x256 case is
#: commented upstream as "~64MB (would OOM loading everything into VMEM)" --
#: that is the case distinguishing this kernel from `xpose_full`.
PIPELINE_CASES = [
    ((1024, 2048), (1, 0), 128, 128, 0, 1),
    ((2048, 1024), (1, 0), 256, 256, 0, 1),
    ((512, 1024, 16), (1, 0, 2), 64, 128, 0, 1),
    ((256, 512, 128), (1, 0, 2), 128, 128, 1, 0),
    ((128, 2048, 256), (1, 0, 2), 128, 128, 1, 0),
    ((4, 256, 512, 128), (0, 2, 1, 3), 64, 64, 2, 1),
    ((192, 256, 64), (1, 0, 2), 160, 64, 0, 1),
    ((320, 256, 64), (1, 0, 2), 160, 64, 0, 1),
]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return (load("xpose_kernel", FAMILY / "tpu_inference_optimized.py"),
            load("xpose_baseline", FAMILY / "baseline.py"))


def fp8(shape, seed=42):
    """Upstream generates float8_e4m3fn for every transpose case."""
    return jax.random.normal(jax.random.PRNGKey(seed), shape,
                             dtype=jnp.float8_e4m3fn)


def identical(actual, expected, what):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, what
    assert actual.dtype == expected.dtype, what
    # A transpose relocates elements; it never changes them.
    np.testing.assert_array_equal(actual, expected, err_msg=what)


@pytest.mark.parametrize("shape,axes", FULL_CASES)
def test_xpose_full_matches_jnp_transpose(modules, shape, axes):
    kernel, baseline = modules
    x = fp8(shape)
    actual = jax.jit(lambda a: kernel.xpose_full(a, transpose_axes=axes))(x)[0]
    identical(actual, baseline.transposed(x, axes), f"{shape} -> {axes}")


@pytest.mark.parametrize("shape,axes,n_tile,m_tile,parallel,pipeline",
                         PIPELINE_CASES)
def test_xpose_pipeline_matches_jnp_transpose(
        modules, shape, axes, n_tile, m_tile, parallel, pipeline):
    kernel, baseline = modules
    x = fp8(shape)
    actual = kernel.xpose_pipeline(
        x, transpose_axes=axes, n_tile=n_tile, m_tile=m_tile,
        parallel_axis=parallel, pipeline_axis=pipeline)[0]
    identical(actual, baseline.transposed(x, axes), f"{shape} -> {axes}")


def test_pin_vmem_returns_its_input_unchanged(modules):
    """The identity kernel: its effect is where the buffer lives, not its value.

    So the only thing a correctness test can pin down is that nothing changed —
    which, for a kernel whose whole body is a copy loop, is exactly the failure
    worth catching. That it *runs* rather than folding away is checked by
    `test_all_three_launch_points_reach_pallas`.
    """
    kernel, baseline = modules
    x = fp8((256, 512))
    identical(kernel.pin_vmem_custom_call(x)[0], baseline.pinned(x), "pin_vmem")


def test_all_three_launch_points_reach_pallas(modules):
    """Three launches in one file, so check all three, not just the alias."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    kernel, _ = modules
    x = fp8((256, 512))
    launches = {
        "xpose_full": count_pallas_launches(
            lambda a: kernel.xpose_full(a, transpose_axes=(1, 0)), (x,)),
        "xpose_pipeline": count_pallas_launches(
            lambda a: kernel.xpose_pipeline(a, transpose_axes=(1, 0)), (x,)),
        "pin_vmem_custom_call": count_pallas_launches(
            kernel.pin_vmem_custom_call, (x,)),
    }
    assert launches == dict.fromkeys(launches, 1), launches


def test_the_reference_is_not_itself_pallas(modules):
    """`jnp.transpose` must lower to a real transpose, not a Mosaic call."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    _, baseline = modules
    x = fp8((256, 512))
    assert count_pallas_launches(
        lambda a: baseline.transposed(a, (1, 0)), (x,)) == 0


def test_a_tile_that_does_not_divide_its_axis_is_lowered_not_used(modules):
    """Tile sizes are requests. Using one as given would drop rows silently.

    Upstream's case: shape[0]=448 with n_tile=160. For fp8 the sublane multiple
    is `get_dtype_packing(fp8) * 8 == 32`, and the largest divisor of 448 that
    is <= 160 and a multiple of 32 is 64. The kernel must pick 64 and still be
    exactly right — a non-divisor tile would leave the tail unprocessed.
    """
    kernel, baseline = modules
    assert kernel.get_dtype_packing(jnp.float8_e4m3fn) * 8 == 32
    assert kernel.prev_closest_valid_divisor(448, 160, multiple_of=32) == 64

    x = fp8((448, 192, 128), seed=0)
    actual = kernel.xpose_pipeline(
        x, transpose_axes=(1, 0, 2), n_tile=160, m_tile=64)[0]
    identical(actual, baseline.transposed(x, (1, 0, 2)), "lowered n_tile")


def test_no_valid_tile_raises_rather_than_silently_mistiling(modules):
    """300 has no divisor that is both <= 160 and a multiple of 32.

    The kernel refuses instead of proceeding, which is the behaviour that makes
    the test above safe to trust.
    """
    kernel, _ = modules
    with pytest.raises(ValueError):
        kernel.xpose_pipeline(fp8((300, 192, 128), seed=0),
                              transpose_axes=(1, 0, 2), n_tile=160, m_tile=64)


def test_vmem_limit_bytes_admits_a_tiling_the_default_rejects(modules):
    """Upstream's own regression case, and this corpus's recurring trap.

    A bf16 384x128x512 transpose at n_tile=128, m_tile=64 needs ~33.6 MiB of
    *scoped* VMEM — over XLA's 32 MiB default, so it fails compilation with
    E1001 CompileTimeScopedVmemOom while sitting comfortably inside total VMEM.
    `vmem_limit_bytes` raises the limit for this one `pallas_call`, with no
    global flag. This is the shape MLA v2's `prepare_outputs` transpose emits,
    so it is not hypothetical.
    """
    kernel, baseline = modules
    x = jax.random.normal(jax.random.PRNGKey(0), (384, 128, 512),
                          dtype=jnp.bfloat16)
    with pytest.raises(Exception) as excinfo:
        kernel.xpose_pipeline(x, transpose_axes=(1, 0, 2), n_tile=128,
                              m_tile=64)
    assert "vmem" in str(excinfo.value).lower(), (
        "expected a scoped-VMEM failure without vmem_limit_bytes")

    actual = kernel.xpose_pipeline(
        x, transpose_axes=(1, 0, 2), n_tile=128, m_tile=64,
        vmem_limit_bytes=64 * 1024 * 1024)[0]
    identical(actual, baseline.transposed(x, (1, 0, 2)), "raised vmem limit")


@pytest.mark.parametrize("number,divider,expected", [
    (112, 160, 112), (300, 160, 150), (160, 160, 160), (256, 160, 128),
    (160, 128, 80), (128, 1, 1), (64, 1000, 64),
])
def test_prev_closest_valid_divisor(modules, number, divider, expected):
    """Upstream's own table, which is the host-side half of the contract."""
    kernel, _ = modules
    assert kernel.prev_closest_valid_divisor(number, divider) == expected


@pytest.mark.parametrize("number,divider,multiple_of,expected", [
    (128, 160, 8, 128), (512, 160, 8, 128), (120, 100, 8, 40),
    (256, 128, 8, 128), (64, 1000, 8, 64), (300, 160, 1, 150), (4, 10, 8, 4),
])
def test_prev_closest_valid_divisor_multiple_of(
        modules, number, divider, multiple_of, expected):
    kernel, _ = modules
    assert kernel.prev_closest_valid_divisor(
        number, divider, multiple_of=multiple_of) == expected


@pytest.mark.parametrize("number,divider,multiple_of", [
    (4, 2, 8), (300, 160, 8), (150, 100, 8),
])
def test_prev_closest_valid_divisor_raises(modules, number, divider, multiple_of):
    kernel, _ = modules
    with pytest.raises(ValueError):
        kernel.prev_closest_valid_divisor(number, divider,
                                          multiple_of=multiple_of)


def test_the_divisors_replacement_matches_the_definition(modules):
    """`sympy.divisors` is not in the pinned set, so it was reimplemented."""
    kernel, _ = modules
    for n in list(range(1, 200)) + [300, 448, 1024, 2048, 4096]:
        assert kernel.divisors(n) == [d for d in range(1, n + 1) if n % d == 0], n


def test_the_mla_copy_has_not_drifted_from_this_one():
    """MLA v2 inlines two of these functions; the copies must stay identical.

    This is the corpus's only deliberate duplication of a kernel: MLA v2
    physically transposes its head-major operands on entry and exit, so it
    cannot run without `xpose_pipeline`, and that inlined copy is a dependency
    rather than a counted launch point. Two copies invite silent divergence, so
    compare them structurally — by AST, which ignores comments and formatting
    and catches anything that would change behaviour.
    """
    shared = ("xpose_pipeline", "prev_closest_valid_divisor", "divisors")

    def definitions(path: Path) -> dict[str, str]:
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node)
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name in shared}

    here = definitions(FAMILY / "tpu_inference_optimized.py")
    there = definitions(MLA / "tpu_inference_v2_optimized.py")
    assert set(here) == set(shared), sorted(here)
    assert set(there) == set(shared), sorted(there)
    for name in shared:
        assert here[name] == there[name], (
            f"{name} has drifted between the layout_transpose migration and "
            f"the copy inlined into MLA v2")
