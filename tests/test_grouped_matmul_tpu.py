"""Small-shape TPU correctness for the grouped-matmul family.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest tests/test_grouped_matmul_tpu.py -q
"""

from __future__ import annotations

import inspect

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "moe" / "grouped_matmul"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="grouped-matmul kernels are Mosaic TPU kernels",
)


def load(name: str, filename: str):
    path = FAMILY / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return {
        "baseline": load("gm_baseline", "baseline.py"),
        "jaxbench": load("gm_jaxbench", "jaxbench_optimized.py"),
        "tpu_inference": load("gm_tpu_inference", "tpu_inference_optimized.py"),
        "sglang_jax": load("gm_sglang_jax", "sglang_jax_optimized.py"),
        "maxtext_v2": load("gm_maxtext_v2", "maxtext_optimized.py"),
        "tokamax_v2": load("gm_tokamax_v2", "tokamax_optimized.py"),
        "sglang_jax_v2": load("gm_sglang_v2", "sglang_jax_v2_optimized.py"),
        "tpu_inference_v2": load("gm_ti_v2", "tpu_inference_v2_optimized.py"),
    }


def cosine(actual, expected) -> float:
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))


def rms_relative(actual, expected) -> float:
    """RMS error relative to the reference's own RMS magnitude."""
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(np.sqrt(((a - e) ** 2).mean())
                 / (np.sqrt((e ** 2).mean()) + 1e-12))


def assert_matches(actual, expected, name: str = "", *,
                   cos_bar: float = 0.9999, rms_bar: float = 2e-2) -> None:
    """Direction *and* magnitude, because `cosine` alone checks only direction.

    `cosine` is scale-invariant: an output multiplied by any constant -- 0.5,
    2, a million -- still scores 1.0 against its reference and sails past a
    0.9999 bar. A wrong dequantization scalar, fp8 scale or softmax
    normalisation constant is exactly that kind of error, and it is a plausible
    failure mode for these kernels, so direction alone is not evidence.

    The RMS-relative bar is this corpus's own existing pattern, used in the
    splash and flash backward tests and in gated linear attention. 2e-2 is the
    value those chose. It is comfortably satisfied here: at the 0.9999 cosine
    bar the angular term alone contributes sqrt(2*(1-0.9999)) = 1.4e-2, and the
    cosines this suite actually records are 0.99997 or better, putting the
    angular contribution at 7.7e-3 or less. What the bar catches is any scale
    error above ~2%.
    """
    a = np.asarray(actual, np.float32)
    e = np.asarray(expected, np.float32)
    label = f"{name}: " if name else ""
    similarity = cosine(a, e)
    assert similarity > cos_bar, f"{label}cosine {similarity} <= {cos_bar}"
    scale = rms_relative(a, e)
    assert scale < rms_bar, f"{label}RMS-relative {scale} >= {rms_bar}"


IMPLEMENTATIONS = ("jaxbench", "tpu_inference", "sglang_jax")


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("balanced", [True, False])
def test_matches_pure_jax_reference(modules, implementation, balanced):
    baseline = modules["baseline"]
    inputs = baseline.create_inputs(
        rows=1024, num_groups=8, k=256, n=256, balanced=balanced
    )
    reference = (
        baseline.grouped_matmul_batched_dense
        if balanced
        else baseline.grouped_matmul_loop
    )
    expected = jax.jit(reference)(*inputs)
    actual = jax.jit(modules[implementation].kernel)(*inputs)
    assert actual.shape == expected.shape
    assert_matches(actual, expected)
    assert float(jnp.max(jnp.abs(actual - expected))) < 1e-3


def test_pure_jax_references_agree(modules):
    """The two baselines must agree, or neither can be trusted as a denominator."""
    baseline = modules["baseline"]
    inputs = baseline.create_inputs(rows=1024, num_groups=8, k=256, n=256)
    dense = jax.jit(baseline.grouped_matmul_batched_dense)(*inputs)
    loop = jax.jit(baseline.grouped_matmul_loop)(*inputs)
    np.testing.assert_array_equal(np.asarray(dense), np.asarray(loop))


def test_batched_dense_baseline_rejects_uneven_groups(modules):
    baseline = modules["baseline"]
    inputs = baseline.create_inputs(
        rows=1024, num_groups=8, k=256, n=256, balanced=False
    )
    with pytest.raises(ValueError, match="equal group sizes"):
        baseline.grouped_matmul_batched_dense(*inputs)


def test_ragged_dot_is_a_mosaic_kernel_not_a_baseline(modules):
    """Guard the finding that jax.lax.ragged_dot is itself a TPU custom call.

    If a future JAX release lowers it to plain HLO it would become a legitimate
    baseline, and this test should fail so that choice is revisited rather than
    silently inherited.
    """
    baseline = modules["baseline"]
    inputs = baseline.create_inputs(rows=1024, num_groups=8, k=256, n=256)
    text = jax.jit(baseline.grouped_matmul_ragged_dot).lower(*inputs).compile().as_text()
    assert 'custom_call_target="tpu_custom_call"' in text


def test_covered_rows_agree_when_a_group_is_empty(modules):
    """An empty group is inside the contract; the rows it does not cover are not.

    All implementations must agree on rows [0, sum(group_sizes)).  Nothing is
    asserted about the rest: JAXBench leaves those rows uninitialised, so their
    contents depend on allocator history -- see baseline.UNCOVERED_ROWS.
    """
    baseline = modules["baseline"]
    lhs, rhs, group_sizes = baseline.create_inputs(
        rows=1024, num_groups=8, k=256, n=256
    )
    group_sizes = group_sizes.at[-1].set(0)  # last group empty: 896 rows covered
    covered = slice(0, 896)

    expected = np.asarray(jax.jit(baseline.grouped_matmul_loop)(lhs, rhs, group_sizes))
    for name in IMPLEMENTATIONS:
        got = np.asarray(jax.jit(modules[name].kernel)(lhs, rhs, group_sizes))
        assert_matches(got[covered], expected[covered], name)
        assert np.abs(got[covered] - expected[covered]).max() < 1e-3, name


def test_uncovered_rows_are_not_a_contract(modules):
    """Pin down that the uncovered region is unusable, so nobody compares on it.

    JAXBench never writes these rows.  Running the same jitted call after other
    same-shaped work changes what they contain, which is the evidence that they
    are uninitialised memory rather than a second valid convention.
    """
    baseline = modules["baseline"]
    lhs, rhs, group_sizes = baseline.create_inputs(
        rows=1024, num_groups=8, k=256, n=256
    )
    empty_last = group_sizes.at[-1].set(0)
    tail = slice(896, 1024)
    jaxbench = jax.jit(modules["jaxbench"].kernel)

    before = np.asarray(jaxbench(lhs, rhs, empty_last))[tail].copy()
    for _ in range(6):
        dirty = jax.jit(baseline.grouped_matmul_batched_dense)(lhs, rhs, group_sizes)
        dirty.block_until_ready()
        del dirty
    after = np.asarray(jaxbench(lhs, rhs, empty_last))[tail].copy()

    assert not np.array_equal(before, after), (
        "the uncovered region became reproducible; re-check whether it is now "
        "a real contract before comparing implementations on it"
    )


# ---------------------------------------------------------------------------
# Megablox v2: the qwix-free kernels
# ---------------------------------------------------------------------------

V2_IMPLEMENTATIONS = ("maxtext_v2", "tokamax_v2",
                      "sglang_jax_v2", "tpu_inference_v2")
#: Only MaxText and Tokamax ship the transposed pass beside the forward.
V2_WITH_TGMM = ("maxtext_v2", "tokamax_v2")


@pytest.mark.parametrize("implementation", V2_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", [(1024, 8, 256, 256), (2048, 16, 512, 512)])
def test_v2_matches_reference_bitwise(modules, implementation, shape):
    """With preferred_element_type pinned to f32, v2 is bit-exact."""
    baseline = modules["baseline"]
    rows, groups, k, n = shape
    lhs, rhs, group_sizes = baseline.create_inputs(
        rows=rows, num_groups=groups, k=k, n=n
    )
    expected = jax.jit(baseline.grouped_matmul_batched_dense)(lhs, rhs, group_sizes)
    actual = modules[implementation].gmm_v2(
        lhs, rhs, group_sizes, preferred_element_type=jnp.float32
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("implementation", V2_IMPLEMENTATIONS)
def test_v2_default_output_dtype_loses_precision(modules, implementation):
    """Pin the v1 -> v2 API change that silently costs precision.

    v1 gmm defaults ``preferred_element_type`` to float32.  v2 defaults it to
    the input dtype, so accumulation rounds to bf16 per k-tile.  At k=4096 that
    is enough to drop below the corpus 0.9999 cosine threshold, which makes a
    drop-in v1 -> v2 swap a silent accuracy regression.
    """
    baseline = modules["baseline"]
    # k=1024 is enough to show the divergence; the profiled k=4096 shape makes
    # this test dominate the suite's runtime for no extra signal.
    lhs, rhs, group_sizes = baseline.create_inputs(
        rows=2048, num_groups=16, k=1024, n=512
    )
    expected = np.asarray(
        jax.jit(baseline.grouped_matmul_batched_dense)(lhs, rhs, group_sizes),
        np.float32,
    )
    module = modules[implementation]

    exact = np.asarray(
        module.gmm_v2(lhs, rhs, group_sizes, preferred_element_type=jnp.float32),
        np.float32,
    )
    np.testing.assert_array_equal(exact, expected)

    default = np.asarray(module.gmm_v2(lhs, rhs, group_sizes), np.float32)
    assert not np.array_equal(default, expected), (
        "the default output dtype now matches float32; re-check whether the "
        "v1 -> v2 precision note still applies"
    )


@pytest.mark.parametrize(
    "filename",
    ["maxtext_optimized.py", "tokamax_optimized.py",
     "sglang_jax_v2_optimized.py", "tpu_inference_v2_optimized.py"],
)
def test_v2_needs_no_qwix_or_flax(filename):
    """The whole point of migrating v2: no quantization framework needed.

    Checks imports via the AST rather than grepping the text, because these
    files legitimately *mention* qwix in their headers to explain that they do
    not depend on it.
    """
    import ast

    tree = ast.parse((FAMILY / filename).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "qwix" not in imported, sorted(imported)
    assert "flax" not in imported, sorted(imported)


# ---------------------------------------------------------------------------
# tgmm -- the backward-dW pass
#
# Three launch points implement it: JAXBench's `tgmm` and the `tgmm_v2` beside
# each megablox v2 forward.  They were counted as migrated before this file
# exercised them; these tests close that gap.
# ---------------------------------------------------------------------------

TGMM_SHAPE = dict(rows=512, num_groups=8, k=256, n=256)


def test_tgmm_reference_is_the_definition(modules):
    """Guard the reference itself against a ragged-slice reading of tgmm."""
    baseline = modules["baseline"]
    lhs, grad, group_sizes = baseline.create_transpose_inputs(**TGMM_SHAPE)
    out = jax.jit(baseline.grouped_matmul_transpose)(lhs, grad, group_sizes)
    assert out.shape == (TGMM_SHAPE["num_groups"], TGMM_SHAPE["k"],
                         TGMM_SHAPE["n"])
    # Group 0 covers rows [0, rows // num_groups); check it directly.
    size = TGMM_SHAPE["rows"] // TGMM_SHAPE["num_groups"]
    direct = np.asarray(
        jnp.dot(lhs[:size].astype(jnp.float32).T,
                grad[:size].astype(jnp.float32)),
        np.float32,
    )
    np.testing.assert_allclose(np.asarray(out[0], np.float32), direct,
                               rtol=1e-6, atol=1e-6)


def test_jaxbench_tgmm_matches_the_reference(modules):
    """JAXBench takes lhs already transposed, [k, m]."""
    baseline = modules["baseline"]
    lhs, grad, group_sizes = baseline.create_transpose_inputs(**TGMM_SHAPE)
    expected = jax.jit(baseline.grouped_matmul_transpose)(lhs, grad,
                                                          group_sizes)
    actual = modules["jaxbench"].tgmm(lhs.T, grad, group_sizes)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("implementation", V2_WITH_TGMM)
def test_v2_tgmm_matches_the_reference(modules, implementation):
    """megablox v2 takes lhs [m, k] and transposes it inside."""
    baseline = modules["baseline"]
    lhs, grad, group_sizes = baseline.create_transpose_inputs(**TGMM_SHAPE)
    expected = jax.jit(baseline.grouped_matmul_transpose)(lhs, grad,
                                                          group_sizes)
    actual = modules[implementation].tgmm_v2(
        lhs, grad, group_sizes, TGMM_SHAPE["num_groups"],
        preferred_element_type=jnp.float32,
    )
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_tgmm_calling_conventions_differ_by_a_transpose(modules):
    """The trap: same 2-D float lhs, opposite axis order, no error either way.

    JAXBench wants [k, m]; megablox v2 wants [m, k].  Feeding v2 the JAXBench
    orientation at a non-square shape raises, but the shapes alone do not say
    which is which, and at m == k nothing would catch it.
    """
    baseline = modules["baseline"]
    lhs, grad, group_sizes = baseline.create_transpose_inputs(
        rows=512, num_groups=8, k=128, n=256
    )
    correct = modules["maxtext_v2"].tgmm_v2(
        lhs, grad, group_sizes, 8, preferred_element_type=jnp.float32)
    jaxbench = modules["jaxbench"].tgmm(lhs.T, grad, group_sizes)
    np.testing.assert_array_equal(np.asarray(correct), np.asarray(jaxbench))

    # Handing v2 the JAXBench orientation is rejected here only because
    # m != k makes the shapes inconsistent -- not because either kernel
    # checks which axis is which.
    with pytest.raises(Exception):
        modules["maxtext_v2"].tgmm_v2(
            lhs.T, grad, group_sizes, 8, preferred_element_type=jnp.float32)
    assert "[k, m]" in inspect.getdoc(modules["jaxbench"].tgmm)


def test_only_maxtext_and_tokamax_ship_a_transposed_pass(modules):
    """sglang-jax's and tpu-inference's v2 files are forward-only."""
    for name in V2_WITH_TGMM:
        assert hasattr(modules[name], "tgmm_v2"), name
    for name in ("sglang_jax_v2", "tpu_inference_v2"):
        assert not hasattr(modules[name], "tgmm_v2"), name
        assert modules[name].SOURCE["launch_points"] == 1
