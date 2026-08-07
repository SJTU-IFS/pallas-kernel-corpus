"""Small-shape TPU correctness for the SparseCore ragged gather family.

These are the corpus's first **SparseCore** kernels -- they run on the v6e's
SparseCores via ``plsc.VectorSubcoreMesh`` rather than its TensorCore.

The references are not reverse-engineered: each upstream kernel falls back to
plain ``x[indices]`` when no SparseCore is present, so the semantics are stated
in the kernel's own code.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_sparsecore_ragged_gather_tpu.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax.experimental.pallas import tpu as pltpu


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "memory" / "sparsecore_ragged_gather"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="SparseCore ragged gather kernels are Mosaic TPU kernels",
)


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FAMILY / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    return {
        "baseline": load("scg_baseline", "baseline.py"),
        "tokamax": load("scg_tokamax", "tokamax_optimized.py"),
        "tokamax_v2": load("scg_tokamax_v2", "tokamax_v2_optimized.py"),
        "tokamax_gather_reduce": load(
            "scg_tokamax_gr", "tokamax_gather_reduce_optimized.py"
        ),
        "maxtext": load("scg_maxtext", "maxtext_optimized.py"),
    }


GATHER_IMPLEMENTATIONS = ("tokamax", "tokamax_v2", "maxtext")
SHAPES = [(512, 256, 128), (1024, 512, 256)]


def entry_of(modules, name):
    module = modules[name]
    return module.ragged_gather if name == "maxtext" else module.ragged_gather_pallas


def test_sparsecore_is_available():
    """These kernels have a non-SparseCore fallback; make sure we test the real path."""
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None, "no SparseCore: the kernels would silently fall back"
    assert info.num_cores >= 1 and info.num_subcores >= 1


@pytest.mark.parametrize("implementation", GATHER_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_gather_matches_x_at_indices(modules, implementation, shape):
    """The kernels are bit-exact: a gather moves data, it does not compute."""
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = shape
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    expected = np.asarray(
        baseline.ragged_gather(
            built["x"], built["indices"], built["start"], built["end"]
        )
    )
    actual = np.asarray(
        entry_of(modules, implementation)(
            built["x"], built["indices"], built["start"], built["end"]
        )
    )
    # The kernels pad to the SparseCore block size and column tile.
    np.testing.assert_array_equal(actual[:out_rows, :hidden], expected)


@pytest.mark.parametrize("implementation", GATHER_IMPLEMENTATIONS)
def test_gather_is_bit_exact_in_bfloat16(modules, implementation):
    """bf16 is what an MoE layer actually carries, and what the overlap study
    recommends -- so it needs the same bit-exactness guarantee as fp32.

    A gather moves rows without computing on them, so narrowing the element
    type must not introduce any error at all. It also confirms the kernels are
    genuinely dtype-generic rather than fp32-shaped, and that bf16 does not
    quietly take the non-SparseCore fallback.
    """
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = SHAPES[0]
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows,
        dtype=jnp.bfloat16,
    )
    assert built["x"].dtype == jnp.bfloat16
    args = (built["x"], built["indices"], built["start"], built["end"])
    assert pallas_launches(entry_of(modules, implementation), *args) >= 1

    expected = np.asarray(baseline.ragged_gather(*args).astype(jnp.float32))
    actual = np.asarray(
        entry_of(modules, implementation)(*args).astype(jnp.float32)
    )
    np.testing.assert_array_equal(actual[:out_rows, :hidden], expected)


@pytest.mark.parametrize("implementation", GATHER_IMPLEMENTATIONS)
def test_gather_respects_the_live_range(modules, implementation):
    """Only rows in [start, end) are meaningful; those must still be exact."""
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = SHAPES[0]
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    live = out_rows // 2
    actual = np.asarray(
        entry_of(modules, implementation)(
            built["x"], built["indices"], built["start"],
            jnp.array([live], jnp.int32),
        )
    )
    expected = np.asarray(baseline.ragged_gather(
        built["x"], built["indices"], built["start"], built["end"]
    ))
    np.testing.assert_array_equal(actual[:live, :hidden], expected[:live])


def test_maxtext_weighted_gather(modules):
    """MaxText adds an optional weights argument the others do not have."""
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = SHAPES[0]
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    expected = np.asarray(
        baseline.ragged_gather_weighted(
            built["x"], built["indices"], built["weights"]
        )
    )
    actual = np.asarray(
        modules["maxtext"].ragged_gather(
            built["x"], built["indices"], built["start"], built["end"],
            weights=built["weights"], has_weights=True,
        )
    )
    assert np.abs(actual[:out_rows, :hidden] - expected).max() < 1e-5


# `ragged_gather_reduce` refuses its own Pallas path unless `x` is large:
# upstream falls back to XLA when `size(x) * itemsize * 2 < 0.6 *
# vmem_capacity_bytes` (38.4 MiB of fp32 `x` on a 128 MiB-VMEM v6e), and its
# column-partition assertion needs hidden >= 2048 on this device.
GATHER_REDUCE_SHAPE = (8192, 2048, 2048)


def pallas_launches(function, *args, **kwargs) -> int:
    """Count `tpu_custom_call`s in the lowered HLO -- 0 means an XLA fallback."""
    return (
        jax.jit(function, **kwargs).lower(*args).compile().as_text()
    ).count("tpu_custom_call")


@pytest.mark.parametrize("implementation", GATHER_IMPLEMENTATIONS)
def test_gather_actually_reaches_pallas(modules, implementation):
    """Guard against measuring the fallback and calling it a kernel.

    Every kernel here returns plain ``x[indices]`` when no SparseCore is
    present, so a passing correctness check proves nothing on its own about
    which path ran. Pallas lowers to a ``tpu_custom_call``; XLA does not.
    """
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = SHAPES[0]
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    args = (built["x"], built["indices"], built["start"], built["end"])
    assert pallas_launches(entry_of(modules, implementation), *args) >= 1
    # The reference must not be Pallas, or it would not be a valid baseline.
    assert pallas_launches(baseline.ragged_gather, *args) == 0


def test_gather_reduce_falls_back_to_xla_below_the_vmem_threshold(modules):
    """Pin upstream's size heuristic, which is easy to profile straight past.

    ``ragged_gather_reduce_pallas`` checks ``size(x) * itemsize * 2 < 0.6 *
    vmem_capacity_bytes`` and returns its XLA fallback when that holds -- the
    SparseCore path only pays off once ``x`` no longer fits comfortably in
    VMEM. Below the threshold the "kernel" and the baseline are the same code,
    which is exactly what an unguarded benchmark would report as a tie.
    """
    baseline = modules["baseline"]
    entry = modules["tokamax_gather_reduce"].ragged_gather_reduce_pallas

    def launches(num_rows, hidden, out_rows=2048, group=4):
        built = baseline.create_inputs(
            num_rows=num_rows, hidden=hidden, out_rows=out_rows
        )
        return pallas_launches(
            entry,
            built["x"], built["indices"], built["weights"],
            built["valid_rows_mask"], group,
            static_argnums=4,
        )

    vmem = pltpu.get_tpu_info().vmem_capacity_bytes
    assert vmem == 128 * 2**20, "threshold arithmetic below assumes a 128 MiB VMEM"

    # 16 MiB of fp32 x -> 32 MiB < 76.8 MiB: upstream chooses XLA.
    assert launches(4096, 2048) == 0
    # 64 MiB -> 128 MiB > 76.8 MiB: the SparseCore path is taken.
    assert launches(*GATHER_REDUCE_SHAPE[:2]) >= 1


@pytest.mark.parametrize("group", [2, 4])
def test_gather_reduce_matches_reference(modules, group):
    """Checked at a shape large enough that the SparseCore path actually runs."""
    baseline = modules["baseline"]
    num_rows, hidden, out_rows = GATHER_REDUCE_SHAPE
    built = baseline.create_inputs(
        num_rows=num_rows, hidden=hidden, out_rows=out_rows
    )
    args = (
        built["x"], built["indices"], built["weights"],
        built["valid_rows_mask"], group,
    )
    entry = modules["tokamax_gather_reduce"].ragged_gather_reduce_pallas
    assert pallas_launches(entry, *args, static_argnums=4) >= 1

    expected = np.asarray(baseline.ragged_gather_reduce(*args))
    actual = np.asarray(entry(*args))
    assert actual.shape[0] >= expected.shape[0]
    assert np.abs(actual[: expected.shape[0], :hidden] - expected).max() < 1e-4


def test_upstream_documents_the_reference_itself(modules):
    """Guard the claim that the references are upstream's own fallback.

    Each kernel returns ``x[indices]`` when no SparseCore is present.  If that
    fallback ever changes, the corpus reference should be revisited rather than
    silently drifting from what upstream says the op means.
    """
    import inspect

    for name in ("tokamax", "tokamax_v2"):
        source = inspect.getsource(entry_of(modules, name))
        assert "sparse_core" in source
        assert "x[indices]" in source

    # gather_reduce routes through a named helper rather than inlining it.
    module = modules["tokamax_gather_reduce"]
    source = inspect.getsource(module.ragged_gather_reduce_pallas)
    assert "sparse_core" in source
    assert "_fallback_implementation" in source
    assert "x[indices]" in inspect.getsource(module._fallback_implementation)


# ---------------------------------------------------------------------------
# The six kernels migrated on 2026-08-04.  Each carries a shape or dtype
# precondition that upstream does not state, and several fail LOUDLY only
# sometimes -- MaxText's v1 halts the SparseCore at small inputs -- so every
# test below stays inside a validated regime and asserts the preconditions
# from the source rather than by tripping them.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def newer():
    return {
        "ti_gather_v2": load(
            "scg_ti_g2", "tpu_inference_gather_v2_optimized.py"),
        "ti_dense_gather_reduce": load(
            "scg_ti_dgr", "tpu_inference_dense_gather_reduce_optimized.py"),
        "ti_gather_reduce_v2": load(
            "scg_ti_gr2", "tpu_inference_gather_reduce_v2_optimized.py"),
        "mt_gather_reduce": load(
            "scg_mt_gr", "maxtext_gather_reduce_optimized.py"),
        "mt_gather_reduce_v2": load(
            "scg_mt_gr2", "maxtext_gather_reduce_v2_optimized.py"),
        "mt_sc_gather_reduce": load(
            "scg_mt_sc", "maxtext_sc_gather_reduce_optimized.py"),
    }


def _inputs(num_rows, hidden, out_rows, dtype, seed=3):
    keys = jax.random.split(jax.random.key(seed), 3)
    return (
        jax.random.normal(keys[0], (num_rows, hidden), dtype),
        jax.random.randint(keys[1], (out_rows,), 0, num_rows, jnp.int32),
        jax.random.uniform(keys[2], (out_rows,), jnp.float32),
        jnp.ones((out_rows,), jnp.bool_),
    )


def _close(got, want, tol=2e-2):
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    rows, cols = min(g.shape[0], w.shape[0]), min(g.shape[-1], w.shape[-1])
    g, w = g[:rows, :cols], w[:rows, :cols]
    return float(np.max(np.abs(g - w))) <= tol * max(1.0, float(np.max(np.abs(w))))


def test_tpu_inference_gather_v2_is_bit_exact(modules, newer):
    """The same contract, and the same bit-exactness, as the Tokamax gathers."""
    x, indices, _, _ = _inputs(512, 512, 256, jnp.float32)
    start, end = jnp.array([0], jnp.int32), jnp.array([256], jnp.int32)
    got = newer["ti_gather_v2"].ragged_gather_v2(x, indices, start, end)
    want = modules["baseline"].ragged_gather(x, indices, start, end)
    jax.block_until_ready(got)
    np.testing.assert_array_equal(
        np.asarray(got)[:256, :512], np.asarray(want))


def test_tpu_inference_dense_gather_reduce_matches_the_reference(modules, newer):
    """A different contract: no valid_rows_mask, and 2-D topk_weights.

    ``out_rows`` must be a multiple of ``row_chunk_size * num_cores *
    num_subcores`` = 16384 at the defaults, or ``is_compatible`` sends the call
    to ``_jax_fallback`` and the Pallas kernel never runs.  The companion test
    below pins that, because a fallback compared against the JAX reference
    agrees trivially -- this test read as a bit-exact pass at out_rows=256 while
    exercising no kernel at all.
    """
    x, indices, weights, valid = _inputs(16384, 256, 16384, jnp.float32)
    assert newer["ti_dense_gather_reduce"].is_compatible(x, indices, 4)
    got = newer["ti_dense_gather_reduce"].dense_gather_reduce(
        x, indices, weights.reshape(4096, 4), 4)
    want = modules["baseline"].ragged_gather_reduce(
        x, indices, weights, valid, 4)
    jax.block_until_ready(got)
    assert _close(got, want)


@pytest.mark.parametrize("implementation,num_rows,hidden,out_rows,dtype", [
    # tpu-inference v2 falls back to XLA whenever `size(x) * itemsize * 2` is
    # under 60% of TensorCore VMEM, so a small shape validates nothing; and
    # hidden must be >= 2048 or its own partitioning assertion fires.
    pytest.param("ti_gather_reduce_v2", 16384, 2048, 4096, jnp.bfloat16,
                 id="ti_v2"),
    pytest.param("mt_gather_reduce_v2", 1024, 2048, 256, jnp.bfloat16,
                 id="mt_v2"),
    pytest.param("mt_gather_reduce", 1024, 1024, 1024, jnp.bfloat16,
                 id="mt_v1"),
])
def test_gather_reduce_variants_match_the_reference(
    modules, newer, implementation, num_rows, hidden, out_rows, dtype
):
    x, indices, weights, valid = _inputs(num_rows, hidden, out_rows, dtype)
    got = newer[implementation].ragged_gather_reduce(
        x, indices, weights, valid, 4)
    want = modules["baseline"].ragged_gather_reduce(
        x, indices, weights, valid, 4)
    jax.block_until_ready(got)
    assert _close(got, want)


def test_maxtext_sc_gather_reduce_matches_the_reference(modules, newer):
    """Its own tiling defaults do not fit this v6e's SparseCore VMEM.

    `col_chunk_size` defaults to 3584, which overflows the 256 KiB SparseCore
    VMEM here; 512 fits.  The row constraint is separate: M must be divisible
    by row_chunk_size * num_cores * num_subcores = 16384.
    """
    x, indices, weights, valid = _inputs(16384, 3584, 16384, jnp.bfloat16)
    got = newer["mt_sc_gather_reduce"].sc_gather_reduce(
        x, indices, weights, reduce_group_size=4,
        row_chunk_size=512, col_chunk_size=512)
    want = modules["baseline"].ragged_gather_reduce(
        x, indices, weights, valid, 4)
    jax.block_until_ready(got)
    assert _close(got, want)


def test_maxtext_sc_gather_reduce_is_bf16_only_despite_its_message(newer):
    """The error text says "f32 or bf16"; the check is bf16 only."""
    import inspect

    source = inspect.getsource(newer["mt_sc_gather_reduce"].sc_gather_reduce)
    assert "op.dtype must be f32 or bf16" in source
    assert "if op.dtype != jnp.bfloat16:" in source

    x, indices, weights, _ = _inputs(16384, 3584, 16384, jnp.float32)
    with pytest.raises(ValueError, match="must be f32 or bf16"):
        newer["mt_sc_gather_reduce"].sc_gather_reduce(
            x, indices, weights, reduce_group_size=4,
            row_chunk_size=512, col_chunk_size=512)


def test_maxtext_sc_gather_reduce_group_is_bounded_by_lanes_over_packing(newer):
    """reduce_group_size=8 -- upstream's own test value -- cannot work here.

    The output BlockSpec row dim is ``(num_lanes // group) // packing``.  With
    8 SparseCore lanes and bf16 (packing 2) a group of 8 makes that **zero**,
    and Pallas rejects the block rather than the argument.
    """
    import inspect

    source = inspect.getsource(newer["mt_sc_gather_reduce"].sc_gather_reduce)
    assert "out_rows_per_step = row_subchunk_size // reduce_group_size" in source
    assert "(out_rows_per_step // packing, col_chunk_size)" in source

    x, indices, weights, _ = _inputs(16384, 3584, 16384, jnp.bfloat16)
    with pytest.raises(ValueError):
        newer["mt_sc_gather_reduce"].sc_gather_reduce(
            x, indices, weights, reduce_group_size=8,
            row_chunk_size=512, col_chunk_size=512)


def test_maxtext_v1_hardcodes_the_partitioning_that_gives_it_a_shape_floor(newer):
    """Why the v1 ledger row records `indices.shape[0] >= 1024`.

    v1 fixes ``num_column_partitions = 8`` and derives the rest; v2 computes
    both from a cost model.  Below the floor v1 returns a wrong answer at some
    shapes and halts the SparseCore at others, so this asserts the cause from
    the source instead of triggering it -- a crash here would take the whole
    test process down.
    """
    import inspect

    v1 = inspect.getsource(newer["mt_gather_reduce"].ragged_gather_reduce)
    assert "num_column_partitions = 8" in v1
    v2 = inspect.getsource(newer["mt_gather_reduce_v2"])
    assert "_calculate_num_column_partitions" in v2
    assert "num_row_partitions <= num_simd_lanes" in v2


def test_maxtext_v2_and_tpu_inference_v2_are_a_diverged_vendored_pair():
    """85% AST-identical, so measured rather than assumed -- as for gdn v3."""
    import ast

    def defs(path):
        text = (FAMILY / path).read_text()
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body = ast.parse(ast.get_source_segment(text, node))
                for sub in ast.walk(body):
                    if isinstance(sub, (ast.FunctionDef, ast.ClassDef,
                                        ast.Module)):
                        if (sub.body and isinstance(sub.body[0], ast.Expr)
                                and isinstance(sub.body[0].value, ast.Constant)
                                and isinstance(sub.body[0].value.value, str)):
                            sub.body = sub.body[1:] or [ast.Pass()]
                out[node.name] = ast.dump(ast.fix_missing_locations(body))
        return out

    a = defs("maxtext_gather_reduce_v2_optimized.py")
    b = defs("tpu_inference_gather_reduce_v2_optimized.py")
    shared = set(a) & set(b)
    identical = sum(1 for n in shared if a[n] == b[n])
    assert len(shared) >= 12, sorted(shared)
    # Same code, diverged: not the verbatim copy the streamindex pair is.
    assert 0.7 <= identical / len(shared) < 1.0, (identical, len(shared))


def test_core_map_helper_was_inlined_and_renamed(newer):
    """`core_map_helper.kernel` would collide with the corpus `kernel` export."""
    for name in ("ti_gather_v2", "ti_gather_reduce_v2"):
        module = newer[name]
        assert hasattr(module, "core_map_kernel")
        assert module.kernel is not module.core_map_kernel


# ---------------------------------------------------------------------------
# Every gather-reduce here has an XLA fallback, and the corpus's own standing
# rule applies: a passing correctness check does not prove the kernel ran.
# `test_gather_actually_reaches_pallas` covers the plain gathers; these cover
# the reduce variants, which were originally tested at shapes where two of
# them silently took the fallback and agreed with the reference trivially.
# ---------------------------------------------------------------------------


def _pallas_launches(function, args):
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    return count_pallas_launches(function, args)


def test_dense_gather_reduce_reaches_pallas_only_above_its_row_wave(newer):
    """`is_compatible` gates on idx.size % 16384; below it, no kernel runs."""
    import functools

    module = newer["ti_dense_gather_reduce"]
    call = functools.partial(module.dense_gather_reduce, reduce_group_size=4)

    small = _inputs(1024, 256, 256, jnp.float32)
    assert not module.is_compatible(small[0], small[1], 4)
    assert _pallas_launches(
        call, (small[0], small[1], small[2].reshape(64, 4))) == 0

    big = _inputs(16384, 256, 16384, jnp.float32)
    assert module.is_compatible(big[0], big[1], 4)
    assert _pallas_launches(
        call, (big[0], big[1], big[2].reshape(4096, 4))) == 1


def test_gather_reduce_v2_reaches_pallas_only_above_the_vmem_threshold(newer):
    """tpu-inference v2 keeps small problems on the TensorCore.

    Same fallback the Tokamax gather-reduce has, and the reason its test shape
    had to grow: at 512x512 the call never entered Pallas.
    """
    import functools

    call = functools.partial(
        newer["ti_gather_reduce_v2"].ragged_gather_reduce, reduce_group_size=4)
    small = _inputs(512, 512, 256, jnp.float32)
    assert _pallas_launches(call, small) == 0
    big = _inputs(16384, 2048, 4096, jnp.bfloat16)
    assert _pallas_launches(call, big) == 1


def test_maxtext_gather_reduce_variants_reach_pallas(newer):
    """MaxText's two have no VMEM fallback: they enter Pallas at test shapes."""
    import functools

    for name, args in (
        ("mt_gather_reduce_v2", _inputs(1024, 2048, 256, jnp.bfloat16)),
        ("mt_gather_reduce", _inputs(1024, 1024, 1024, jnp.bfloat16)),
    ):
        call = functools.partial(
            newer[name].ragged_gather_reduce, reduce_group_size=4)
        assert _pallas_launches(call, args) == 1, name
