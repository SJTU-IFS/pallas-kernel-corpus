"""Small-shape TPU correctness for the paged KV-cache update family.

This is a pure data-movement kernel, so the bar is higher than a cosine
threshold: the output must be *bitwise identical* to the pure-JAX reference.
Any difference at all would mean bytes were moved wrongly, not rounded.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_kv_cache_update_tpu.py -q
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "memory" / "kv_cache_update"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="KV-cache update kernels are Mosaic TPU kernels",
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
        "baseline": load("kv_baseline", "baseline.py"),
        "tpu_inference": load("kv_tpu_inference", "tpu_inference_optimized.py"),
        "sglang_jax": load("kv_sglang_jax", "sglang_jax_optimized.py"),
    }


IMPLEMENTATIONS = ("tpu_inference", "sglang_jax")

BASE_CONFIG = dict(
    total_num_tokens=1024,
    num_combined_kv_heads=16,
    head_dim=128,
    total_num_pages=256,
    page_size=32,
)


@contextlib.contextmanager
def maybe_mesh(implementation):
    """sglang-jax always shard_maps, so it needs a mesh even on one device."""
    if implementation != "sglang_jax":
        yield
        return
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1), ("tensor",))
    with jax.sharding.set_mesh(mesh):
        yield


def call(implementation, modules, inputs, page_size, num_slices_per_block=8):
    new_kv, slices, kv_cache, num_slices = inputs
    module = modules[implementation]
    extra = (
        {"kv_partition_axis": "tensor"} if implementation == "sglang_jax" else {}
    )
    with maybe_mesh(implementation):
        return jax.block_until_ready(
            module.kv_cache_update(
                new_kv,
                slices,
                kv_cache,
                num_slices,
                page_size=page_size,
                num_slices_per_block=num_slices_per_block,
                **extra,
            )
        )


def reference(modules, inputs, page_size):
    new_kv, slices, kv_cache, num_slices = inputs
    return jax.jit(
        modules["baseline"].kv_cache_update, static_argnames="max_slice_len"
    )(new_kv, slices, kv_cache, num_slices, max_slice_len=page_size)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("num_slices", [8, 24, 32])
def test_matches_reference_bitwise(modules, implementation, num_slices):
    """A copy kernel that is only approximately right is wrong."""
    baseline = modules["baseline"]
    build = lambda: baseline.create_inputs(  # noqa: E731
        **BASE_CONFIG, num_slices=num_slices, padded_num_slices=32
    )
    *inputs, page_size = build()
    expected = np.asarray(reference(modules, inputs, page_size))
    *inputs, page_size = build()  # the kernels donate kv_cache
    actual = np.asarray(call(implementation, modules, inputs, page_size))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_ignores_padded_slice_columns(modules, implementation):
    """Columns at or past num_slices must not be written."""
    baseline = modules["baseline"]
    build = lambda: baseline.create_inputs(  # noqa: E731
        **BASE_CONFIG, num_slices=32, padded_num_slices=32
    )
    *inputs, page_size = build()
    new_kv, slices, kv_cache, _ = inputs
    live = jnp.array([8], jnp.int32)

    untouched = np.asarray(kv_cache)
    expected = np.asarray(
        reference(modules, (new_kv, slices, kv_cache, live), page_size)
    )
    *fresh, page_size = build()
    actual = np.asarray(
        call(implementation, modules, (fresh[0], fresh[1], fresh[2], live), page_size)
    )
    np.testing.assert_array_equal(actual, expected)
    # Only the first 8 pages should differ from the original cache.
    changed = np.any(actual != untouched, axis=(1, 2))
    assert changed[: 8 * BASE_CONFIG["page_size"]].any()
    assert not changed[8 * BASE_CONFIG["page_size"] :].any()


def test_baseline_actually_writes_something(modules):
    """Guard against a reference that silently returns the cache unchanged."""
    baseline = modules["baseline"]
    *inputs, page_size = baseline.create_inputs(**BASE_CONFIG, num_slices=32)
    before = np.asarray(inputs[2])
    after = np.asarray(reference(modules, inputs, page_size))
    assert not np.array_equal(before, after)


def test_kernels_donate_the_cache(modules):
    """Document the donation trap: the input cache is consumed by the call."""
    baseline = modules["baseline"]
    *inputs, page_size = baseline.create_inputs(**BASE_CONFIG, num_slices=32)
    kv_cache = inputs[2]
    call("tpu_inference", modules, inputs, page_size)
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(kv_cache)


# ---------------------------------------------------------------------------
# DeepSeek-V4 compressor  (experimental/deepseek_v4, 2 launch points)
#
# Flattened and running on TPU, but NOT validated and NOT counted as migrated:
# the Pallas cache packing has no published host-side readback, and the only
# pure-JAX implementation upstream uses a different layout.  See the ledger
# entry for tpu_inference_dsv4.  These tests pin what is actually established,
# so the open question stays visible and the two contracts cannot quietly be
# treated as one.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dsv4():
    return {
        "kernel": load("kv_dsv4", "tpu_inference_dsv4_optimized.py"),
        "reference": load("kv_dsv4_ref", "dsv4_reference.py"),
    }


def test_dsv4_has_two_launch_points_and_is_standalone(dsv4):
    import ast

    module = dsv4["kernel"]
    source = (FAMILY / "tpu_inference_dsv4_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 2
    assert module.SOURCE["launch_points"] == 2
    assert module.SOURCE["contract"] == "dsv4_compress_and_store"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in (
                "tpu_inference", "tokamax", "maxtext", "sgl_jax"
            ), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in (
                    "tpu_inference", "tokamax", "maxtext", "sgl_jax"
                ), alias.name


def test_dsv4_reference_is_pallas_free(dsv4):
    """The pure-JAX twin is kept as a reference, so it must contain no Pallas."""
    source = (FAMILY / "dsv4_reference.py").read_text()
    assert "pallas" not in source.replace("pallas_mosaic_tpu", "")
    assert dsv4["reference"].SOURCE["launch_points"] == 0
    assert dsv4["reference"].SOURCE["kind"] == "reference"


def test_dsv4_pallas_and_pure_jax_use_different_cache_layouts(dsv4):
    """This is why the pure-JAX twin is not a drop-in reference.

    If these two ever agree, the blocker recorded in the ledger is gone and the
    kernels can be validated directly -- so this test failing is good news, not
    a regression.
    """
    kernel, reference = dsv4["kernel"], dsv4["reference"]
    cfgs = kernel.Configs.make(
        kernel._select_mode(512, True), size_n=128, physical_page_size=64,
        rms_eps=1e-6, tile_n=4, head_dim=512, rope_head_dim=64,
        compress_ratio=4, quant_block=64,
    )
    pallas_shape = tuple(cfgs.cache_shape(8))
    pure_jax_shape = tuple(
        reference.shared_sparse_cache_shape(8, 64, 512 - 64, 64, 64)
    )
    assert pallas_shape != pure_jax_shape
    # The last axis is the concrete divergence: one lane vs a packed record.
    assert pallas_shape[-1] == 128
    assert pure_jax_shape[-1] == 640


def test_dsv4_renamed_names_survive_the_flatten(dsv4):
    """Three classes and one function are renamed to avoid collisions."""
    module = dsv4["kernel"]
    assert module.SOURCE["renamed_on_flatten"] == {
        "Configs": "ProjConfigs",
        "Dimensions": "ProjDimensions",
        "TileSizes": "ProjTileSizes",
        "kernel": "proj_and_save_state_kernel",
    }
    # The compress-side Configs keeps its name; the proj-side one is renamed.
    assert hasattr(module, "Configs")
    assert hasattr(module, "ProjConfigs")
    assert module.Configs is not module.ProjConfigs
    assert hasattr(module, "proj_and_save_state_kernel")
    # And `kernel` is the corpus entry point, not upstream's inner kernel.
    assert module.kernel is module.compressor_forward


def test_dsv4_proj_requires_token_count_to_be_a_multiple_of_the_tile(dsv4):
    """A precondition with teeth: violating it page-faults the whole chip.

    `proj_and_save_state` hard-codes tile_n = 128 and its pallas_call sets
    `disable_bounds_checks=True`, so a num_tokens that is not a multiple of 128
    makes the tile DMA read past the end of `hidden_states`.  That does not
    raise -- it kills the process with a TPU page fault.  This test asserts the
    precondition from the source rather than by triggering it.
    """
    import inspect

    source = inspect.getsource(dsv4["kernel"].proj_and_save_state)
    assert "tile_n = 128" in source
    assert "disable_bounds_checks=True" in source


# ---------------------------------------------------------------------------
# sglang-jax ships TWO launch points, and only one was reachable from the tests
# until 2026-08-05.  `kv_cache_update` shard_maps an inner wrapper; the
# separate top-level `kv_cache_update_impl` takes **5-D** inputs and flattens
# them itself.  Both were counted as migrated; only the shard_map path ran.
# ---------------------------------------------------------------------------


def test_sglang_impl_takes_five_dimensional_inputs(modules):
    """The second entry point, and a different shape contract from the first."""
    baseline = modules["baseline"]
    *inputs, page_size = baseline.create_inputs(
        **BASE_CONFIG, num_slices=24, padded_num_slices=32
    )
    expected = np.asarray(reference(modules, inputs, page_size))

    *inputs, page_size = baseline.create_inputs(
        **BASE_CONFIG, num_slices=24, padded_num_slices=32
    )
    new_kv, slices, kv_cache, num_slices = inputs
    tokens, heads, head_dim = new_kv.shape
    slots = kv_cache.shape[0]

    # [T, H, D] -> [1, T, 1, H, D] and [S, H, D] -> [pages, page_size, 1, H, D];
    # the impl multiplies the first two and the middle two axes back together.
    new_kv_5d = new_kv.reshape(1, tokens, 1, heads, head_dim)
    cache_5d = kv_cache.reshape(
        slots // page_size, page_size, 1, heads, head_dim
    )
    actual = modules["sglang_jax"].kv_cache_update_impl(
        new_kv_5d, slices, cache_5d, num_slices,
        page_size=page_size, num_slices_per_block=8,
    )
    actual = np.asarray(jax.block_until_ready(actual)).reshape(
        slots, heads, head_dim
    )
    np.testing.assert_array_equal(actual, expected)


def test_sglang_both_launch_points_reach_pallas(modules):
    """Each entry point launches the same kernel body from its own site.

    `tools/launch_coverage.py` cannot tell the two apart -- it identifies a
    launch by the kernel body's file and qualified name, and both sites pass
    `kv_cache_update_kernel` -- so this test is what pins that the second one
    is reachable at all.
    """
    import functools
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    baseline = modules["baseline"]
    *inputs, page_size = baseline.create_inputs(
        **BASE_CONFIG, num_slices=24, padded_num_slices=32
    )
    new_kv, slices, kv_cache, num_slices = inputs
    tokens, heads, head_dim = new_kv.shape
    slots = kv_cache.shape[0]

    impl = functools.partial(
        modules["sglang_jax"].kv_cache_update_impl,
        page_size=page_size, num_slices_per_block=8,
    )
    assert count_pallas_launches(
        impl,
        (new_kv.reshape(1, tokens, 1, heads, head_dim), slices,
         kv_cache.reshape(slots // page_size, page_size, 1, heads, head_dim),
         num_slices),
    ) == 1

    with maybe_mesh("sglang_jax"):
        wrapped = functools.partial(
            modules["sglang_jax"].kv_cache_update,
            page_size=page_size, num_slices_per_block=8,
            kv_partition_axis="tensor",
        )
        assert count_pallas_launches(
            wrapped, (new_kv, slices, kv_cache, num_slices)) == 1


# ---------------------------------------------------------------------------
# The DeepSeek-V4 compressor: two Pallas launch points that were carried
# unvalidated because their packed cache had no host-side readback.  The layout
# is now measured (see baseline.read_dsv4_state / read_dsv4_record) and both
# kernels check out.  These tests pin the measurement as much as the kernels:
# if the layout ever changes, the round-trip assertions fail first and say so.
# ---------------------------------------------------------------------------

DSV4_TOKENS, DSV4_RATIO, DSV4_QUANT = 128, 4, 64
DSV4_HEAD_DIM, DSV4_ROPE = 512, 64
DSV4_NOPE = DSV4_HEAD_DIM - DSV4_ROPE


@pytest.fixture(scope="module")
def dsv4():
    kernel = load("kv_dsv4_k", "tpu_inference_dsv4_optimized.py")
    reference = load("kv_dsv4_r", "dsv4_reference.py")
    mode = kernel._select_mode(DSV4_HEAD_DIM, True)

    def configs(physical_page_size):
        return kernel.Configs.make(
            mode, size_n=DSV4_TOKENS, physical_page_size=physical_page_size,
            rms_eps=1e-6, tile_n=4, head_dim=DSV4_HEAD_DIM,
            rope_head_dim=DSV4_ROPE, compress_ratio=DSV4_RATIO,
            quant_block=DSV4_QUANT,
        )

    cfgs = configs(configs(64).state_rows_per_token * 4)
    return {"kernel": kernel, "reference": reference, "cfgs": cfgs}


def _dsv4_project(dsv4, seed=0, num_pages=32):
    """Run proj_and_save_state and return everything needed to check it."""
    cfgs = dsv4["cfgs"]
    state_dim, width = 2 * cfgs.state_width, cfgs.state_width
    rows, page_size = cfgs.state_rows_per_token, cfgs.dims.physical_page_size
    rng = np.random.default_rng(seed)

    hidden = rng.standard_normal((DSV4_TOKENS, 2048), np.float32)
    wgate = rng.standard_normal((2048, state_dim), np.float32) * 0.05
    ape = rng.standard_normal((DSV4_RATIO, width), np.float32)
    positions = np.arange(DSV4_TOKENS, dtype=np.int32)
    # slot_mapping is in PHYSICAL ROWS, so tokens must be `rows` apart.
    slots = (np.arange(DSV4_TOKENS, dtype=np.int32) * rows) % (
        num_pages * page_size)

    cache = dsv4["kernel"].proj_and_save_state(
        hidden_states=jnp.asarray(hidden), wkv_wgate=jnp.asarray(wgate),
        ape=jnp.asarray(ape), positions=jnp.asarray(positions),
        slot_mapping=jnp.asarray(slots),
        cache=jnp.zeros(cfgs.cache_shape(num_pages), jnp.uint8),
        compress_ratio=DSV4_RATIO,
    )
    jax.block_until_ready(cache)
    return dict(cache=cache, hidden=hidden, wgate=wgate, ape=ape,
                positions=positions, slots=slots, num_pages=num_pages)


def test_dsv4_state_layout_round_trips_small_integers_exactly(dsv4):
    """The layout measurement, stated as a property no wrong reading satisfies.

    Sub-slot b of the SLOT_PACK axis holds byte b of every float32.  Feeding
    integers that are exact in bfloat16 -- so the kernel's bf16 matmul does not
    round them -- must return them unchanged.
    """
    kernel, cfgs = dsv4["kernel"], dsv4["cfgs"]
    baseline = load("kv_baseline_dsv4", "baseline.py")
    state_dim = 2 * cfgs.state_width
    rng = np.random.default_rng(1)

    hidden = np.zeros((DSV4_TOKENS, state_dim), np.float32)
    hidden[0] = rng.integers(-128, 129, state_dim).astype(np.float32)
    slots = np.full((DSV4_TOKENS,), -1, np.int32)
    slots[0] = 0

    cache = kernel.proj_and_save_state(
        hidden_states=jnp.asarray(hidden),
        wkv_wgate=jnp.asarray(np.eye(state_dim, dtype=np.float32)),
        ape=jnp.zeros((DSV4_RATIO, cfgs.state_width), jnp.float32),
        positions=jnp.zeros((DSV4_TOKENS,), jnp.int32),
        slot_mapping=jnp.asarray(slots),
        cache=jnp.zeros(cfgs.cache_shape(8), jnp.uint8),
        compress_ratio=DSV4_RATIO,
    )
    got = np.asarray(baseline.read_dsv4_state(
        cache, 0, cfgs.state_rows_per_token), np.float32)
    np.testing.assert_array_equal(got, hidden[0])


def test_dsv4_slot_mapping_is_in_physical_rows_not_tokens(dsv4):
    """A trap with no error attached: token indices silently overlap.

    Slot S occupies rows [S, S + state_rows_per_token) of page
    S // physical_page_size, so consecutive tokens must be spaced
    state_rows_per_token apart.  Passing 0, 1, 2, ... makes each token overwrite
    all but one row of the previous one, and nothing complains.
    """
    kernel, cfgs = dsv4["kernel"], dsv4["cfgs"]
    rows, page_size = cfgs.state_rows_per_token, cfgs.dims.physical_page_size
    state_dim = 2 * cfgs.state_width

    def written_rows(slot):
        hidden = np.zeros((DSV4_TOKENS, state_dim), np.float32)
        hidden[0] = 1.0
        slots = np.full((DSV4_TOKENS,), -1, np.int32)
        slots[0] = slot
        out = np.asarray(kernel.proj_and_save_state(
            hidden_states=jnp.asarray(hidden),
            wkv_wgate=jnp.asarray(np.eye(state_dim, dtype=np.float32)),
            ape=jnp.zeros((DSV4_RATIO, cfgs.state_width), jnp.float32),
            positions=jnp.zeros((DSV4_TOKENS,), jnp.int32),
            slot_mapping=jnp.asarray(slots),
            cache=jnp.zeros(cfgs.cache_shape(8), jnp.uint8),
            compress_ratio=DSV4_RATIO))
        touched = np.argwhere(out != 0)
        return sorted(set(touched[:, 0].tolist())), sorted(
            set(touched[:, 1].tolist()))

    for slot in (0, rows, 2 * rows, page_size, page_size + rows):
        pages, rws = written_rows(slot)
        assert pages == [slot // page_size], slot
        assert rws == list(range(slot % page_size,
                                 slot % page_size + rows)), slot

    # Consecutive slot numbers overlap, which is what makes the trap silent.
    _, rows_0 = written_rows(0)
    _, rows_1 = written_rows(1)
    assert len(set(rows_0) & set(rows_1)) == rows - 1


def test_dsv4_proj_and_save_state_matches_the_reference(dsv4):
    """The projection plus the scatter, over every written slot.

    The kernel's matmul runs in bf16 on the MXU, so the tolerance is one bf16
    ulp at the peak value -- comparing against exact float32 is what made this
    look like a layout error while the readback was being worked out.
    """
    baseline = load("kv_baseline_dsv4b", "baseline.py")
    cfgs = dsv4["cfgs"]
    run = _dsv4_project(dsv4)
    kv_score = run["hidden"] @ run["wgate"]
    peak = float(np.max(np.abs(kv_score)))

    worst = 0.0
    for token, slot in enumerate(run["slots"]):
        want = np.asarray(baseline.dsv4_expected_state(
            jnp.asarray(kv_score[token]), int(run["positions"][token]),
            jnp.asarray(run["ape"]), cfgs.state_width, DSV4_RATIO), np.float32)
        got = np.asarray(baseline.read_dsv4_state(
            run["cache"], int(slot), cfgs.state_rows_per_token), np.float32)
        worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst <= peak * 2 ** -7, f"{worst} exceeds a bf16 ulp at {peak}"


def test_dsv4_compress_norm_rope_store_matches_the_pure_jax_twin(dsv4):
    """The boundary compress-and-store, against upstream's own pure-JAX path.

    Both are fed the *same* state -- the kernel reads it from its packed cache,
    the reference gets it via the measured readback -- so only the record
    layout differs, and the dequantized records must agree bit for bit.
    """
    baseline = load("kv_baseline_dsv4c", "baseline.py")
    kernel, reference, cfgs = dsv4["kernel"], dsv4["reference"], dsv4["cfgs"]
    rows, page_size = cfgs.state_rows_per_token, cfgs.dims.physical_page_size
    block = cfgs.state_block_size
    run = _dsv4_project(dsv4, seed=2)
    num_pages = run["num_pages"]
    rng = np.random.default_rng(3)

    state = np.zeros((num_pages, block, 2 * cfgs.state_width), np.float32)
    for page in range(num_pages):
        for within in range(block):
            state[page, within] = np.asarray(baseline.read_dsv4_state(
                run["cache"], page * page_size + within * rows, rows),
                np.float32)

    norm_weight = rng.standard_normal(DSV4_HEAD_DIM, np.float32)
    cos_sin = rng.standard_normal(
        (DSV4_TOKENS + DSV4_RATIO, DSV4_ROPE), np.float32)
    block_table = np.arange(num_pages, dtype=np.int32).reshape(1, num_pages)
    tok2req = np.zeros(DSV4_TOKENS, np.int32)
    kv_slot, token = 0, 3            # (3 + 1) % 4 == 0, a compression boundary
    kv_slots = np.full(DSV4_TOKENS, -1, np.int32)
    kv_slots[token] = kv_slot

    out, _ = kernel.compress_norm_rope_store(
        jnp.array(run["cache"], copy=True), jnp.asarray(run["positions"]),
        jnp.asarray(block_table), jnp.asarray(tok2req), jnp.asarray(kv_slots),
        jnp.asarray(norm_weight),
        rope_cache=jnp.zeros(cfgs.rope_cache_shape(num_pages), jnp.uint8),
        cos_sin_cache=jnp.asarray(cos_sin), compress_ratio=DSV4_RATIO,
        overlap=True, quant_block=DSV4_QUANT, rms_eps=1e-6,
    )
    jax.block_until_ready(out)
    got = np.asarray(baseline.read_dsv4_record(
        out, kv_slot, DSV4_NOPE, DSV4_QUANT), np.float32)

    ref_slots = np.array(
        [(t // block) * block + (t % block) for t in range(DSV4_TOKENS)],
        np.int32)
    want_cache = reference.compress_norm_rope_store(
        cache=jnp.zeros(reference.shared_sparse_cache_shape(
            num_pages, 64, DSV4_NOPE, DSV4_ROPE, DSV4_QUANT), jnp.uint8),
        state_cache=jnp.asarray(state), positions=jnp.asarray(run["positions"]),
        slot_mapping=jnp.asarray(ref_slots),
        block_table=jnp.asarray(block_table),
        token_to_req_indices=jnp.asarray(tok2req),
        kv_slot_mapping=jnp.asarray(kv_slots),
        rms_weight=jnp.asarray(norm_weight), cos_sin_cache=jnp.asarray(cos_sin),
        state_block_size=block, head_dim=DSV4_HEAD_DIM,
        rope_head_dim=DSV4_ROPE, compress_ratio=DSV4_RATIO, overlap=True,
        rms_eps=1e-6, quant_block=DSV4_QUANT,
    )
    nope, _, scales = reference.unpack_sparse_kv_cache(
        want_cache, DSV4_NOPE, DSV4_ROPE, DSV4_QUANT)
    nope = np.asarray(nope, np.float32).reshape(-1, DSV4_NOPE)[kv_slot]
    scales = np.asarray(scales, np.float32).reshape(
        -1, DSV4_NOPE // DSV4_QUANT)[kv_slot]
    want = (nope.reshape(-1, DSV4_QUANT) * scales[:, None]).reshape(-1)

    assert np.max(np.abs(want)) > 0, "reference record is all zeros"
    np.testing.assert_array_equal(got, want)


def test_dsv4_proj_reaches_pallas(dsv4):
    """The standing rule: a passing check does not prove the kernel ran."""
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    cfgs = dsv4["cfgs"]
    state_dim = 2 * cfgs.state_width
    hidden = np.zeros((DSV4_TOKENS, state_dim), np.float32)
    slots = np.full((DSV4_TOKENS,), -1, np.int32)
    slots[0] = 0
    launched = count_pallas_launches(
        lambda h, w, a, p, s, c: dsv4["kernel"].proj_and_save_state(
            hidden_states=h, wkv_wgate=w, ape=a, positions=p,
            slot_mapping=s, cache=c, compress_ratio=DSV4_RATIO),
        (jnp.asarray(hidden),
         jnp.asarray(np.eye(state_dim, dtype=np.float32)),
         jnp.zeros((DSV4_RATIO, cfgs.state_width), jnp.float32),
         jnp.zeros((DSV4_TOKENS,), jnp.int32), jnp.asarray(slots),
         jnp.zeros(cfgs.cache_shape(8), jnp.uint8)),
    )
    assert launched == 1
