"""Small-shape TPU correctness for the ragged-paged-attention family.

Two contracts live here and are tested differently:

``rpa_v2``  checked against ``baseline.rpa_v2``, an independent pure-JAX
            reference in this corpus.
``rpa_v3``  checked against tpu-inference v3's upstream pure-JAX reference.
            sglang-jax's kernel takes two extra required arguments and its own
            bundled reference implements a different signature again, so it is
            validated against tpu-inference's reference: a genuine
            cross-repository check.  See baseline.py.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_ragged_paged_attention_tpu.py -q
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "attention" / "ragged_paged_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="ragged paged attention kernels are Mosaic TPU kernels",
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
        "baseline": load("rpa_baseline", "baseline.py"),
        "jaxbench": load("rpa_jaxbench", "jaxbench_optimized.py"),
        "tpu_inference": load("rpa_tpu_inference", "tpu_inference_optimized.py"),
        "tpu_inference_v3": load("rpa_ti_v3", "tpu_inference_v3_optimized.py"),
        "sglang_jax": load("rpa_sg_v3", "sglang_jax_optimized.py"),
    }


def cosine(actual, expected) -> float:
    a = np.asarray(actual, np.float32).ravel()
    e = np.asarray(expected, np.float32).ravel()
    return float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-12))


V2_IMPLEMENTATIONS = ("jaxbench", "tpu_inference")
V3_IMPLEMENTATIONS = ("tpu_inference_v3", "sglang_jax")

V2_SHAPES = [
    pytest.param(
        dict(
            max_num_batched_tokens=128,
            max_num_seqs=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            pages_per_seq=8,
        ),
        id="gqa4",
    ),
    pytest.param(
        dict(
            max_num_batched_tokens=256,
            max_num_seqs=8,
            num_q_heads=16,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
            pages_per_seq=16,
        ),
        id="gqa8",
    ),
    pytest.param(
        dict(
            max_num_batched_tokens=256,
            max_num_seqs=4,
            num_q_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            pages_per_seq=8,
        ),
        id="mqa",
    ),
]


@pytest.mark.parametrize("implementation", V2_IMPLEMENTATIONS)
@pytest.mark.parametrize("config", V2_SHAPES)
def test_v2_matches_pure_jax_reference(modules, implementation, config):
    baseline = modules["baseline"]
    inputs = baseline.create_inputs(**config)
    expected = jax.jit(baseline.rpa_v2)(*inputs)
    actual = jax.jit(modules[implementation].kernel)(*inputs)
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


@pytest.mark.parametrize("implementation", V2_IMPLEMENTATIONS)
def test_v2_handles_uneven_kv_lengths(modules, implementation):
    """Sequences of different KV length are the point of a *ragged* kernel."""
    baseline = modules["baseline"]
    q, kv_pages, _, page_indices, cu_q_lens, num_seqs = baseline.create_inputs(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        pages_per_seq=8,
    )
    kv_lens = jnp.array([128, 100, 64, 33], jnp.int32)
    args = (q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)
    expected = jax.jit(baseline.rpa_v2)(*args)
    actual = jax.jit(modules[implementation].kernel)(*args)
    assert cosine(actual, expected) > 0.9999


@pytest.mark.parametrize("implementation", V2_IMPLEMENTATIONS)
def test_v2_ignores_sequences_past_num_seqs(modules, implementation):
    baseline = modules["baseline"]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, _ = baseline.create_inputs(
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        pages_per_seq=8,
    )
    live = 2
    args = (
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([live], jnp.int32),
    )
    expected = jax.jit(baseline.rpa_v2)(*args)
    actual = jax.jit(modules[implementation].kernel)(*args)
    rows = live * (q.shape[0] // kv_lens.shape[0])
    assert cosine(actual[:rows], expected[:rows]) > 0.9999


V3_CONFIG = dict(
    num_seqs=4, q_len=32, kv_len=256, num_q_heads=8, num_kv_heads=2,
    head_dim=128, page_size=16,
)


def _v3_inputs(shape_fn, dtype=jnp.bfloat16, **overrides):
    """Fresh inputs for one call.

    Must be rebuilt per call: the v3 kernels default to ``update_kv_cache=True``
    and donate the cache buffer, so reusing it raises "Array has been deleted".
    """
    config = {**V3_CONFIG, **overrides}
    num_seqs, q_len, kv_len = config["num_seqs"], config["q_len"], config["kv_len"]
    page_size, head_dim = config["page_size"], config["head_dim"]
    total_pages = num_seqs * (kv_len // page_size)
    max_tokens = num_seqs * q_len
    keys = jax.random.split(jax.random.key(7), 4)
    kv_lens = jnp.full((num_seqs,), kv_len, jnp.int32)
    return dict(
        queries=jax.random.normal(
            keys[0], (max_tokens, config["num_q_heads"], head_dim), dtype=dtype
        ),
        keys_=jax.random.normal(
            keys[1], (max_tokens, config["num_kv_heads"], head_dim), dtype=dtype
        ),
        values=jax.random.normal(
            keys[2], (max_tokens, config["num_kv_heads"], head_dim), dtype=dtype
        ),
        kv_cache=jax.random.normal(
            keys[3],
            shape_fn(total_pages, page_size, config["num_kv_heads"], head_dim, dtype),
            dtype=dtype,
        ),
        kv_lens=kv_lens,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),  # flattened
        cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        cu_kv_lens=jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(kv_lens)]
        ),
        distribution=jnp.array([0, 0, num_seqs], jnp.int32),  # all mixed
    )


def _unwrap(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _call_v3(name, module, inputs, scale):
    """Invoke a v3 kernel through its own signature.

    tpu-inference takes 8 required arguments; sglang-jax takes 10, adding
    cu_kv_lens and a required custom_mask.
    """
    common = (
        inputs["queries"], inputs["keys_"], inputs["values"], inputs["kv_cache"],
        inputs["kv_lens"], inputs["page_indices"], inputs["cu_q_lens"],
    )
    if name == "sglang_jax":
        args = (*common, inputs["cu_kv_lens"], inputs["distribution"], None)
    else:
        args = (*common, inputs["distribution"])
    return _unwrap(module.ragged_paged_attention(*args, sm_scale=scale))


def test_v3_implementations_share_a_cache_layout(modules):
    """The packed layouts must match, or no shared input could test both."""
    shapes = {
        name: tuple(modules[name].get_kv_cache_shape(64, 16, 2, 128, jnp.bfloat16))
        for name in V3_IMPLEMENTATIONS
    }
    assert len(set(shapes.values())) == 1, shapes


@pytest.mark.parametrize("implementation", V3_IMPLEMENTATIONS)
def test_v3_matches_tpu_inference_pure_jax_reference(modules, implementation):
    """Both v3 kernels are checked against tpu-inference's pure-JAX reference.

    For sglang-jax this is a cross-repository check: its own bundled reference
    implements a different signature, so it cannot serve as its own baseline.
    """
    reference_module = modules["tpu_inference_v3"]
    scale = 1.0 / math.sqrt(V3_CONFIG["head_dim"])

    reference_inputs = _v3_inputs(reference_module.get_kv_cache_shape)
    # The reference returns (output, updated_cache), like the kernels do.
    expected = _unwrap(
        reference_module.ref_ragged_paged_attention(
            reference_inputs["queries"], reference_inputs["keys_"],
            reference_inputs["values"], reference_inputs["kv_cache"],
            reference_inputs["kv_lens"], reference_inputs["page_indices"],
            reference_inputs["cu_q_lens"], reference_inputs["distribution"],
            sm_scale=scale,
        )
    )

    module = modules[implementation]
    actual = _call_v3(
        implementation, module, _v3_inputs(module.get_kv_cache_shape), scale
    )
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


def test_v3_reference_is_pallas_free(modules):
    """A reference that lowered to Pallas would not be a valid baseline."""
    import inspect

    source = inspect.getsource(
        modules["tpu_inference_v3"].ref_ragged_paged_attention
    )
    assert "pallas" not in source
    assert "pl." not in source


def test_v3_signatures_differ_as_documented(modules):
    """Guard the finding that these two share semantics but not a signature."""
    import inspect

    required = {}
    for name in V3_IMPLEMENTATIONS:
        parameters = inspect.signature(
            modules[name].ragged_paged_attention
        ).parameters.values()
        required[name] = [
            p.name
            for p in parameters
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty
        ]
    assert len(required["tpu_inference_v3"]) == 8, required["tpu_inference_v3"]
    assert len(required["sglang_jax"]) == 10, required["sglang_jax"]
    assert "cu_kv_lens" in required["sglang_jax"]
    assert "cu_kv_lens" not in required["tpu_inference_v3"]


# ---------------------------------------------------------------------------
# The head-dim-64 v3 specialisation
# ---------------------------------------------------------------------------
# `kernel_hd64` exists to avoid the padding the general v3 kernel pays at
# head_dim 64. Their KV-cache layouts genuinely differ:
#
#   v3    (pages, page_size, align_to(2H, packing)//packing, packing, align_to(D,128))
#   hd64  (pages, page_size, align_to( H, packing)//packing, packing, 128)
#
# At H=2, D=64, bf16 that is (.,.,2,2,128) against (.,.,1,2,128): v3 gives K and
# V a 128-lane each and wastes half of both, hd64 packs K and V of one head into
# a single lane. So a comparison must NOT draw a random cache per layout -- the
# two would hold different content and the outputs could not agree.
#
# These tests sidestep a layout converter by setting `kv_len == q_len`, so no KV
# pre-exists in the cache: every element comes from this call's `keys_`/`values`
# and a zeroed cache is logically identical in either layout. That validates the
# compute path; validating cache *reuse* across layouts would need the
# converter, and is not attempted here.

HD64_CONFIG = dict(num_seqs=4, q_len=32, num_q_heads=8, num_kv_heads=2,
                   head_dim=64, page_size=16)


def _hd64_inputs(shape_fn, dtype=jnp.bfloat16):
    config = HD64_CONFIG
    num_seqs, q_len = config["num_seqs"], config["q_len"]
    kv_len = q_len  # nothing pre-exists in the cache
    page_size, head_dim = config["page_size"], config["head_dim"]
    total_pages = num_seqs * max(1, -(-kv_len // page_size))
    max_tokens = num_seqs * q_len
    keys = jax.random.split(jax.random.key(7), 3)
    return dict(
        queries=jax.random.normal(
            keys[0], (max_tokens, config["num_q_heads"], head_dim), dtype
        ),
        keys_=jax.random.normal(
            keys[1], (max_tokens, config["num_kv_heads"], head_dim), dtype
        ),
        values=jax.random.normal(
            keys[2], (max_tokens, config["num_kv_heads"], head_dim), dtype
        ),
        kv_cache=jnp.zeros(
            shape_fn(total_pages, page_size, config["num_kv_heads"], head_dim,
                     dtype),
            dtype,
        ),
        kv_lens=jnp.full((num_seqs,), kv_len, jnp.int32),
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        distribution=jnp.array([0, 0, num_seqs], jnp.int32),
    )


def _hd64_call(module, function):
    built = _hd64_inputs(module.get_kv_cache_shape)
    return _unwrap(
        function(
            built["queries"], built["keys_"], built["values"],
            built["kv_cache"], built["kv_lens"], built["page_indices"],
            built["cu_q_lens"], built["distribution"],
        )
    )


def test_hd64_cache_layout_differs_from_v3(modules):
    """Pin the difference, because assuming it away silently corrupts a check."""
    hd64 = load("rpa_hd64", "tpu_inference_hd64_optimized.py")
    v3 = modules["tpu_inference_v3"]
    args = (64, 16, 2, 64, jnp.bfloat16)
    assert v3.get_kv_cache_shape(*args) == (64, 16, 2, 2, 128)
    assert hd64.get_kv_cache_shape(*args) == (64, 16, 1, 2, 128)
    # hd64 stores exactly the logical bytes; v3 pads head_dim 64 into 128 lanes.
    assert np.prod(hd64.get_kv_cache_shape(*args)[2:]) == 2 * 2 * 64
    assert np.prod(v3.get_kv_cache_shape(*args)[2:]) == 2 * (2 * 2 * 64)


def test_hd64_matches_the_v3_reference(modules):
    """Checked against tpu-inference's own pure-JAX v3 reference."""
    hd64 = load("rpa_hd64", "tpu_inference_hd64_optimized.py")
    v3 = modules["tpu_inference_v3"]
    expected = _hd64_call(v3, v3.ref_ragged_paged_attention)
    actual = _hd64_call(hd64, hd64.kernel)
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999
    # The general v3 kernel answers the same question at this shape, so it is a
    # second opinion on the reference rather than on hd64.
    assert cosine(_hd64_call(v3, v3.kernel), expected) > 0.9999


def test_hd64_reaches_pallas_and_is_standalone():
    hd64 = load("rpa_hd64", "tpu_inference_hd64_optimized.py")
    source = (FAMILY / "tpu_inference_hd64_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 1
    # This family's SOURCE schema predates the `launch_points` field the newer
    # flatteners emit, so the count is asserted from the source above.
    assert "launch_points" not in hd64.SOURCE
    assert hd64.SOURCE["contract"] == "rpa_v3"
    import ast

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in (
                "tpu_inference", "tokamax", "maxtext", "sgl_jax"
            ), node.module
    # The two symbols pulled in from outside the kernels package.
    assert "def get_device_name(" in source
    assert "init_logger" not in source


# ---------------------------------------------------------------------------
# sglang-jax's `ragged_paged_attention.py` -- a third contract in this family
# ---------------------------------------------------------------------------
# Despite the filename this is NOT rpa_v2. It takes the same ten parameters as
# sglang's v3 file, plus a `custom_mask` no other migrated contract has, and it
# wants a **4-D** fused KV cache -- (pages, page_size, 2*num_kv_heads, head_dim),
# K and V interleaved on the head axis and unpacked -- where rpa_v3 uses the 5-D
# packed layout.
#
# That 4-D form was read out of the kernel's own validation, which destructures
# `(_, page_size, cache_num_kv_heads_interleaved, head_dim)`. Upstream's only
# call site (benchmark/kernels/flash_attention/) builds the **5-D** shape its
# own `get_kv_cache_shape` returns and would hit the same assertion, so that
# benchmark is stale against the kernel -- worth knowing before trusting it.

def _sglang_v2_inputs(dtype=jnp.bfloat16):
    num_seqs, q_len, num_q_heads, num_kv_heads, head_dim, page_size = (
        4, 32, 8, 2, 128, 16
    )
    kv_len = q_len  # no pre-existing KV, so no layout conversion is needed
    total_pages = num_seqs * max(1, -(-kv_len // page_size))
    max_tokens = num_seqs * q_len
    keys = jax.random.split(jax.random.key(7), 3)
    kv_lens = jnp.full((num_seqs,), kv_len, jnp.int32)
    return dict(
        queries=jax.random.normal(keys[0], (max_tokens, num_q_heads, head_dim), dtype),
        keys_=jax.random.normal(keys[1], (max_tokens, num_kv_heads, head_dim), dtype),
        values=jax.random.normal(keys[2], (max_tokens, num_kv_heads, head_dim), dtype),
        cache_4d=jnp.zeros((total_pages, page_size, 2 * num_kv_heads, head_dim), dtype),
        kv_lens=kv_lens,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        cu_kv_lens=jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(kv_lens)]
        ),
        distribution=jnp.array([0, 0, num_seqs], jnp.int32),
        total_pages=total_pages, page_size=page_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )


def test_sglang_v2_matches_the_v3_reference(modules):
    sglang_v2 = load("rpa_sg_v2", "sglang_jax_v2_optimized.py")
    v3 = modules["tpu_inference_v3"]
    built = _sglang_v2_inputs()
    expected = _unwrap(
        v3.ref_ragged_paged_attention(
            built["queries"], built["keys_"], built["values"],
            jnp.zeros(
                v3.get_kv_cache_shape(
                    built["total_pages"], built["page_size"],
                    built["num_kv_heads"], built["head_dim"], jnp.bfloat16,
                ),
                jnp.bfloat16,
            ),
            built["kv_lens"], built["page_indices"], built["cu_q_lens"],
            built["distribution"],
        )
    )
    actual = _unwrap(
        sglang_v2.kernel(
            built["queries"], built["keys_"], built["values"], built["cache_4d"],
            built["kv_lens"], built["page_indices"], built["cu_q_lens"],
            built["cu_kv_lens"], built["distribution"], None,
        )
    )
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


def test_sglang_v2_wants_a_4d_cache_not_its_own_helper_shape(modules):
    """Pin the mismatch, and that the 5-D helper output is rejected.

    `get_kv_cache_shape` in this same file returns the 5-D packed layout, which
    the kernel refuses. Anyone reaching for the obvious helper gets a rank
    error, so the requirement is recorded rather than left to be rediscovered.
    """
    sglang_v2 = load("rpa_sg_v2", "sglang_jax_v2_optimized.py")
    built = _sglang_v2_inputs()
    helper_shape = sglang_v2.get_kv_cache_shape(
        built["total_pages"], built["page_size"], built["num_kv_heads"],
        built["head_dim"], jnp.bfloat16,
    )
    assert len(helper_shape) == 5
    assert built["cache_4d"].ndim == 4
    with pytest.raises(ValueError, match="Expected 4D kv_cache_fused"):
        sglang_v2.kernel(
            built["queries"], built["keys_"], built["values"],
            jnp.zeros(helper_shape, jnp.bfloat16),
            built["kv_lens"], built["page_indices"], built["cu_q_lens"],
            built["cu_kv_lens"], built["distribution"], None,
        )


def test_sglang_v2_is_standalone_and_reaches_pallas():
    sglang_v2 = load("rpa_sg_v2", "sglang_jax_v2_optimized.py")
    source = (FAMILY / "sglang_jax_v2_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 1
    assert sglang_v2.SOURCE["contract"] == "rpa_sglang_fused_4d"
    import ast

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in (
                "sgl_jax", "tpu_inference", "tokamax", "maxtext"
            ), node.module


# ---------------------------------------------------------------------------
# The context-parallel v3 variant
# ---------------------------------------------------------------------------
# `rpa_v3_cp` shards the KV sequence across devices, so its contract carries
# `cp_rank`, `cp_group_size` and `q_pos_offsets` that the single-device v3
# kernel has none of. This host is a single v6e chip, so only the degenerate
# `cp_group_size=1` case is exercised here -- the sharded path needs a
# multi-device host and is NOT validated.

def test_v3_cp_matches_its_own_reference():
    """Single-device degenerate case: cp_group_size defaults to 1."""
    cp = load("rpa_cp", "tpu_inference_v3_cp_optimized.py")
    num_seqs, q_len, num_q_heads, num_kv_heads, head_dim, page_size = (
        4, 32, 8, 2, 128, 16
    )
    kv_len = q_len
    total_pages = num_seqs * max(1, -(-kv_len // page_size))
    max_tokens = num_seqs * q_len
    keys = jax.random.split(jax.random.key(7), 3)
    dtype = jnp.bfloat16
    q = jax.random.normal(keys[0], (max_tokens, num_q_heads, head_dim), dtype)
    k = jax.random.normal(keys[1], (max_tokens, num_kv_heads, head_dim), dtype)
    v = jax.random.normal(keys[2], (max_tokens, num_kv_heads, head_dim), dtype)
    cache = jnp.zeros(
        cp.get_kv_cache_shape(total_pages, page_size, num_kv_heads, head_dim, dtype),
        dtype,
    )
    common = (
        q, k, v, cache,
        jnp.full((num_seqs,), kv_len, jnp.int32),
        jnp.arange(total_pages, dtype=jnp.int32),
        jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        jnp.array([0, 0, num_seqs], jnp.int32),
    )
    expected = _unwrap(cp.ref_ragged_paged_attention(*common))
    actual = _unwrap(cp.kernel(*common))
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


def test_v3_cp_contract_carries_the_sharding_arguments():
    """What distinguishes this from the single-device v3 kernel."""
    import inspect

    cp = load("rpa_cp", "tpu_inference_v3_cp_optimized.py")
    params = inspect.signature(cp.kernel).parameters
    for name in ("cp_rank", "cp_group_size", "q_pos_offsets"):
        assert name in params, name
    source = (FAMILY / "tpu_inference_v3_cp_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 1
    assert cp.SOURCE["contract"] == "rpa_v3_cp"
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] != "tpu_inference", node.module


def test_hd64_reuses_a_populated_cache():
    """Cache **reuse**, which the single-phase test deliberately avoided.

    hd64's KV-cache layout differs from v3's, so a pre-populated cache cannot
    simply be shared between them. Rather than hand-write a layout converter --
    a wrong one would produce a false failure, which this family has already
    supplied twice -- each kernel fills its **own** cache in a first call and
    reuses it in a second. Both caches then hold the same logical KV in their
    own layouts, by construction rather than by conversion.

    Phase 1 prefills 64 tokens per sequence into a zeroed cache; phase 2 adds
    32 more with `kv_lens` grown accordingly, so phase 2 genuinely attends over
    KV the kernel itself wrote earlier.
    """
    hd64 = load("rpa_hd64", "tpu_inference_hd64_optimized.py")
    v3 = load("rpa_ti_v3", "tpu_inference_v3_optimized.py")

    num_seqs, num_q_heads, num_kv_heads, head_dim, page_size = 4, 8, 2, 64, 16
    phase1, phase2 = 64, 32
    total_pages = num_seqs * max(1, -(-(phase1 + phase2) // page_size))
    dtype = jnp.bfloat16
    keys = jax.random.split(jax.random.key(11), 6)

    def tokens(key, per_seq, heads):
        return jax.random.normal(
            key, (num_seqs * per_seq, heads, head_dim), dtype
        )

    stage1 = (tokens(keys[0], phase1, num_q_heads),
              tokens(keys[1], phase1, num_kv_heads),
              tokens(keys[2], phase1, num_kv_heads))
    stage2 = (tokens(keys[3], phase2, num_q_heads),
              tokens(keys[4], phase2, num_kv_heads),
              tokens(keys[5], phase2, num_kv_heads))
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], jnp.int32)

    def cu(per_seq):
        return jnp.arange(num_seqs + 1, dtype=jnp.int32) * per_seq

    def two_phase(module, function):
        # These kernels alias their output onto `q`, so each call needs its own
        # copy: a shared buffer is consumed by the first caller and the next
        # gets "Array has been deleted".
        a1, b1, c1 = (jnp.array(x, copy=True) for x in stage1)
        a2, b2, c2 = (jnp.array(x, copy=True) for x in stage2)
        cache = jnp.zeros(
            module.get_kv_cache_shape(
                total_pages, page_size, num_kv_heads, head_dim, dtype
            ),
            dtype,
        )
        first = function(
            a1, b1, c1, cache, jnp.full((num_seqs,), phase1, jnp.int32),
            page_indices, cu(phase1), distribution,
        )
        assert isinstance(first, (tuple, list)), "expected (output, cache)"
        second = function(
            a2, b2, c2, first[1],
            jnp.full((num_seqs,), phase1 + phase2, jnp.int32),
            page_indices, cu(phase2), distribution,
        )
        return _unwrap(second)

    expected = two_phase(v3, v3.ref_ragged_paged_attention)
    actual = two_phase(hd64, hd64.kernel)
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999
    # The general v3 kernel on the same workload, as a second opinion.
    assert cosine(two_phase(v3, v3.kernel), expected) > 0.9999


# ---------------------------------------------------------------------------
# batched_rpa  (experimental/batched_rpa, 2 launch points)
# ---------------------------------------------------------------------------

BRPA_BLOCKS = dict(bq_sz=32, bq_c_sz=32, bkv_sz=128, batch_size=4, n_buffer=2)


@pytest.fixture(scope="module")
def brpa():
    return load("rpa_batched", "tpu_inference_batched_optimized.py")


def _brpa_blocks(module):
    return module.BlockSizes(**BRPA_BLOCKS)


def _brpa_inputs(shape_fn, *, page_size, q_len, kv_len=None, seed=7,
                 num_seqs=4, num_q_heads=8, num_kv_heads=2, head_dim=128,
                 dtype=jnp.bfloat16, zero_cache=False):
    kv_len = q_len if kv_len is None else kv_len
    total_pages = num_seqs * max(1, -(-kv_len // page_size))
    max_tokens = num_seqs * q_len
    keys = jax.random.split(jax.random.key(seed), 4)
    cache_shape = shape_fn(total_pages, page_size, num_kv_heads, head_dim, dtype)
    return dict(
        queries=jax.random.normal(
            keys[0], (max_tokens, num_q_heads, head_dim), dtype=dtype),
        keys_=jax.random.normal(
            keys[1], (max_tokens, num_kv_heads, head_dim), dtype=dtype),
        values=jax.random.normal(
            keys[2], (max_tokens, num_kv_heads, head_dim), dtype=dtype),
        kv_cache=(jnp.zeros(cache_shape, dtype) if zero_cache
                  else jax.random.normal(keys[3], cache_shape, dtype=dtype)),
        kv_lens=jnp.full((num_seqs,), kv_len, jnp.int32),
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        distribution=jnp.array([0, 0, num_seqs], jnp.int32),
    )


def _brpa_args(inputs):
    return (
        inputs["queries"], inputs["keys_"], inputs["values"],
        inputs["kv_cache"], inputs["kv_lens"], inputs["page_indices"],
        inputs["cu_q_lens"], inputs["distribution"],
    )


def test_batched_shares_the_v3_cache_layout_in_its_default_layout(modules, brpa):
    """HEAD_ALONG_SUBLANE is upstream's default and matches rpa_v3 exactly.

    That is what lets the v3 reference validate this kernel without any
    layout-conversion code in between.
    """
    v3 = modules["tpu_inference_v3"]
    assert brpa.envs.USE_BATCHED_RPA_SEQ_ON_LANE is False
    assert tuple(brpa.get_kv_cache_shape(
        64, 16, 2, 128, jnp.bfloat16,
        kv_layout=brpa.KVLayout.HEAD_ALONG_SUBLANE,
    )) == tuple(v3.get_kv_cache_shape(64, 16, 2, 128, jnp.bfloat16))
    # The other layout is genuinely different, not a re-spelling.
    assert tuple(brpa.get_kv_cache_shape(
        8, 128, 2, 128, jnp.bfloat16,
        kv_layout=brpa.KVLayout.SEQ_ALONG_LANE,
    )) != tuple(v3.get_kv_cache_shape(8, 128, 2, 128, jnp.bfloat16))


@pytest.mark.parametrize("page_size", [16, 128])
def test_batched_matches_the_v3_reference(modules, brpa, page_size):
    v3 = modules["tpu_inference_v3"]
    scale = 1.0 / math.sqrt(128)
    expected = _unwrap(v3.ref_ragged_paged_attention(
        *_brpa_args(_brpa_inputs(v3.get_kv_cache_shape, page_size=page_size,
                                 q_len=32, kv_len=256)),
        sm_scale=scale,
    ))
    layout = brpa.KVLayout.HEAD_ALONG_SUBLANE
    shape_fn = functools.partial(brpa.get_kv_cache_shape, kv_layout=layout)
    actual = _unwrap(brpa.ragged_paged_attention(
        *_brpa_args(_brpa_inputs(shape_fn, page_size=page_size, q_len=32,
                                 kv_len=256)),
        sm_scale=scale, kv_layout=layout,
        decode_block_sizes=_brpa_blocks(brpa),
        prefill_block_sizes=_brpa_blocks(brpa),
    ))
    assert actual.shape == expected.shape
    assert cosine(actual, expected) > 0.9999


def test_batched_layouts_agree_when_every_kv_token_is_new(modules, brpa):
    """Compare the two KV layouts without re-deriving either packing.

    With ``kv_lens == q_len`` the kernel writes every KV token itself, so a
    zeroed cache in either layout gives the same answer as the v3 reference.
    Feeding independently-random caches of different shapes instead would
    compare different data and report a spurious mismatch (cosine 0.097).
    """
    v3 = modules["tpu_inference_v3"]
    scale = 1.0 / math.sqrt(128)
    expected = _unwrap(v3.ref_ragged_paged_attention(
        *_brpa_args(_brpa_inputs(v3.get_kv_cache_shape, page_size=128,
                                 q_len=128, zero_cache=True)),
        sm_scale=scale,
    ))
    for layout in (brpa.KVLayout.HEAD_ALONG_SUBLANE,
                   brpa.KVLayout.SEQ_ALONG_LANE):
        shape_fn = functools.partial(brpa.get_kv_cache_shape, kv_layout=layout)
        actual = _unwrap(brpa.ragged_paged_attention(
            *_brpa_args(_brpa_inputs(shape_fn, page_size=128, q_len=128,
                                     zero_cache=True)),
            sm_scale=scale, kv_layout=layout,
            decode_block_sizes=_brpa_blocks(brpa),
            prefill_block_sizes=_brpa_blocks(brpa),
        ))
        assert cosine(actual, expected) > 0.9999, layout


def test_batched_seq_along_lane_requires_page_size_128(brpa):
    """A tile-alignment guard, not a tuning preference."""
    layout = brpa.KVLayout.SEQ_ALONG_LANE
    shape_fn = functools.partial(brpa.get_kv_cache_shape, kv_layout=layout)
    with pytest.raises(ValueError, match="page_size=128"):
        brpa.ragged_paged_attention(
            *_brpa_args(_brpa_inputs(shape_fn, page_size=16, q_len=32,
                                     kv_len=256)),
            sm_scale=1.0, kv_layout=layout,
            decode_block_sizes=_brpa_blocks(brpa),
            prefill_block_sizes=_brpa_blocks(brpa),
        )


def test_batched_reaches_pallas_and_is_standalone(brpa):
    """Two launch points in the source; the wrapper runs each twice per call."""
    sys.path.insert(0, str(ROOT / "tools"))
    from profile_kernel import count_pallas_launches

    layout = brpa.KVLayout.HEAD_ALONG_SUBLANE
    shape_fn = functools.partial(brpa.get_kv_cache_shape, kv_layout=layout)
    inputs = _brpa_inputs(shape_fn, page_size=16, q_len=32, kv_len=256)
    function = functools.partial(
        brpa.ragged_paged_attention, sm_scale=1.0, kv_layout=layout,
        decode_block_sizes=_brpa_blocks(brpa),
        prefill_block_sizes=_brpa_blocks(brpa),
    )
    # generate_rpa_metadata + rpa_kernel, once for the decode region and once
    # for the mixed region.
    assert count_pallas_launches(function, _brpa_args(inputs)) == 4

    source = (FAMILY / "tpu_inference_batched_optimized.py").read_text()
    assert source.count("pl.pallas_call(") == 2
    assert brpa.SOURCE["launch_points"] == 2
    assert brpa.SOURCE["contract"] == "batched_rpa"
    # `tpu_inference` survives only in provenance strings and the envs-shim
    # docstring, so check for executable references rather than for the text.
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
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "tpu_inference"
        # vLLM's logger factory is named in the header as a recorded
        # substitution, so check that nothing *calls* it.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "init_logger"


def test_batched_compute_metadata_writes_through_the_schedule_argument(brpa):
    """Regression test for a flatten bug that only failed at run time.

    ``batched_rpa/schedule.py`` is a module named ``schedule`` that also takes a
    parameter named ``schedule``.  The flatten used to strip ``schedule.`` as a
    module qualifier everywhere, turning ``schedule.s_idx[step, lane] = s_idx``
    into ``s_idx[step, lane] = s_idx`` -- which parses, imports, and passes every
    static check, then fails at run time with "JAX arrays are immutable".
    """
    import inspect

    source = inspect.getsource(brpa.compute_metadata)
    for field in ("s_idx", "q_idx", "k_idx", "is_last_k", "do_writeback",
                  "dma_q", "dma_kv_cache", "dma_kv_new"):
        assert f"schedule.{field}[" in source, field
        # The mangled form assigned a loop variable to itself.
        assert f"\n        {field}[step, target_lane] = {field}\n" not in source
