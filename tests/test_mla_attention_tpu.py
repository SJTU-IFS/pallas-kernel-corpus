"""Small-shape TPU correctness for the MLA attention family.

The family holds five contracts.  Three -- Tokamax, tpu-inference v1 and v2, and
sglang-jax -- are checked against v1's shipped pure-JAX
``ref_mla_ragged_paged_attention``, a genuine cross-repository check for the
other repositories.  The two DeepSeek-V4 kernels are checked against
``baseline.ref_dsv4_sparse`` and ``baseline.ref_dsv4_sliding_window``, ported
from upstream's own test files.

Requires a TPU.  Run with::

    uv run --frozen --with pytest python -m pytest \
        tests/test_mla_attention_tpu.py -q
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp


ROOT = Path(__file__).parents[1]
FAMILY = ROOT / "kernels" / "attention" / "mla_attention"

pytestmark = pytest.mark.skipif(
    not any("TPU" in device.device_kind for device in jax.devices()),
    reason="MLA kernels are Mosaic TPU kernels",
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
        "baseline": load("mla_baseline", "baseline.py"),
        "tokamax": load("mla_tokamax", "tokamax_optimized.py"),
        "tpu_inference": load("mla_tpu_inference", "tpu_inference_optimized.py"),
        "tpu_inference_v2": load("mla_tpu_inference_v2", "tpu_inference_v2_optimized.py"),
        "sglang_jax": load("mla_sglang_jax", "sglang_jax_optimized.py"),
        "dsv4": load("mla_dsv4", "tpu_inference_dsv4_optimized.py"),
        "dsv4_swa": load("mla_dsv4_swa", "tpu_inference_dsv4_swa_optimized.py"),
    }


IMPLEMENTATIONS = ("tokamax", "tpu_inference", "tpu_inference_v2", "sglang_jax")

# The smallest block configuration all of them accept, so comparisons isolate
# the kernels rather than their tiling.
BLOCKS = {"num_kv_pages_per_block": 2, "num_queries_per_block": 16}
CONFIG = dict(
    num_seqs=4, q_len=32, kv_len=256, num_q_heads=16,
    lkv_dim=512, r_dim=64, page_size=16,
)


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


def unwrap(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def call(name, modules, built, scale):
    module = modules[name]
    baseline = modules["baseline"]
    kwargs = {"sm_scale": scale, **BLOCKS}
    if name == "tokamax":
        kwargs["vmem_limit_bytes"] = 64 * 1024 * 1024
    if name == "sglang_jax":
        args = baseline.ten_args(built)
    elif name == "tpu_inference_v2":
        args = baseline.nine_args_head_major(built)
        # v2 has no default block sizes and wants an explicit vmem limit.
        kwargs.update(
            num_queries_per_block=8, vmem_limit_bytes=100 * 1024 * 1024,
            s_dtype=jnp.float32, decode_batch_size=1,
        )
    else:
        args = baseline.nine_args(built)
    out = unwrap(module.mla_ragged_paged_attention(*args, **kwargs))
    return baseline.from_head_major(out) if name == "tpu_inference_v2" else out


def reference(modules, built, scale):
    baseline = modules["baseline"]
    return unwrap(
        modules["tpu_inference"].ref_mla_ragged_paged_attention(
            *baseline.nine_args(built), sm_scale=scale
        )
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_matches_pure_jax_reference(modules, implementation):
    baseline = modules["baseline"]
    scale = 1.0 / math.sqrt(CONFIG["lkv_dim"] + CONFIG["r_dim"])
    # cache_kv is donated, so each call needs its own inputs.
    expected = reference(modules, baseline.create_inputs(**CONFIG), scale)
    actual = call(implementation, modules, baseline.create_inputs(**CONFIG), scale)
    assert actual.shape == expected.shape
    assert_matches(actual, expected)


def test_cache_kv_is_donated(modules):
    """Pin the donation trap: the cache buffer is consumed by the call."""
    baseline = modules["baseline"]
    built = baseline.create_inputs(**CONFIG)
    cache = built["cache_kv"]
    call("tokamax", modules, built, 1.0 / math.sqrt(576))
    with pytest.raises(RuntimeError, match="deleted"):
        np.asarray(cache)


def test_cache_dim_is_latent_plus_rotary(modules):
    """kv_dim = align(lkv,128) + align(r,128), not lkv and not lkv + r."""
    baseline = modules["baseline"]
    built = baseline.create_inputs(**CONFIG)
    assert built["cache_kv"].shape[-1] == 512 + 128

    bad = dict(built)
    bad["cache_kv"] = jnp.zeros(
        (*built["cache_kv"].shape[:-1], CONFIG["lkv_dim"]),
        built["cache_kv"].dtype,
    )
    with pytest.raises(ValueError, match="kv_dim"):
        call("tpu_inference", modules, bad, 1.0 / math.sqrt(576))


def test_reference_is_pallas_free(modules):
    """A reference that lowered to Pallas would not be a valid check."""
    import inspect

    source = inspect.getsource(
        modules["tpu_inference"].ref_mla_ragged_paged_attention
    )
    assert "pallas" not in source
    assert "pl.pallas_call" not in source


def test_reference_is_not_jittable(modules):
    """Guard why this family reports no pure-JAX speed denominator.

    The reference validates traced values with Python control flow.  If it ever
    becomes jittable it could serve as a timing baseline, and this test should
    fail so that gets revisited.
    """
    baseline = modules["baseline"]
    built = baseline.create_inputs(**CONFIG)
    scale = 1.0 / math.sqrt(576)

    def run(*args):
        return modules["tpu_inference"].ref_mla_ragged_paged_attention(
            *args, sm_scale=scale
        )

    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(run)(*baseline.nine_args(built))


def test_sglang_takes_one_more_required_argument(modules):
    """The two nine/ten-argument MLA contracts differ by cu_kv_lens."""
    import inspect

    def required(module):
        return [
            p.name
            for p in inspect.signature(
                module.mla_ragged_paged_attention
            ).parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty
        ]

    assert len(required(modules["tokamax"])) == 9
    assert len(required(modules["tpu_inference"])) == 9
    assert len(required(modules["tpu_inference_v2"])) == 9
    sglang = required(modules["sglang_jax"])
    assert len(sglang) == 10
    assert "cu_kv_lens" in sglang


# ---------------------------------------------------------------------------
# tpu-inference v2: same nine argument names, transposed ql_nope
# ---------------------------------------------------------------------------


def test_v2_rejects_token_major_when_shapes_disagree(modules):
    """The loud half of the layout trap."""
    baseline = modules["baseline"]
    built = baseline.create_inputs(**CONFIG)  # 128 tokens, 16 heads
    with pytest.raises(ValueError, match="num_heads"):
        modules["tpu_inference_v2"].mla_ragged_paged_attention(
            *baseline.nine_args(built),
            sm_scale=1.0 / math.sqrt(576), s_dtype=jnp.float32,
            decode_batch_size=1, vmem_limit_bytes=100 * 1024 * 1024,
            num_kv_pages_per_block=2, num_queries_per_block=8,
        )


def test_v2_token_major_is_silently_wrong_when_square(modules):
    """The quiet half: at num_tokens == num_q_heads nothing catches it.

    This is why baseline exposes ``nine_args_head_major`` rather than leaving
    the transpose to call sites.
    """
    baseline = modules["baseline"]
    square = dict(CONFIG, num_seqs=2, q_len=8)  # 16 tokens, 16 heads
    scale = 1.0 / math.sqrt(576)
    kwargs = dict(
        sm_scale=scale, s_dtype=jnp.float32, decode_batch_size=1,
        vmem_limit_bytes=100 * 1024 * 1024,
        num_kv_pages_per_block=2, num_queries_per_block=8,
    )
    expected = reference(modules, baseline.create_inputs(**square), scale)

    good = baseline.from_head_major(
        modules["tpu_inference_v2"].mla_ragged_paged_attention(
            *baseline.nine_args_head_major(baseline.create_inputs(**square)),
            **kwargs,
        )[0]
    )
    assert_matches(good, expected)

    # No exception, right shape, wrong answer.
    bad = modules["tpu_inference_v2"].mla_ragged_paged_attention(
        *baseline.nine_args(baseline.create_inputs(**square)), **kwargs
    )[0]
    assert bad.shape == expected.shape
    assert cosine(bad, expected) < 0.9


def test_v2_accepts_an_fp8_cache(modules):
    """Upstream tests v2 only on fp8; bf16 works too, and both are checked.

    Agreement is looser with fp8 (0.996 vs 0.99999) because the cache the
    kernel writes back is itself quantized.
    """
    baseline = modules["baseline"]
    scale = 1.0 / math.sqrt(576)

    def build_fp8():
        built = baseline.create_inputs(**CONFIG)
        packing = baseline.get_dtype_packing(jnp.float8_e4m3fn)
        pages = built["cache_kv"].shape[0]
        built["cache_kv"] = jax.random.normal(
            jax.random.key(5),
            (pages, CONFIG["page_size"] // packing, packing, 512 + 128),
            jnp.float32,
        ).astype(jnp.float8_e4m3fn)
        # v1's validator requires new_kv_c.dtype == cache_kv.dtype.
        built["new_kv_c"] = built["new_kv_c"].astype(jnp.float8_e4m3fn)
        built["new_k_pe"] = built["new_k_pe"].astype(jnp.float8_e4m3fn)
        return built

    expected = reference(modules, build_fp8(), scale)
    actual = baseline.from_head_major(
        modules["tpu_inference_v2"].mla_ragged_paged_attention(
            *baseline.nine_args_head_major(build_fp8()),
            sm_scale=scale, s_dtype=jnp.float32, decode_batch_size=1,
            vmem_limit_bytes=100 * 1024 * 1024,
            num_kv_pages_per_block=2, num_queries_per_block=8,
        )[0]
    )
    assert cosine(actual, expected) > 0.99


def test_v2_divisors_matches_the_brute_force_definition(modules):
    """sympy is not in the pinned dependency set; the substitute must be exact."""
    divisors = modules["tpu_inference_v2"].divisors
    for n in range(1, 2001):
        assert divisors(n) == [d for d in range(1, n + 1) if n % d == 0], n


# ---------------------------------------------------------------------------
# DeepSeek-V4 sparse top-k  (experimental/deepseek_v4/mla.py)
# ---------------------------------------------------------------------------


DSV4_BLOCKS = dict(num_kv_pages_per_block=1, num_queries_per_block=16)


def _run_sparse(modules, built):
    baseline = modules["baseline"]
    expected = baseline.ref_dsv4_sparse(
        built["q"], built["cache_bf16"], built["kv_lens"],
        built["kv_lens_to_attend"], built["page_indices"], built["cu_q_lens"],
        built["distribution"], built["attention_sinks"],
        built["swa_accumution"], built["swa_l"], built["swa_m"],
        topk_indices=built["topk_indices"], sm_scale=1.0,
    )
    actual = modules["dsv4"].mla_ragged_paged_attention(
        built["q"], built["cache_packed"], built["kv_lens"],
        built["kv_lens_to_attend"], built["topk_indices"],
        built["page_indices"], built["cu_q_lens"], built["distribution"],
        built["attention_sinks"], built["swa_accumution"], built["swa_l"],
        built["swa_m"], sm_scale=1.0, **DSV4_BLOCKS,
    )
    jax.block_until_ready((expected, actual))
    return np.asarray(expected, np.float32), np.asarray(actual, np.float32)


@pytest.mark.parametrize("topk", [1024, None], ids=["csa_topk", "hca_kv_lens"])
def test_dsv4_sparse_matches_reference(modules, topk):
    baseline = modules["baseline"]
    expected, actual = _run_sparse(
        modules, baseline.create_sparse_inputs(topk=topk)
    )
    np.testing.assert_allclose(expected, actual, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("topk", [1024, None], ids=["csa_topk", "hca_kv_lens"])
def test_dsv4_sparse_matches_reference_without_a_dominant_carry_in(
    modules, topk
):
    """Upstream's settings pin the output near 5000/200 = 25 whatever the
    attention does.  With a neutral carry-in and no sinks the kernel's own
    attention math is the only thing setting the result."""
    baseline = modules["baseline"]
    built = baseline.create_sparse_inputs(
        topk=topk, sink_value=0.0, neutral_carry=True
    )
    expected, actual = _run_sparse(modules, built)
    # Values land around absmean 10, so this is a real constraint.
    assert np.mean(np.abs(expected)) > 1.0
    np.testing.assert_allclose(expected, actual, rtol=0.1, atol=0.15)


def test_dsv4_sparse_takes_exactly_one_of_topk_or_kv_lens(modules):
    """The two modes are mutually exclusive; neither argument has a default."""
    import inspect

    params = inspect.signature(
        modules["dsv4"].mla_ragged_paged_attention
    ).parameters
    required = [
        p.name for p in params.values()
        if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty
    ]
    assert "kv_lens_to_attend" in required
    assert "topk_indices" in required
    # No ql_nope/q_pe split and no new_kv: this contract is not the v1 one.
    assert "ql_nope" not in required
    assert "new_kv_c" not in required


# ---------------------------------------------------------------------------
# DeepSeek-V4 sliding window  (experimental/deepseek_v4/mla_swa.py)
# ---------------------------------------------------------------------------


SWA_BLOCKS = dict(num_queries_per_block=8, num_kv_pages_per_block=2)


def _swa_step(modules, state, new_kv_lens, distribution):
    """Advance both caches one decode/prefill step and compare every output."""
    baseline = modules["baseline"]
    rng = state["rng"]
    cu_q_lens = jnp.concatenate(
        [jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)]
    )
    state["kv_lens"] = state["kv_lens"] + new_kv_lens
    total_tokens = int(jnp.sum(new_kv_lens))
    q = jnp.array(
        rng.random(size=(total_tokens, state["num_heads"], state["head_dim"]),
                   dtype=np.float32)
    ).astype(jnp.bfloat16)
    new_kv = jnp.array(
        rng.random(size=(total_tokens, state["head_dim"]), dtype=np.float32)
    ).astype(jnp.bfloat16)

    out_ref, state["ref_cache"], l_ref, m_ref = (
        baseline.ref_dsv4_sliding_window(
            q, new_kv, state["ref_cache"], state["kv_lens"],
            state["page_indices"], cu_q_lens, distribution,
            state["attention_sinks"], sm_scale=1.0,
            sliding_window=state["sliding_window"],
        )
    )
    out, state["packed_cache"], L, m = (
        modules["dsv4_swa"].mla_sliding_window_ragged_paged_attention(
            q, new_kv, state["packed_cache"], state["kv_lens"],
            state["page_indices"], cu_q_lens, distribution,
            state["attention_sinks"], sm_scale=1.0,
            sliding_window=state["sliding_window"],
            logical_page_size=state["page_size"], **SWA_BLOCKS,
        )
    )
    jax.block_until_ready((out_ref, out, l_ref, L, m_ref, m))

    assert L.shape == (total_tokens, state["num_heads"])
    assert m.shape == (total_tokens, state["num_heads"])

    # L and m come back near-exact -- m bitwise, L to ~1e-6 -- so comparing
    # them at the loose rtol=0.1 used for the outputs would let a badly wrong
    # kernel through.  m in particular is nearly constant across tokens, and at
    # rtol=0.1 a kernel returning its mean everywhere would pass.
    np.testing.assert_array_equal(
        np.asarray(m_ref, np.float32), np.asarray(m, np.float32),
        err_msg="m (the running max) should match bitwise",
    )
    np.testing.assert_allclose(
        np.asarray(l_ref, np.float32), np.asarray(L, np.float32),
        rtol=1e-5, atol=1e-5, err_msg="L",
    )

    # The output comparison is only meaningful when the reference is not itself
    # ~0.  At upstream's own sink settings it is: see the test below.  Assert
    # which regime we are in rather than running an allclose that two arrays of
    # zeros would satisfy however wrong the kernel is.
    want_out = np.asarray(out_ref, np.float32)
    got_out = np.asarray(out, np.float32)
    if np.max(np.abs(want_out)) <= 0.1:
        assert np.max(np.abs(got_out)) <= 0.1, (
            "reference collapsed to zero but the kernel did not; the sink "
            "regime assumed by this test no longer holds"
        )
    else:
        np.testing.assert_allclose(want_out, got_out, rtol=0.1, atol=0.1,
                                   err_msg="out")
    _compare_swa_cache(modules, state)
    return want_out


def _compare_swa_cache(modules, state):
    """The kernel's uint8 cache must dequantize to the reference's bf16 one."""
    baseline = modules["baseline"]
    page_size, packing = state["page_size"], state["packing"]
    pages, batch = state["pages_per_seq"], state["batch_size"]

    rows = np.arange(page_size // packing)[None, :, None]
    cols = np.arange(packing)[None, None, :]
    token_indices = (
        np.arange(pages)[:, None, None] * page_size + rows * packing + cols
    )
    valid = (
        token_indices[None, ...] < np.array(state["kv_lens"])[:, None, None, None]
    ).reshape(batch * pages, page_size // packing, packing, 1)

    want = np.where(
        valid, baseline.quantize_dequantize_dsv4(state["ref_cache"]), 0
    )
    uint8_packing = baseline.get_dtype_packing(jnp.uint8)
    got = baseline.unpack_dsv4_fp8_cache(
        state["packed_cache"][:, : page_size // uint8_packing, :, :]
    ).reshape(batch * pages, page_size // packing, packing, 512)
    np.testing.assert_allclose(
        np.asarray(want, np.float32), np.asarray(np.where(valid, got, 0),
                                                 np.float32),
        rtol=0.1, atol=0.1, err_msg="swa cache",
    )


def _swa_schedule(modules, state):
    """Prefill, then mixed, then full decode -- upstream's own three steps."""
    baseline = modules["baseline"]
    batch, window = state["batch_size"], state["sliding_window"]
    key = jax.random.PRNGKey(1234)
    outs = []

    key, sub = jax.random.split(key)
    outs.append(_swa_step(
        modules, state,
        jax.random.randint(sub, (batch,), window // 2, window * 2, jnp.int32),
        jnp.array([0, 0, batch], jnp.int32),
    ))

    decode = batch // 2
    key, sub = jax.random.split(key)
    outs.append(_swa_step(
        modules, state,
        jnp.concatenate([
            jnp.ones((decode,), jnp.int32),
            jax.random.randint(sub, (batch - decode,), window // 2,
                               window * 2, jnp.int32),
        ]),
        jnp.array([decode, decode, batch], jnp.int32),
    ))

    outs.append(_swa_step(
        modules, state, jnp.ones((batch,), jnp.int32),
        jnp.array([batch, batch, batch], jnp.int32),
    ))
    assert baseline is not None
    return outs


def test_dsv4_sliding_window_matches_reference(modules):
    """Upstream's own settings: sinks in [200, 500]."""
    baseline = modules["baseline"]
    _swa_schedule(modules, baseline.create_sliding_window_inputs())


def test_dsv4_sliding_window_output_check_needs_a_small_sink(modules):
    """At upstream's sinks the outputs are ~1e-26 and agree vacuously.

    Two arrays of zeros match at any tolerance, so the output comparison in
    upstream's own test constrains nothing; L, m and the cache carry it.  With
    the sink at 0.0 the outputs land near absmean 0.52 and the comparison bites.
    """
    baseline = modules["baseline"]
    dominated = _swa_schedule(
        modules, baseline.create_sliding_window_inputs()
    )
    assert max(np.max(np.abs(o)) for o in dominated) < 1e-20

    informative = _swa_schedule(
        modules, baseline.create_sliding_window_inputs(sink_value=0.0)
    )
    assert min(np.mean(np.abs(o)) for o in informative) > 0.1


def test_dsv4_kernels_chain_through_l_and_m(modules):
    """The two DSV4 kernels are a pipeline, and the flag that joins them
    defaults to the wrong value.

    ``mla_swa`` returns ``(out, cache, L, m)``; ``mla`` consumes exactly those
    three as ``swa_accumution``/``swa_l``/``swa_m``.  Upstream chains them with
    ``unnormalized_output=True`` -- whose default is ``False``.
    """
    import inspect

    swa = inspect.signature(
        modules["dsv4_swa"].mla_sliding_window_ragged_paged_attention
    ).parameters
    assert swa["unnormalized_output"].default is False

    sparse = inspect.signature(
        modules["dsv4"].mla_ragged_paged_attention
    ).parameters
    assert {"swa_accumution", "swa_l", "swa_m"} <= set(sparse)
