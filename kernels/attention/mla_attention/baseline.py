"""Reference contracts for the MLA (multi-head latent attention) kernels here.

MLA compresses the KV cache into a single latent stream shared by every query
head, plus a small rotary part.  That sharing is what makes the cache small,
and it is why the contract looks unlike ordinary attention::

    ql_nope       [num_tokens, num_q_heads, lkv_dim]   latent query part
    q_pe          [num_tokens, num_q_heads, r_dim]     rotary query part
    new_kv_c      [num_tokens, lkv_dim]                latent KV to append
    new_k_pe      [num_tokens, r_dim]                  rotary K to append
    cache_kv      [total_num_pages, page_size // packing, packing, kv_dim]
    kv_lens       i32[max_num_seqs]
    page_indices  i32[max_num_seqs * pages_per_seq]    flattened
    cu_q_lens     i32[max_num_seqs + 1]
    distribution  i32[3]  decode / prefill / mixed split, as in rpa_v3
    ->            [num_tokens, num_q_heads, lkv_dim]

Two shape facts are easy to get wrong and cost real debugging time:

* ``kv_dim == align_to(lkv_dim, 128) + align_to(r_dim, 128)``.  The cache holds
  the latent and rotary parts **concatenated**, each padded to a 128-lane
  multiple.  At DeepSeek-V3 dims (``lkv=512, r=64``) that is ``512 + 128 = 640``,
  not 512 and not 576.
* ``cache_kv`` is **donated**: the entry points are decorated
  ``@jax.jit(donate_argnames=("cache_kv"))``, so the buffer is consumed. Reusing
  one across two calls raises "Array has been deleted". There is no flag to
  disable it, so a benchmark loop needs a fresh cache per call.

## Five contracts live here, and none of them substitute for another

``mla_ragged_paged_attention``              Tokamax, vLLM tpu-inference v1
    nine required arguments, ending ``cu_q_lens, distribution``.

``mla_ragged_paged_attention_cu_kv``        sglang-jax v2
    ten required arguments, inserting ``cu_kv_lens`` before ``distribution`` --
    exactly the divergence seen between rpa_v2 and rpa_v3.

``mla_ragged_paged_attention_head_major``   vLLM tpu-inference v2
    the same nine argument *names* as v1, but ``ql_nope`` is
    ``[num_q_heads, num_tokens, lkv_dim]`` and the output comes back head-major
    too.  ``q_pe`` stays token-major.  See the warning below.

``mla_sparse_topk``                         vLLM tpu-inference DeepSeek-V4
    a single fused ``q``, no ``new_kv``, an FP8-packed cache, attention sinks,
    a carry-in accumulator, and either ``topk_indices`` or ``kv_lens_to_attend``.

``mla_sliding_window``                      vLLM tpu-inference DeepSeek-V4
    ``new_kv`` separate from ``cache_kv``, an FP8-packed cache written by the
    kernel, and four return values ``(out, cache, L, m)``.

### The v2 layout trap

v1 and v2 share nine argument names and reject nothing when a caller mixes up
``ql_nope``'s first two axes -- as long as ``num_tokens != num_q_heads``, v2's
validator catches it with a shape error.  When ``num_tokens == num_q_heads`` the
array is square in exactly those two axes, nothing raises, and the kernel
returns a **wrong answer**: measured cosine **0.534** against the reference,
versus 0.99999 for the correctly transposed call.  ``nine_args_head_major``
below exists so call sites do not open-code the transpose, and a test pins both
the loud and the silent case.

### The two DeepSeek-V4 kernels are a pipeline, not alternatives

``mla_swa`` runs first over the sliding window and returns ``(out, cache, L, m)``;
``mla``'s ``swa_accumution``/``swa_l``/``swa_m`` arguments are exactly those
three, rescaled into its own online softmax.  Upstream chains them in
``deepseek_v4_attention.py`` with ``unnormalized_output=True`` on the SWA call --
a flag whose **default is False**, which is the wrong value for chaining.
Neither upstream test exercises the chain: ``mla_swa_test`` runs SWA alone at
the default, and ``mla_test`` feeds hand-made constants for the carry-in.

## Which references are runnable

Three of the five contracts have a pure-JAX reference, and all three come from
upstream rather than from this corpus:

* ``mla_ragged_paged_attention`` (and, by cross-check, the ``cu_kv`` and
  head-major variants) uses vLLM tpu-inference v1's shipped
  ``ref_mla_ragged_paged_attention``, which is genuinely Pallas-free.  Using it
  for Tokamax and sglang-jax is a **cross-repository** check, a stronger
  independence property than a hand-written baseline would give.
* ``mla_sparse_topk`` and ``mla_sliding_window`` use ``ref_dsv4_sparse`` and
  ``ref_dsv4_sliding_window`` below, ported from upstream's own
  ``tests/kernels/deepseek_v4/mla_test.py`` and ``mla_swa_test.py`` at the
  audited commit.  They are ports, not re-derivations: re-deriving the DSV4 FP8
  cache layout here would risk a subtly wrong reference producing false
  failures, the same reasoning applied in the ragged-paged-attention family.

**None of them is a speed denominator.**  v1's reference calls
``dynamic_validate_inputs``, which tests traced values with Python ``if``, so it
cannot be jitted (``TracerBoolConversionError``); the two DSV4 references loop
in Python over ``distribution[-1]``.  Speedups in this family are therefore
reported *between implementations*, and ``result.json`` records
``reference_is_jittable: false`` so that cannot be misread.

### What upstream's own dsv4 output comparison does not check

At upstream's settings the sliding-window test compares two arrays that are both
about ``1e-26``: ``attention_sinks`` is drawn from [200, 500] while the attention
logits reach ~135, so ``exp(sink - m)`` swamps the softmax denominator and every
output collapses to zero.  Two arrays of zeros agree to any tolerance.  The
meaningful checks at those settings are ``L``, ``m`` and the cache; the corpus
test therefore *also* runs the sink at 0.0, where outputs land at ``absmean
0.52`` and the comparison actually constrains the attention math.  The sparse
kernel has a milder version of the same problem -- its carry-in fixes the output
near ``5000/200 = 25`` -- so that test also runs a neutral carry-in.

## What is not migrated

Of the family's nine audited TPU launch points, six are migrated.  The other
three are JAXBench's ``3p_MLA_Attention``, which contributes no new Pallas
kernel: they are the *flash-attention* kernel already migrated at
``kernels/attention/flash_attention/jaxbench_optimized.py`` -- 24 of its 27
shared definitions are AST-identical to ``1p_Flash_Attention``, differing only
in ``benchmark``/``create_inputs``/``workload`` plus an added ``_apply_rope``.
Its MLA-ness lives in the surrounding JAX (LoRA projections and RoPE), not in
the Pallas launch, so migrating it would double-count a kernel the corpus
already has.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp


SOURCE = {
    "kind": "reference-notes",
    "backend": "jax",
    "target": "portable",
    "contracts": (
        "mla_ragged_paged_attention",
        "mla_ragged_paged_attention_cu_kv",
        "mla_ragged_paged_attention_head_major",
        "mla_sparse_topk",
        "mla_sliding_window",
    ),
}

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def align_to(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_dtype_packing(dtype) -> int:
    return 32 // jax.dtypes.itemsize_bits(dtype)


# ---------------------------------------------------------------------------
# mla_ragged_paged_attention / _cu_kv / _head_major
# ---------------------------------------------------------------------------


def create_inputs(
    *,
    num_seqs: int = 8,
    q_len: int = 64,
    kv_len: int = 1024,
    num_q_heads: int = 16,
    lkv_dim: int = 512,
    r_dim: int = 64,
    page_size: int = 16,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 11,
) -> dict[str, jax.Array]:
    """Build MLA inputs; defaults are DeepSeek-V3 MLA dimensions.

    Returns a dict so callers can assemble the nine-, ten- or head-major form.
    Build a fresh one per kernel call: ``cache_kv`` is donated.
    """
    tokens = num_seqs * q_len
    pages_per_seq = kv_len // page_size
    total_pages = num_seqs * pages_per_seq
    packing = get_dtype_packing(dtype)
    kv_dim = align_to(lkv_dim, 128) + align_to(r_dim, 128)
    keys = jax.random.split(jax.random.key(seed), 5)
    kv_lens = jnp.full((num_seqs,), kv_len, jnp.int32)
    return dict(
        ql_nope=jax.random.normal(keys[0], (tokens, num_q_heads, lkv_dim), dtype),
        q_pe=jax.random.normal(keys[1], (tokens, num_q_heads, r_dim), dtype),
        new_kv_c=jax.random.normal(keys[2], (tokens, lkv_dim), dtype),
        new_k_pe=jax.random.normal(keys[3], (tokens, r_dim), dtype),
        cache_kv=jax.random.normal(
            keys[4], (total_pages, page_size // packing, packing, kv_dim), dtype
        ),
        kv_lens=kv_lens,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=jnp.arange(num_seqs + 1, dtype=jnp.int32) * q_len,
        cu_kv_lens=jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(kv_lens)]
        ),
        distribution=jnp.array([0, 0, num_seqs], jnp.int32),
    )


NINE_ARG_ORDER = (
    "ql_nope", "q_pe", "new_kv_c", "new_k_pe", "cache_kv",
    "kv_lens", "page_indices", "cu_q_lens", "distribution",
)
TEN_ARG_ORDER = (*NINE_ARG_ORDER[:8], "cu_kv_lens", "distribution")


def nine_args(built: dict[str, jax.Array]) -> list[jax.Array]:
    return [built[name] for name in NINE_ARG_ORDER]


def ten_args(built: dict[str, jax.Array]) -> list[jax.Array]:
    return [built[name] for name in TEN_ARG_ORDER]


def nine_args_head_major(built: dict[str, jax.Array]) -> list[jax.Array]:
    """The v2 argument list: ``ql_nope`` head-major, everything else unchanged.

    Do not open-code this transpose at call sites.  When
    ``num_tokens == num_q_heads`` the mistake is undetectable and silent; see
    the module docstring.
    """
    args = nine_args(built)
    args[0] = jnp.transpose(args[0], (1, 0, 2))
    return args


def from_head_major(out: jax.Array) -> jax.Array:
    """Undo v2's head-major output layout, for comparison against v1."""
    return jnp.transpose(out, (1, 0, 2))


def logical_flops(
    *, num_seqs: int, q_len: int, kv_len: int, num_q_heads: int,
    lkv_dim: int, r_dim: int,
) -> int:
    """QK over (lkv + r) and PV over lkv, for every query head."""
    tokens = num_seqs * q_len
    qk = 2 * tokens * kv_len * num_q_heads * (lkv_dim + r_dim)
    pv = 2 * tokens * kv_len * num_q_heads * lkv_dim
    return qk + pv


# ---------------------------------------------------------------------------
# The DeepSeek-V4 FP8 cache layout, shared by both DSV4 contracts
#
# Ported from tests/kernels/deepseek_v4/mla_test.py and mla_swa_test.py at
# tpu-inference commit 8b9c90928c94c7230d1bc891534a301510a6a30d.  A 640-lane
# uint8 page row holds, in order: 448 bytes of float8_e4m3fn latent, 128 bytes
# of bfloat16 rotary (64 values), 7 bytes of float8_e8m0fnu block scales, and
# 57 bytes of padding.  The 448 latent lanes are scaled in 7 blocks of 64.
# ---------------------------------------------------------------------------

DSV4_FP8_LANES = 448
DSV4_ROPE_LANES = 64
DSV4_SCALE_BLOCKS = 7
DSV4_SCALE_BLOCK = 64
DSV4_CACHE_LANES = 640


def _dsv4_block_scales(fp8_blocked: jax.Array) -> jax.Array:
    """Power-of-two per-block scale factors, as upstream computes them."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_amax = jnp.clip(
        jnp.max(jnp.abs(fp8_blocked), axis=-1, keepdims=True), 1e-4, None
    )
    return jnp.power(2.0, jnp.ceil(jnp.log2(x_amax / fp8_max)))


def quantize_dequantize_dsv4(cache_kv: jax.Array) -> jax.Array:
    """Apply the DSV4 FP8 round trip to a bf16 cache, keeping its shape.

    The kernel reads a quantized cache, so a reference fed the unquantized one
    would differ from it for a reason that is not a kernel bug.
    """
    total_num_pages, rows, packing, lkv_dim = cache_kv.shape
    page_size = rows * packing
    flat = cache_kv.reshape(total_num_pages, page_size, lkv_dim)
    fp8_part = flat[..., :DSV4_FP8_LANES]
    bf16_part = flat[..., DSV4_FP8_LANES:DSV4_FP8_LANES + DSV4_ROPE_LANES]

    fp8_blocked = fp8_part.reshape(
        total_num_pages, page_size, DSV4_SCALE_BLOCKS, DSV4_SCALE_BLOCK
    )
    sf = _dsv4_block_scales(fp8_blocked)
    fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    scales = sf.reshape(
        total_num_pages, page_size, DSV4_SCALE_BLOCKS
    ).astype(jnp.float8_e8m0fnu)
    fp8_dequant = (
        fp8_quant.astype(jnp.bfloat16) * scales[..., None].astype(jnp.bfloat16)
    ).reshape(total_num_pages, page_size, DSV4_FP8_LANES)
    out = jnp.concatenate([fp8_dequant, bf16_part], axis=-1)
    return out.reshape(total_num_pages, rows, packing, lkv_dim)


def pack_dsv4_fp8_cache(cache_kv: jax.Array) -> jax.Array:
    """Pack a bf16 cache into the uint8 DSV4 layout the kernels read.

    Returns ``[total_num_pages, page_size // 4, 4, 640]`` of uint8.
    """
    total_num_pages, rows, packing, lkv_dim = cache_kv.shape
    page_size = rows * packing
    flat = cache_kv.reshape(total_num_pages, page_size, lkv_dim)
    fp8_part = flat[..., :DSV4_FP8_LANES]
    bf16_part = flat[..., DSV4_FP8_LANES:DSV4_FP8_LANES + DSV4_ROPE_LANES]

    fp8_blocked = fp8_part.reshape(
        total_num_pages, page_size, DSV4_SCALE_BLOCKS, DSV4_SCALE_BLOCK
    )
    sf = _dsv4_block_scales(fp8_blocked)
    fp8_quant = (fp8_blocked * (1.0 / sf)).astype(jnp.float8_e4m3fn)
    scales = sf.reshape(
        total_num_pages, page_size, DSV4_SCALE_BLOCKS
    ).astype(jnp.float8_e8m0fnu)

    pad_lanes = DSV4_CACHE_LANES - (DSV4_FP8_LANES + 2 * DSV4_ROPE_LANES
                                    + DSV4_SCALE_BLOCKS)
    packed = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(
                fp8_quant.reshape(total_num_pages, page_size, DSV4_FP8_LANES),
                jnp.uint8,
            ),
            jax.lax.bitcast_convert_type(bf16_part, jnp.uint8).reshape(
                total_num_pages, page_size, 2 * DSV4_ROPE_LANES
            ),
            jax.lax.bitcast_convert_type(scales, jnp.uint8),
            jnp.zeros((total_num_pages, page_size, pad_lanes), jnp.uint8),
        ],
        axis=-1,
    )
    uint8_packing = get_dtype_packing(jnp.uint8)
    return packed.reshape(
        total_num_pages, page_size // uint8_packing, uint8_packing,
        DSV4_CACHE_LANES,
    )


def unpack_dsv4_fp8_cache(packed: jax.Array) -> jax.Array:
    """Inverse of :func:`pack_dsv4_fp8_cache`, to bf16 ``[pages, tokens, 512]``."""
    total_num_pages, rows = packed.shape[0], packed.shape[1]
    uint8_packing = get_dtype_packing(jnp.uint8)
    flat = packed.reshape(total_num_pages * rows * uint8_packing,
                          DSV4_CACHE_LANES)
    rope_end = DSV4_FP8_LANES + 2 * DSV4_ROPE_LANES
    fp8_val = jax.lax.bitcast_convert_type(
        flat[..., :DSV4_FP8_LANES], jnp.float8_e4m3fn
    ).astype(jnp.bfloat16)
    rope_val = jax.lax.bitcast_convert_type(
        flat[..., DSV4_FP8_LANES:rope_end].reshape(
            flat.shape[:-1] + (DSV4_ROPE_LANES, 2)
        ),
        jnp.bfloat16,
    )
    scales = jax.lax.bitcast_convert_type(
        flat[..., rope_end:rope_end + DSV4_SCALE_BLOCKS], jnp.float8_e8m0fnu
    ).astype(jnp.bfloat16)
    dequant = (
        fp8_val.reshape(-1, DSV4_SCALE_BLOCKS, DSV4_SCALE_BLOCK)
        * scales[..., None]
    ).reshape(-1, DSV4_FP8_LANES)
    return jnp.concatenate([dequant, rope_val], axis=-1).reshape(
        total_num_pages, rows * uint8_packing,
        DSV4_FP8_LANES + DSV4_ROPE_LANES,
    )


# ---------------------------------------------------------------------------
# mla_sparse_topk  (experimental/deepseek_v4/mla.py)
# ---------------------------------------------------------------------------


def create_sparse_inputs(
    *,
    batch_size: int = 12,
    num_heads: int = 64,
    head_dim: int = 512,
    page_size: int = 16,
    pages_per_seq_tokens: int = 500,
    topk: int | None = 1024,
    sink_value: float | None = None,
    neutral_carry: bool = False,
    seed: int = 1234,
) -> dict:
    """Build sparse-MLA inputs; defaults follow upstream's own correctness test.

    ``topk`` selects the mode: an int builds ``topk_indices`` (CSA), ``None``
    builds ``kv_lens_to_attend`` (HCA).  ``sink_value`` and ``neutral_carry``
    exist because upstream's defaults make the output comparison weak; see the
    module docstring.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    num_decode_seqs = batch_size // 2

    def rand(shape, dtype=jnp.bfloat16):
        return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    new_kv_lens = jnp.concatenate([
        jnp.ones((num_decode_seqs,), jnp.int32),
        jnp.array(rng.integers(low=4, high=60, size=(batch_size - num_decode_seqs,))),
    ])
    cu_q_lens = jnp.concatenate(
        [jnp.array([0]), jnp.cumulative_sum(new_kv_lens, dtype=jnp.int32)]
    )
    kv_lens = new_kv_lens + jnp.array(
        rng.integers(low=30, high=200, size=(batch_size,))
    )
    total_tokens = int(jnp.sum(new_kv_lens))
    q = rand((total_tokens, num_heads, head_dim))

    topk_indices = kv_lens_to_attend = None
    if topk is not None:
        rows = []
        for i in range(batch_size):
            kv_len_i = int(kv_lens[i])
            for _ in range(int(new_kv_lens[i])):
                picked = list(rng.permutation(kv_len_i)[:topk])
                picked.extend([-1] * (topk - len(picked)))
                rows.append(picked)
        topk_indices = jnp.array(rows, dtype=jnp.int32)
    else:
        rows = []
        for i in range(batch_size):
            for _ in range(int(new_kv_lens[i])):
                rows.append(rng.integers(low=0, high=int(kv_lens[i])))
        kv_lens_to_attend = jnp.array(rows, dtype=jnp.int32)

    pages_per_seq = cdiv(pages_per_seq_tokens, page_size)
    total_pages = batch_size * pages_per_seq
    packing = get_dtype_packing(jnp.bfloat16)
    cache_bf16 = (
        rand((total_pages, page_size // packing, packing,
              align_to(head_dim, 128)), jnp.float32) * 40.0 - 20.0
    ).astype(jnp.bfloat16)

    if sink_value is None:
        attention_sinks = jnp.array(
            rng.random(size=(num_heads,), dtype=np.float32) * 300.0 + 200.0
        )
    else:
        attention_sinks = jnp.full((num_heads,), sink_value, jnp.float32)

    if neutral_carry:
        swa_accumution = jnp.zeros_like(q)
        swa_l = jnp.zeros((total_tokens, num_heads), jnp.float32)
        swa_m = jnp.full((total_tokens, num_heads), DEFAULT_MASK_VALUE,
                         jnp.float32)
    else:
        swa_accumution = jnp.ones_like(q) * 5000
        swa_l = jnp.full((total_tokens, num_heads), 200.0, jnp.float32)
        swa_m = jnp.full((total_tokens, num_heads), 500.0, jnp.float32)

    return dict(
        q=q,
        cache_bf16=cache_bf16,
        cache_packed=pack_dsv4_fp8_cache(cache_bf16),
        kv_lens=kv_lens,
        kv_lens_to_attend=kv_lens_to_attend,
        topk_indices=topk_indices,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        cu_q_lens=cu_q_lens,
        distribution=jnp.array(
            [num_decode_seqs, num_decode_seqs, batch_size], jnp.int32
        ),
        attention_sinks=attention_sinks,
        swa_accumution=swa_accumution,
        swa_l=swa_l,
        swa_m=swa_m,
    )


def ref_dsv4_sparse(
    q, cache_kv, kv_lens, kv_lens_to_attend, page_indices, cu_q_lens,
    distribution, attention_sinks, swa_accumution, swa_l, swa_m,
    topk_indices=None, *, sm_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):
    """Pure-JAX sparse MLA, ported from upstream's mla_test.ref_implementation.

    ``cache_kv`` is the **unquantized** bf16 cache; the DSV4 FP8 round trip is
    applied here so the reference sees what the kernel sees.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    actual_lkv_dim = q.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)))

    pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
    total_num_pages, rows, packing, _ = cache_kv.shape
    page_size = rows * packing
    kv_c_cache = quantize_dequantize_dsv4(cache_kv[..., :lkv_dim]).reshape(
        total_num_pages, page_size, lkv_dim
    )

    outputs = []
    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        kv_len = kv_lens[i]
        q_i = q[q_start:q_end]

        start = i * pages_per_seq
        indices = page_indices[start:start + cdiv(kv_len, page_size)]
        flat_kv_c = kv_c_cache[indices].reshape(-1, lkv_dim)
        k_i = v_i = flat_kv_c[:kv_len]

        attn = jnp.einsum(
            "qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32
        ) * sm_scale

        if topk_indices is not None:
            topk_i = topk_indices[q_start:q_end]
            csa = (topk_i[:, :, None] == jnp.arange(kv_len)[None, None, :]).any(
                axis=1
            )
            mask = ~csa[None, :, :]
        else:
            attend_i = kv_lens_to_attend[q_start:q_end]
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = attend_i[None, :, None] <= kv_span
        attn = jnp.where(mask, mask_value, attn)

        # Merge this pass into the carry-in with an online-softmax rescale.
        m_2 = jnp.max(attn, axis=-1, keepdims=True)
        m_1 = jnp.transpose(swa_m[q_start:q_end], (1, 0))[:, :, None]
        m = jnp.maximum(m_1, m_2)
        l_1 = jnp.transpose(swa_l[q_start:q_end], (1, 0))[:, :, None]
        L = (
            l_1 * jnp.exp(m_1 - m)
            + jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
            + jnp.exp(attention_sinks[..., None, None] - m)
        )
        acc = (
            swa_accumution[q_start:q_end]
            * jnp.transpose(jnp.exp(m_1 - m), (1, 0, 2))
            + jnp.einsum("nqk,kl->qnl", jnp.exp(attn - m), v_i)
        )
        outputs.append((acc / jnp.transpose(L, (1, 0, 2))).astype(q_i.dtype))
    return jnp.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# mla_sliding_window  (experimental/deepseek_v4/mla_swa.py)
# ---------------------------------------------------------------------------


@jax.jit(donate_argnames="cache_kv")
def update_kv_cache(new_kv, cache_kv, kv_lens, page_indices, cu_q_lens,
                    distribution):
    """Write ``new_kv`` into the paged cache, as upstream's SWA reference does."""
    actual_lkv_dim = new_kv.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if actual_lkv_dim != lkv_dim:
        new_kv = jnp.pad(new_kv, ((0, 0), (0, lkv_dim - actual_lkv_dim)))
    _, rows, packing, cache_kv_dim = cache_kv.shape
    if lkv_dim != cache_kv_dim:
        raise ValueError(f"{lkv_dim=} does not match {cache_kv_dim=}")
    page_size = rows * packing
    pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]

    def seq_loop_body(i, cache):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        def token_loop_body(j, inner):
            token_idx = kv_len - q_len + j
            page_idx = page_indices[i * pages_per_seq + token_idx // page_size]
            row = (token_idx % page_size) // packing
            col = (token_idx % page_size) % packing
            return inner.at[page_idx, row, col, ..., :lkv_dim].set(
                new_kv[q_start + j]
            )

        return jax.lax.fori_loop(0, q_len, token_loop_body, cache)

    return jax.lax.fori_loop(0, distribution[-1], seq_loop_body, cache_kv)


def create_sliding_window_inputs(
    *,
    batch_size: int = 12,
    num_heads: int = 128,
    head_dim: int = 512,
    sliding_window: int = 32,
    page_size: int = 12,
    sink_value: float | None = None,
    seed: int = 1234,
) -> dict:
    """Build the empty caches and metadata for a sliding-window MLA run.

    Both caches start zeroed and are advanced by the caller one step at a time,
    because the kernel writes ``new_kv`` into the cache it is given.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    packing = get_dtype_packing(jnp.bfloat16)
    pages_per_seq = cdiv(sliding_window * 10, page_size)
    total_pages = batch_size * pages_per_seq

    if sink_value is None:
        attention_sinks = jnp.array(
            rng.random(size=(num_heads,), dtype=np.float32) * 300.0 + 200.0
        )
    else:
        attention_sinks = jnp.full((num_heads,), sink_value, jnp.float32)

    return dict(
        rng=rng,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        page_size=page_size,
        packing=packing,
        pages_per_seq=pages_per_seq,
        attention_sinks=attention_sinks,
        page_indices=jnp.arange(total_pages, dtype=jnp.int32),
        kv_lens=jnp.zeros((batch_size,), jnp.int32),
        ref_cache=jnp.zeros(
            (total_pages, page_size // packing, packing, head_dim),
            jnp.bfloat16,
        ),
        # The kernel's own cache is one physical page longer per page and is
        # uint8-packed; page_size + 4 is upstream's own choice.
        packed_cache=jnp.zeros(
            (total_pages, (page_size + 4) // packing,
             get_dtype_packing(jnp.uint8), DSV4_CACHE_LANES),
            jnp.uint8,
        ),
    )


def ref_dsv4_sliding_window(
    q, new_kv, cache_kv, kv_lens, page_indices, cu_q_lens, distribution,
    attention_sinks, *, sliding_window: int, sm_scale: float = 1.0,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):
    """Pure-JAX sliding-window MLA, ported from upstream's mla_swa_test.

    Returns ``(out, updated_cache, L, m)``, matching the kernel.  ``L`` and
    ``m`` are the softmax denominator and running max, which the sparse kernel
    consumes as its carry-in.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    updated_cache_kv = update_kv_cache(
        new_kv, cache_kv, kv_lens, page_indices, cu_q_lens, distribution
    )
    actual_lkv_dim = q.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)))

    pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
    total_num_pages, rows, packing, _ = updated_cache_kv.shape
    page_size = rows * packing
    kv_c_cache = quantize_dequantize_dsv4(
        updated_cache_kv[..., :lkv_dim]
    ).reshape(total_num_pages, page_size, lkv_dim)

    outputs, ls, ms = [], [], []
    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        q_i = q[q_start:q_end]

        start = i * pages_per_seq
        indices = page_indices[start:start + cdiv(kv_len, page_size)]
        flat_kv_c = kv_c_cache[indices].reshape(-1, lkv_dim)
        k_i = v_i = flat_kv_c[:kv_len]

        attn = jnp.einsum(
            "qnh,kh->nqk", q_i, k_i, preferred_element_type=jnp.float32
        ) * sm_scale
        q_span = kv_len - q_len + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        attn = jnp.where(mask, mask_value, attn)

        m = jnp.max(attn, axis=-1, keepdims=True)
        L = jnp.sum(jnp.exp(attn - m), axis=-1, keepdims=True)
        l_final = L + jnp.exp(attention_sinks[..., None, None] - m)
        attn = jnp.exp(attn - m) / l_final
        outputs.append(jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype))
        ls.append(jnp.transpose(L[..., 0]))
        ms.append(jnp.transpose(m[..., 0]))

    return (
        jnp.concatenate(outputs, axis=0),
        updated_cache_kv,
        jnp.concatenate(ls, axis=0),
        jnp.concatenate(ms, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the MLA contracts and a sample input specification."
    )
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--q-len", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=1024)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--lkv-dim", type=int, default=512)
    parser.add_argument("--r-dim", type=int, default=64)
    args = parser.parse_args()

    start = time.perf_counter()
    built = create_inputs(
        num_seqs=args.num_seqs, q_len=args.q_len, kv_len=args.kv_len,
        num_q_heads=args.q_heads, lkv_dim=args.lkv_dim, r_dim=args.r_dim,
    )
    jax.block_until_ready(list(built.values()))
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline-notes",
                "contracts": list(SOURCE["contracts"]),
                "shapes": {k: list(v.shape) for k, v in built.items()},
                "head_major_ql_nope": list(
                    nine_args_head_major(built)[0].shape
                ),
                "sm_scale": 1.0 / math.sqrt(args.lkv_dim + args.r_dim),
                "logical_flops": logical_flops(
                    num_seqs=args.num_seqs, q_len=args.q_len,
                    kv_len=args.kv_len, num_q_heads=args.q_heads,
                    lkv_dim=args.lkv_dim, r_dim=args.r_dim,
                ),
                "runnable_references": [
                    "ref_dsv4_sparse", "ref_dsv4_sliding_window",
                ],
                "note": (
                    "the three mla_ragged_paged_attention contracts use "
                    "tpu_inference_optimized.ref_mla_ragged_paged_attention; "
                    "see the module docstring"
                ),
                "build_ms": elapsed_ms,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
