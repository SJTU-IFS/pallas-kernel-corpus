"""JAX references for the MoE router top-k kernels in this directory.

Three contracts, deliberately kept apart because their output layouts and their
tie/bias semantics differ.

``router_topk`` -- ``topk_pallas``::

    router_logits [batch, num_experts]
    ->            (weights[batch, topk], ids[batch, topk])

``router_biased_topk`` -- ``biased_topk_pallas``::

    router_logits   [batch, num_experts]
    correction_bias [num_experts]
    ->              (weights[batch, topk], ids[batch, topk])

    Selection ranks ``logits + correction_bias``, but the returned weights are
    the *pre-bias* logits of the selected experts.  That asymmetry is the whole
    point: the bias steers routing without contaminating the combine weights.

``router_grouped_topk`` -- ``grouped_topk_pallas``::

    router_logits   [batch, num_experts]
    correction_bias [num_experts]
    num_expert_group, topk_group, topk
    ->              (weights[batch, topk], ids[batch, topk])

    DeepSeek-style hierarchy: experts are split into ``num_expert_group``
    contiguous groups, each group scored by the sum of its two best biased
    logits, the best ``topk_group`` groups kept, and ``topk`` experts selected
    from within those groups only.

All three return ``[batch, topk]``.  Inside the kernels the Pallas grid works
in a transposed ``[topk, batch]`` layout -- the batch dimension is the lane
dimension -- and the public wrappers transpose back before returning, which is
why their internal variables are named ``weights_t``/``ids_t``.

Ties: these references use ``jax.lax.top_k``, which breaks ties toward the
lower index.  The kernels use argmax-style selection with the same rule, so
tie-heavy inputs (for example integer-valued logits) still agree.  Random
float logits make ties vanishingly unlikely in any case.

``streamindex_topk`` -- DeepSeek-V4's "lightning indexer", the one contract here
that is **not** a router::

    q               [num_tokens, num_q_heads, head_dim]
    indexer_weights [num_tokens, num_q_heads]
    cache_kv        uint8[pages, page_size // 4, 4, width]  fp8 keys + ue8m0 scales
    seq_lens        i32[max_num_seqs]   UNCOMPRESSED kv length
    page_indices, cu_q_lens, distribution
    ->              i32[num_tokens, k]  positions in COMPRESSED space

    It retrieves rather than routes: the score is ``sum_h relu(q_h . k_s) * w_h``
    with the ReLU *before* the head sum, and the result is a set of kv positions,
    not weights.  ``seq_lens`` is in uncompressed units and the kernel divides by
    ``compression_ratio`` itself -- passing an already-divided length silently
    retrieves from a prefix.  ``streamindex_topk_ref`` below is ported from
    upstream's own ``tests/kernels/deepseek_v4/test_streamindex_topk.py``, and
    ``create_streamindex_inputs`` reproduces its fp8 cache packing, because the
    kernel reads scales packed inline with the keys.

``sparsecore_key_value_topk`` -- the contract Tokamax's SparseCore ``top_k``
*appears* to implement.  Kept as a reference because the investigation below is
worth preserving, but **no SparseCore implementation is migrated**:

    keys   [rows, n]   values [rows, n]   ->  (keys[rows, k], values[rows, k])

Tokamax ships a SparseCore top-k at
``tokamax/_src/ops/experimental/tpu/topk/pallas_mosaic_tpu_kernel.py``.  It
flattens cleanly (its only non-corpus dependency is ``absl`` logging) and it
*runs* on this v6e -- ``plsc.get_sparse_core_info()`` reports 2 cores, 16
subcores, 8 lanes.  It does not reproduce ``jax.lax.top_k`` under the obvious
calling convention, measured at rows=8, n=1024, k=8 with
``use_approx_top_k=False``:

  * with all-negative keys it returns the *most* negative elements, i.e. it
    ranks by raw float bit pattern where the sign bit makes negatives large;
  * with non-negative keys its selected set still differs from ``lax.top_k``
    (it returned 99.6164 where the reference took 99.778) and its output is not
    sorted.

That may be a misuse of ``num_seq_windows`` / ``digit_width`` / the expected
input layout rather than a kernel defect -- the API has several tuning knobs
this corpus has not explored.  Until that is resolved it would be wrong to
count it as a migrated, validated launch point, so it is not in the corpus.

None of these references uses Pallas.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp


SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": (
        "router_topk",
        "router_biased_topk",
        "router_grouped_topk",
        "sparsecore_key_value_topk",
        "streamindex_topk",
    ),
}


def router_topk(
    router_logits: jax.Array, *, topk: int
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for ``topk_pallas``."""
    weights, ids = jax.lax.top_k(router_logits.astype(jnp.float32), topk)
    return weights, ids.astype(jnp.int32)


def router_biased_topk(
    router_logits: jax.Array, correction_bias: jax.Array, *, topk: int
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for ``biased_topk_pallas``.

    Ranks on ``logits + bias`` and returns the pre-bias weights of the winners.
    """
    logits = router_logits.astype(jnp.float32)
    biased = logits + correction_bias.astype(jnp.float32)
    _, ids = jax.lax.top_k(biased, topk)
    weights = jnp.take_along_axis(logits, ids, axis=-1)
    return weights, ids.astype(jnp.int32)


def router_grouped_topk(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for ``grouped_topk_pallas``."""
    batch, num_experts = router_logits.shape
    if num_experts % num_expert_group:
        raise ValueError(f"{num_experts=} is not divisible by {num_expert_group=}")
    per_group = num_experts // num_expert_group

    logits = router_logits.astype(jnp.float32)
    biased = logits + correction_bias.astype(jnp.float32)

    # Score each group by the sum of its two largest biased logits.
    grouped = biased.reshape(batch, num_expert_group, per_group)
    best_two = jax.lax.top_k(grouped, 2)[0].sum(axis=-1)  # [batch, groups]
    _, kept = jax.lax.top_k(best_two, topk_group)  # [batch, topk_group]

    group_mask = jnp.zeros((batch, num_expert_group), jnp.bool_)
    group_mask = group_mask.at[
        jnp.arange(batch)[:, None], kept
    ].set(True)
    expert_mask = jnp.repeat(group_mask, per_group, axis=-1)

    masked = jnp.where(expert_mask, biased, -jnp.inf)
    _, ids = jax.lax.top_k(masked, topk)
    weights = jnp.take_along_axis(logits, ids, axis=-1)
    return weights, ids.astype(jnp.int32)


def sparsecore_key_value_topk(
    keys: jax.Array, values: jax.Array, *, k: int
) -> tuple[jax.Array, jax.Array]:
    """Pure-JAX reference for the Tokamax SparseCore ``top_k``.

    Ranks by ``keys`` descending and carries ``values`` along, rather than
    returning indices the way ``jax.lax.top_k`` does.
    """
    top_keys, indices = jax.lax.top_k(keys, k)
    return top_keys, jnp.take_along_axis(values, indices, axis=-1)


kernel = router_biased_topk
workload = router_biased_topk


# ---------------------------------------------------------------------------
# streamindex_topk  (DeepSeek-V4 lightning indexer)
#
# Ported from tpu-inference tests/kernels/deepseek_v4/test_streamindex_topk.py
# at commit 8b9c90928c94c7230d1bc891534a301510a6a30d.  A port, not a
# re-derivation: the fp8 cache packing is part of the contract, and getting it
# subtly wrong would produce false failures rather than a caught bug.
# ---------------------------------------------------------------------------


def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    return q, jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)


def streamindex_topk_ref(
    q, weights, kv, block_table, T_list, S_list, cu_q_lens, k, comp_ratio,
    H_I, H_KV,
):
    """Naive NumPy ground truth for the lightning indexer.

    ``kv`` is the **dequantized** float32 cache, so the reference sees the same
    values the kernel reads back out of its fp8 records.
    """
    import numpy as np

    num_tokens = q.shape[0]
    expected = np.full((num_tokens, k), -1, dtype=np.int32)
    for b in range(len(T_list)):
        T_seq, S_total, q_start = T_list[b], S_list[b], cu_q_lens[b]
        if T_seq == 0:
            continue
        S_valid = S_total // comp_ratio
        seq_kv = np.concatenate([kv[p] for p in block_table[b]], axis=0)
        scores = np.full((T_seq, max(S_valid, k)), -np.inf, np.float32)
        for t_idx in range(T_seq):
            global_t = q_start + t_idx
            q_abs_pos = (S_total - T_seq) + t_idx
            for s_idx in range(S_valid):
                if s_idx * comp_ratio > q_abs_pos:  # causal, in compressed units
                    continue
                score = 0.0
                for h in range(H_I):
                    h_kv = h // (H_I // H_KV)  # GQA: several q heads per kv head
                    inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
                    score += max(0.0, inner) * weights[global_t, h]
                scores[t_idx, s_idx] = score
        picked = np.argsort(-scores, axis=-1)[:, :k]
        for t_idx in range(T_seq):
            for i in range(k):
                idx = picked[t_idx, i]
                expected[q_start + t_idx, i] = (
                    -1 if scores[t_idx, idx] == -np.inf else idx)
    return expected


def create_streamindex_inputs(
    *,
    T_list=(3, 3),
    S_list=(4, 6),
    page_size: int = 16,
    num_q_heads: int = 4,
    num_kv_heads: int = 1,
    head_dim: int = 64,
    k: int = 512,
    compression_ratio: int = 2,
    block_table=None,
    seed: int = 42,
):
    """Build kernel inputs and the matching dequantized cache for the reference.

    Returns ``(kernel_kwargs, reference_kwargs)``.  The cache is packed exactly
    as upstream's test packs it: per (page, slot, kv-head), an fp8_e4m3fn record
    of the key followed by its ue8m0 block scale, padded to a 128-byte multiple,
    written at ``[page, slot // 4, slot % 4]``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    T_list, S_list = list(T_list), list(S_list)
    batch = len(T_list)
    if block_table is None:
        pages_per_seq = max(1, -(-max(S_list) // page_size)) + 1
        block_table = [
            list(range(b * pages_per_seq, (b + 1) * pages_per_seq))
            for b in range(batch)
        ]
    block_table = np.array(block_table, np.int32)

    num_tokens = sum(T_list)
    q = rng.standard_normal((num_tokens, num_q_heads, head_dim), np.float32)
    weights = rng.uniform(-1.5, 1.5, (num_tokens, num_q_heads)).astype(np.float32)

    max_page = int(np.max(block_table))
    float32_kv = rng.standard_normal(
        (max_page + 1, page_size, num_kv_heads, head_dim), np.float32)

    aligned = ((head_dim + 127) // 128) * 128
    width = (((aligned + aligned // 128) + 127) // 128) * 128
    cache_kv = np.zeros((max_page + 1, page_size // 4, 4, width), np.uint8)
    dequantized = np.zeros_like(float32_kv)
    for p in range(max_page + 1):
        for s in range(page_size):
            for h in range(num_kv_heads):
                q_fp8, scale = quantize_fp8_ue8m0(
                    jnp.asarray(float32_kv[p, s, h]), head_dim)
                dequantized[p, s, h] = (
                    np.asarray(q_fp8).astype(np.float32) * float(scale[0]))
                record = np.concatenate([
                    np.asarray(_to_byte_lane(q_fp8)),
                    np.asarray(_to_byte_lane(scale)),
                ], axis=-1)
                cache_kv[p, s // 4, s % 4] = np.pad(
                    record, (0, width - record.shape[-1]))

    cu_q_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)
    num_decodes = 0
    while num_decodes < batch and T_list[num_decodes] == 1:
        num_decodes += 1

    kernel_kwargs = dict(
        q=jnp.asarray(q),
        indexer_weights=jnp.asarray(weights),
        cache_kv=jnp.asarray(cache_kv),
        seq_lens=jnp.asarray(np.asarray(S_list, np.int32)),
        page_indices=jnp.asarray(block_table.flatten()),
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=(num_decodes, num_decodes, batch),
    )
    reference_kwargs = dict(
        q=q, weights=weights, kv=dequantized, block_table=block_table,
        T_list=T_list, S_list=S_list, cu_q_lens=cu_q_lens, k=k,
        comp_ratio=compression_ratio, H_I=num_q_heads, H_KV=num_kv_heads,
    )
    return kernel_kwargs, reference_kwargs


def create_inputs(
    *,
    batch: int = 4096,
    num_experts: int = 256,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Router logits plus a per-expert correction bias."""
    keys = jax.random.split(jax.random.key(seed), 2)
    router_logits = jax.random.normal(keys[0], (batch, num_experts), dtype=dtype)
    correction_bias = jax.random.normal(
        keys[1], (num_experts,), dtype=jnp.float32
    ) * 0.1
    return router_logits, correction_bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        choices=("topk", "biased", "grouped"),
        default="biased",
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--expert-groups", type=int, default=8)
    parser.add_argument("--topk-group", type=int, default=4)
    args = parser.parse_args()

    router_logits, correction_bias = create_inputs(
        batch=args.batch, num_experts=args.experts
    )
    if args.contract == "topk":
        compiled = jax.jit(router_topk, static_argnames="topk")
        call = lambda: compiled(router_logits, topk=args.topk)  # noqa: E731
    elif args.contract == "biased":
        compiled = jax.jit(router_biased_topk, static_argnames="topk")
        call = lambda: compiled(  # noqa: E731
            router_logits, correction_bias, topk=args.topk
        )
    else:
        compiled = jax.jit(
            router_grouped_topk,
            static_argnames=("num_expert_group", "topk_group", "topk"),
        )
        call = lambda: compiled(  # noqa: E731
            router_logits,
            correction_bias,
            num_expert_group=args.expert_groups,
            topk_group=args.topk_group,
            topk=args.topk,
        )

    start = time.perf_counter()
    weights, ids = call()
    weights.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": f"router_{args.contract}",
                "weights_shape": list(weights.shape),
                "ids_shape": list(ids.shape),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
