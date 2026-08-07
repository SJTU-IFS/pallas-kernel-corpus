"""JAX references for the ragged-paged-attention implementations here.

Two upstream contracts live in this directory and they are not interchangeable,
so each gets its own named reference rather than one blended API.

``rpa_v2`` -- JAXBench 7p, vLLM tpu-inference v2
    ``f(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)``.  K and V are
    interleaved along one axis of a single paged cache, and the number of live
    sequences is carried by a one-element ``num_seqs`` array.

``rpa_v3`` -- vLLM tpu-inference v3, sglang-jax v3
    ``f(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens,
    distribution)``.  K and V arrive separately to be appended to the cache, and
    ``distribution = (i, j, k)`` marks sequences ``[0:i]`` decode-only,
    ``[i:j]`` chunked-prefill-only, and ``[j:k]`` mixed, with ``k`` the total
    sequence count.  ``page_indices`` is flattened rather than 2-D, and
    ``kv_cache`` uses a 5-D dtype-packed layout::

        [total_num_pages, page_size,
         align_to(2 * num_kv_heads, packing) // packing, packing,
         align_to(head_dim, 128)]

    where ``packing = 32 // dtype_bits`` (2 for bf16).  The kernel can also
    update the cache in place.  None of this is expressible through the v2
    signature, so the two contracts are kept apart.

The v2 reference below is derived from the pure-JAX ``workload`` that JAXBench
ships as its own baseline for this benchmark, so the speedup denominator is the
one the upstream benchmark itself uses.  It contains no Pallas call.

The two v3 implementations agree on the packed cache layout -- both report
``[64, 16, 2, 2, 128]`` for ``get_kv_cache_shape(64, 16, 2, 128, bf16)`` -- but
their entry points are NOT drop-in compatible:

    tpu-inference v3   8 required args, ending ``cu_q_lens, distribution``
    sglang-jax v3     10 required args: adds ``cu_kv_lens`` before
                      ``distribution`` and a required ``custom_mask``

and sglang-jax's bundled ``ref_ragged_paged_attention`` is a *third* contract
again -- ``(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens,
num_seqs)``, with K and V as separate page arrays -- so it is not a reference
for sglang-jax's own v3 kernel.  ``rpa_v3`` therefore names a shared *cache
layout and semantics*, not a shared signature.

For v3 this file deliberately provides no reference.  tpu-inference v3 ships an
upstream ``ref_ragged_paged_attention`` that is pure JAX (verified: no
``pallas`` or ``pl.`` reference in the function body) and matches its own
kernel's signature, and re-deriving the packed cache layout here would risk a
subtly wrong reference producing false failures.  The corpus instead validates
sglang-jax's kernel against *tpu-inference's* pure-JAX reference -- a genuine
cross-repository check, since the two were written independently.  See
``tests/test_ragged_paged_attention_tpu.py``.

One trap worth knowing: the v3 kernels donate the ``kv_cache`` buffer, because
``update_kv_cache`` defaults to True.  Reusing the same cache array for a
second call raises "Array has been deleted"; build fresh inputs per call.

Shapes for ``rpa_v2``::

    q            [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages     [total_num_pages, page_size, 2 * num_kv_heads, head_dim]
                 K is [..., 0::2, :] and V is [..., 1::2, :]
    kv_lens      [max_num_seqs]          int32, KV length of each sequence
    page_indices [max_num_seqs, pages_per_seq]  int32, page table
    cu_q_lens    [max_num_seqs + 1]      int32, cumulative query lengths
    num_seqs     [1]                     int32, sequences actually present
    out          [max_num_batched_tokens, num_q_heads, head_dim]

Masking is causal and *right-aligned*: sequence i's queries are its last
``q_len`` tokens, so query row ``r`` attends to KV positions
``[0, kv_len - q_len + r]``.  Rows of ``q`` belonging to sequences at or beyond
``num_seqs`` are not attended and their output is zero.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp


SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("rpa_v2",),
}

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def rpa_v2(
    queries: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    tokens_per_seq: int | None = None,
) -> jax.Array:
    """Pure-JAX reference for the ``rpa_v2`` contract.

    Static shapes and masking are used instead of data-dependent slicing so the
    whole thing is jittable at the native workload size.  This requires every
    sequence to contribute the same number of query tokens, which is how
    JAXBench's own baseline and native input generator are written; uneven
    ``cu_q_lens`` are rejected rather than silently mis-sliced.
    """
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    num_query_per_kv = num_q_heads // num_kv_heads
    max_seqs = kv_lens.shape[0]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    if tokens_per_seq is None:
        tokens_per_seq = queries.shape[0] // max_seqs
    if tokens_per_seq * max_seqs != queries.shape[0]:
        raise ValueError(
            f"{queries.shape[0]=} must be {max_seqs} x {tokens_per_seq}; this "
            "reference only covers the equal-query-length case"
        )

    outputs = []
    for i in range(max_seqs):
        q_start = cu_q_lens[i]
        kv_len = kv_lens[i]
        indices = page_indices[i]

        q = jax.lax.dynamic_slice(
            queries, (q_start, 0, 0), (tokens_per_seq, num_q_heads, head_dim)
        )
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= sm_scale

        # Right-aligned causal mask: the queries are the sequence's last tokens.
        q_span = (kv_len - tokens_per_seq) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        attn = jnp.where((q_span < kv_span) | (kv_span >= kv_len), mask_value, attn)

        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(jnp.where(i < num_seqs[0], out, 0.0))

    return jnp.concatenate(outputs, axis=0)


workload = rpa_v2
kernel = rpa_v2


# JAXBench 7p_Ragged_Paged_Attention CONFIG, Llama-3.1-70B serving shapes.
NATIVE_CONFIG = {
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 64,
    "num_q_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,
    "page_size": 16,
    "pages_per_seq": 256,
}


def create_inputs(
    *,
    max_num_batched_tokens: int = 4096,
    max_num_seqs: int = 64,
    num_q_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    pages_per_seq: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, ...]:
    """Build ``rpa_v2`` inputs; defaults are the JAXBench native configuration."""
    keys = jax.random.split(jax.random.key(42), 2)
    total_num_pages = max_num_seqs * pages_per_seq
    tokens_per_seq = max_num_batched_tokens // max_num_seqs
    q = jax.random.normal(
        keys[0], (max_num_batched_tokens, num_q_heads, head_dim), dtype=dtype
    )
    kv_pages = jax.random.normal(
        keys[1],
        (total_num_pages, page_size, 2 * num_kv_heads, head_dim),
        dtype=dtype,
    )
    kv_lens = jnp.full((max_num_seqs,), pages_per_seq * page_size, dtype=jnp.int32)
    page_indices = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(
        max_num_seqs, pages_per_seq
    )
    cu_q_lens = jnp.arange(max_num_seqs + 1, dtype=jnp.int32) * tokens_per_seq
    num_seqs = jnp.array([max_num_seqs], dtype=jnp.int32)
    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def logical_flops(
    *,
    max_num_seqs: int,
    num_q_heads: int,
    head_dim: int,
    tokens_per_seq: int,
    kv_len: int,
) -> int:
    """Full-rectangle QK + PV FLOPs, matching JAXBench's ``get_flops``."""
    return max_num_seqs * num_q_heads * (4 * tokens_per_seq * kv_len * head_dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", type=int, default=4)
    parser.add_argument("--tokens-per-seq", type=int, default=32)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--pages-per-seq", type=int, default=8)
    args = parser.parse_args()

    inputs = create_inputs(
        max_num_batched_tokens=args.seqs * args.tokens_per_seq,
        max_num_seqs=args.seqs,
        num_q_heads=args.q_heads,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        pages_per_seq=args.pages_per_seq,
    )
    compiled = jax.jit(rpa_v2)
    start = time.perf_counter()
    output = compiled(*inputs)
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": "rpa_v2",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
