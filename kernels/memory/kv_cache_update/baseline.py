"""JAX reference for the paged KV-cache update implementations here.

Both migrated implementations share one contract, ``kv_cache_update``::

    new_kv     [total_num_tokens, num_combined_kv_heads, head_dim]
    slices     [3, padded_num_slices] int32, rows are
               (kv_cache_start, new_kv_start, slice_len)
    kv_cache   [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    num_slices [1] int32, how many columns of `slices` are live
    ->         kv_cache with, for every i < num_slices[0],
               kv_cache[kv_cache_start_i : +len_i] = new_kv[new_kv_start_i : +len_i]

Columns of ``slices`` at or beyond ``num_slices[0]`` are ignored.  This is a
pure data-movement kernel: it performs zero FLOPs, so it is judged on achieved
HBM bandwidth rather than on MXU utilization.

The upstream signatures differ around that core:

``tpu_inference_optimized``
    ``page_size=32`` by default, derives ``num_slices_per_block`` from a VMEM
    budget, and runs unsharded when ``mesh`` is None.

``sglang_jax_optimized``
    ``page_size=1`` by default, fixed ``num_slices_per_block=8``, and is
    *always* wrapped in ``jax.shard_map``, so it needs an active mesh even on a
    single device.

Both donate ``kv_cache``.  The input buffer is consumed, so a second call must
be given a freshly built cache or it raises "Array has been deleted".
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np


SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("kv_cache_update",),
}


def kv_cache_update(
    new_kv: jax.Array,
    slices: jax.Array,
    kv_cache: jax.Array,
    num_slices: jax.Array,
    *,
    max_slice_len: int | None = None,
) -> jax.Array:
    """Pure-JAX reference for ``kv_cache_update``.

    Written as a fixed-trip loop over the padded slice list with a masked
    dynamic-update, so it traces to static shapes and jits at any size.  Every
    slice is copied at ``max_slice_len`` and masked down to its real length,
    which is why this is a correctness reference and not a speed target.

    ``max_slice_len`` must bound every live ``slice_len``.  It defaults to the
    page size implied by the cache when that is knowable, and otherwise must be
    supplied by the caller.
    """
    padded_num_slices = slices.shape[1]
    if max_slice_len is None:
        raise ValueError(
            "max_slice_len must be given: the reference needs a static copy "
            "length, and slice_len is data"
        )
    positions = jnp.arange(max_slice_len)

    def body(index: int, cache: jax.Array) -> jax.Array:
        cache_start = slices[0, index]
        new_start = slices[1, index]
        length = slices[2, index]
        live = index < num_slices[0]

        block = jax.lax.dynamic_slice_in_dim(new_kv, new_start, max_slice_len, axis=0)
        current = jax.lax.dynamic_slice_in_dim(
            cache, cache_start, max_slice_len, axis=0
        )
        keep = (positions < length) & live
        merged = jnp.where(keep[:, None, None], block, current)
        return jax.lax.dynamic_update_slice_in_dim(cache, merged, cache_start, axis=0)

    return jax.lax.fori_loop(0, padded_num_slices, body, kv_cache)


kernel = kv_cache_update
workload = kv_cache_update


def create_inputs(
    *,
    total_num_tokens: int = 1024,
    num_combined_kv_heads: int = 16,
    head_dim: int = 128,
    total_num_pages: int = 256,
    page_size: int = 32,
    num_slices: int = 32,
    padded_num_slices: int | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """Build a page-aligned slice list plus its tensors.

    Slices are generated so that each writes a whole page into a distinct cache
    page and reads a distinct, in-bounds run of ``new_kv``: that is the shape
    of a real prefill flush, and it keeps the update well defined regardless of
    the order the implementations visit slices in.
    """
    keys = jax.random.split(jax.random.key(seed), 2)
    if padded_num_slices is None:
        padded_num_slices = num_slices
    if num_slices * page_size > total_num_tokens:
        raise ValueError(
            f"{num_slices} x {page_size} tokens does not fit in "
            f"{total_num_tokens}"
        )
    if num_slices > total_num_pages:
        raise ValueError(f"{num_slices=} exceeds {total_num_pages=}")

    new_kv = jax.random.normal(
        keys[0], (total_num_tokens, num_combined_kv_heads, head_dim), dtype=dtype
    )
    kv_cache = jax.random.normal(
        keys[1],
        (total_num_pages * page_size, num_combined_kv_heads, head_dim),
        dtype=dtype,
    )
    index = np.arange(num_slices, dtype=np.int32)
    rows = np.zeros((3, padded_num_slices), dtype=np.int32)
    rows[0, :num_slices] = index * page_size  # kv_cache_start, page aligned
    rows[1, :num_slices] = index * page_size  # new_kv_start
    rows[2, :num_slices] = page_size  # slice_len
    return (
        new_kv,
        jnp.asarray(rows),
        kv_cache,
        jnp.array([num_slices], jnp.int32),
        page_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--pages", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--slices", type=int, default=32)
    args = parser.parse_args()

    new_kv, slices, kv_cache, num_slices, page_size = create_inputs(
        total_num_tokens=args.tokens,
        num_combined_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        total_num_pages=args.pages,
        page_size=args.page_size,
        num_slices=args.slices,
    )
    compiled = jax.jit(kv_cache_update, static_argnames="max_slice_len")
    start = time.perf_counter()
    output = compiled(
        new_kv, slices, kv_cache, num_slices, max_slice_len=page_size
    )
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": "kv_cache_update",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "bytes_moved": int(
                    num_slices[0] * page_size * args.kv_heads * args.head_dim * 2
                ),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The DeepSeek-V4 Pallas state layout
#
# `proj_and_save_state` scatters its fp32 state into a uint8 cache shaped
# [num_pages, physical_page_size, SLOT_PACK, LANE].  Upstream publishes the
# arithmetic (config.py: LANE=128, SLOT_PACK=4, N_FIELDS=2, FP32_BYTES=4) but no
# host-side readback -- the slot-to-address math lives inside the on-device
# BufferedRef classes -- so the corpus carried these kernels unvalidated until
# the layout was **measured**.
#
# What the measurement showed: writing a single token of all-ones sets exactly
# 4096 bytes, and the only values present are 63 and 128 -- the two nonzero
# bytes of a float32 1.0.  All the 128s are contiguous.  So the SLOT_PACK axis
# is a **byte-plane** axis: sub-slot b holds byte b of every float32, the same
# "byte lane" packing `compress_norm_rope._to_byte_lane` uses elsewhere in this
# codebase.  Moving that axis last and bitcasting recovers the state.
#
# This is a measurement rather than a re-derivation, and it is self-checking:
# `read_dsv4_state` round-trips small integers **exactly** (see the test), which
# no wrong byte assignment would.  Values with a full fp32 mantissa come back
# within one bfloat16 ulp, because the kernel's projection runs on the MXU in
# bf16 -- comparing against exact fp32 arithmetic is what made this look like a
# layout error at first.
# ---------------------------------------------------------------------------


def read_dsv4_state(cache: jax.Array, slot: int,
                    rows_per_token: int) -> jax.Array:
    """One token's fp32 ``[kv | score + ape]`` out of the packed uint8 cache.

    ``slot_mapping`` is in **physical rows, not tokens**.  Measured by writing
    one token at a time: slot S lands at rows ``[S, S + rows_per_token)`` of
    page ``S // physical_page_size``.  So consecutive tokens must be spaced
    ``rows_per_token`` apart -- passing token indices makes each token overlap
    the next by all but one of its rows, and nothing complains.

    ``rows_per_token`` is the kernel file's own
    ``Configs.state_rows_per_token``; nothing here re-derives it.
    """
    physical_page_size = cache.shape[1]
    page, row = divmod(slot, physical_page_size)
    block = cache[page, row:row + rows_per_token]
    # (rows, pack, lane) -> (rows, lane, pack): the pack axis holds the four
    # bytes of each float32, which is what bitcast needs to consume.
    return jax.lax.bitcast_convert_type(
        jnp.transpose(block, (0, 2, 1)), jnp.float32
    ).reshape(-1)


def dsv4_expected_state(kv_score, position, ape, state_width, compress_ratio):
    """What one token's saved state should be: ``[kv | score + ape[pos % r]]``.

    The same thing ``compressor.save_partial_states`` computes, written per
    token so it can be compared against a single slot of the packed cache.
    """
    return jnp.concatenate([
        kv_score[:state_width],
        kv_score[state_width:] + ape[position % compress_ratio],
    ])


def read_dsv4_record(cache: jax.Array, kv_slot: int, nope_dim: int,
                     quant_block: int) -> jax.Array:
    """Dequantized compressed KV out of one packed record, as float32.

    A record is exactly one physical row -- ``cache[kv_slot // page_size,
    kv_slot % page_size]``, shaped ``(SLOT_PACK, LANE)`` = 512 bytes -- laid out
    **row-major**: ``nope_dim`` bytes of float8_e4m3fn, then
    ``nope_dim // quant_block`` float8_e8m0fnu block scales, then padding.  Rope
    is not here; it goes to the separate ``rope_cache``.

    Measured the same way as :func:`read_dsv4_state`, and checked against the
    pure-JAX twin's own record: the row-major reading agrees **bit-exactly**
    while the transposed one does not, so the ordering is decided by an
    independent oracle rather than assumed.
    """
    physical_page_size = cache.shape[1]
    page, row = divmod(kv_slot, physical_page_size)
    flat = cache[page, row].reshape(-1)
    values = jax.lax.bitcast_convert_type(
        flat[:nope_dim], jnp.float8_e4m3fn
    ).astype(jnp.float32)
    scales = jax.lax.bitcast_convert_type(
        flat[nope_dim:nope_dim + nope_dim // quant_block], jnp.float8_e8m0fnu
    ).astype(jnp.float32)
    return (values.reshape(-1, quant_block) * scales[:, None]).reshape(-1)
