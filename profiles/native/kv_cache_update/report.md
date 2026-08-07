# Paged KV-Cache Update TPU v6e Analysis

Generated: 2026-07-31 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 inputs, JIT compile, 5 warmups, 50 wall-clock diagnostics, then
50 device-profiled iterations, each in its own trace chunk.

**This is a zero-FLOP kernel.** It moves bytes and computes nothing, so MXU
utilization is reported as `null` rather than a misleading zero, and the metric
is achieved HBM bandwidth.

## 1. Scope and contract

| Source | Upstream path | Migrated | Audited there |
|---|---|---:|---:|
| vLLM tpu-inference | `ragged_paged_attention/v2/ragged_kv_cache_update.py` | 1 launch | 1 |
| sglang-jax | `update_kv_cache/update_kv_cache.py` | 2 launches | 2 |

sglang-jax's file contains two `pallas_call` sites — the kernel and its
`shard_map` wrapper — which the audit counts separately; both are migrated
together as one file.

Both share the contract `kv_cache_update`:

```text
new_kv     [total_num_tokens, num_combined_kv_heads, head_dim]
slices     [3, padded_num_slices] int32, rows are
           (kv_cache_start, new_kv_start, slice_len)
kv_cache   [total_num_pages * page_size, num_combined_kv_heads, head_dim]
num_slices [1] int32
->         for every i < num_slices[0]:
           kv_cache[kv_cache_start_i : +len_i] = new_kv[new_kv_start_i : +len_i]
```

The signatures differ around that core: tpu-inference defaults `page_size=32`,
derives `num_slices_per_block` from a VMEM budget, and runs unsharded when
`mesh` is None; sglang-jax defaults `page_size=1`, fixes
`num_slices_per_block=8`, and is *always* wrapped in `jax.shard_map`, so it
needs an active mesh even on one device. Both are profiled at matched
`page_size=32, num_slices_per_block=8`.

Unlike the grouped-matmul family, these are **genuinely different codebases**:
of eight and five top-level definitions they share exactly one name
(`kv_cache_update`), and that one differs. They converge on the same
performance because both are DMA-bound doing identical copies, not because one
was adapted from the other.

### No native shape

Neither repository ships a benchmark configuration for this kernel. Every run
here is therefore a **declared validation shape** with
`native_source_shape: false`, chosen to match the ragged-paged-attention
family's Llama-3.1-70B head count:

```text
total_num_tokens=4096  num_combined_kv_heads=16  head_dim=128
total_num_pages=1024   page_size=32              num_slices=128
```

## 2. Correctness — bitwise, not approximate

A copy kernel that is only approximately right is wrong, so the bar is exact
equality with the pure-JAX reference, not a cosine threshold.

| Implementation | vs `baseline.kv_cache_update` | Cosine | Max abs |
|---|---|---:|---:|
| tpu-inference | **bitwise identical** | 1.0000000000 | 0.0 |
| sglang-jax | **bitwise identical** | 1.0000000000 | 0.0 |

Verified at the validation shape and at 8, 24, and 32 live slices with a
32-column padded slice list, plus a check that columns at or past
`num_slices[0]` leave their cache pages untouched.

Both kernels **donate `kv_cache`**: the input buffer is consumed, and touching
it after the call raises `Array has been deleted`. `tests/test_kv_cache_update_tpu.py`
pins that behaviour so it cannot regress silently.

## 3. Timing

| Configuration | Median (ms) | Std (ms) | Speedup |
|---|---:|---:|---:|
| pure-JAX baseline | 0.982114 | 0.000723 | 1.000× |
| sglang-jax | 0.237664 | 0.000193 | **4.132×** |
| tpu-inference | 0.237768 | 0.000155 | **4.131×** |

The two kernels are 0.04% apart.

## 4. The headline bandwidth number needs a caveat

Two byte counts are recorded per run, because they differ by 8.5× and reporting
either alone would mislead:

| Convention | Bytes | Achieved at 0.2378 ms |
|---|---:|---:|
| `slice_payload` — what the algorithm requires | 33,554,432 | 141.1 GB/s |
| `harness_cache_copy` — what this harness moves | 285,212,672 | **1199.5 GB/s** |

`harness_cache_copy` is the primary convention. The reason: the profiler reuses
its inputs across the 50 iterations, so donation of `kv_cache` **fails** (JAX
emits "Some donated buffers were not usable") and XLA copies the entire cache
on every call.

This was confirmed rather than assumed. Holding the payload fixed at 8.4 MB and
varying only the cache size:

| `total_num_pages` | Cache (MB) | Device median (ms) | Implied copy bandwidth |
|---:|---:|---:|---:|
| 32 | 4.2 | 0.019960 | 420 GB/s |
| 256 | 33.6 | 0.060218 | 1114 GB/s |
| 1024 | 134.2 | 0.198257 | 1354 GB/s |
| 4096 | 536.9 | 0.750103 | 1432 GB/s |

Device time scales linearly with cache size at constant payload, asymptoting to
1.43 TB/s — v6e's HBM read+write copy bandwidth. The measurement is dominated
by the cache copy, not by the update.

So: **141 GB/s is not this kernel's bandwidth.** It is the payload divided by
the time to copy a cache 8× larger than the payload. In a serving loop, where
the cache is genuinely donated and updated in place, the copy does not happen.

## 5. Bottleneck analysis

Per-operation self time from the first trace chunk settles it:

| Configuration | Operation | Occurrences | Time | Share |
|---|---|---:|---:|---:|
| tpu-inference | `copy.4` (forced cache copy) | 1 | 188.976 µs | 79.31% |
| tpu-inference | **`pallas_call`** | 1 | **41.995 µs** | **17.62%** |
| tpu-inference | `copy-done` | 1 | 4.466 µs | 1.87% |
| sglang-jax | `copy.4` | 1 | 188.286 µs | 79.11% |
| sglang-jax | **`pallas_call`** | 1 | **41.920 µs** | **17.61%** |
| pure-JAX baseline | `while.3` | 1 | 797.685 µs | 46.72% |
| pure-JAX baseline | `dynamic_update_slice` | 128 | 195.919 µs | 19.96% |
| pure-JAX baseline | `copy.8` | 1 | 183.917 µs | 18.74% |
| pure-JAX baseline | `and` | 128 | 75.139 µs | 7.65% |

Reading this properly:

- **The kernel itself takes ~42 µs**, not 238 µs. Moving 33.55 MB of payload in
  41.995 µs is **799 GB/s** — roughly half of v6e's peak HBM bandwidth for a
  scattered, page-granular gather/scatter, which is a respectable result for
  ragged 32-token slices.
- The 189 µs `copy.4` is the harness artifact, and it independently reproduces
  the 1.43 TB/s ceiling: 268.4 MB of cache traffic in 188.976 µs is 1420 GB/s.
- **Comparing kernel work only**, the Pallas kernels beat the pure-JAX
  baseline by roughly **27×** (42 µs against ~1130 µs of loop, mask and
  dynamic-slice work), not the 4.13× the end-to-end medians suggest. The
  end-to-end figure is diluted by a copy both sides pay.
- The baseline's cost is structural: a 128-trip loop where each trip does a
  masked `dynamic_slice` + `dynamic_update_slice` over the whole cache. Its
  `while` alone is 4× the entire Pallas call.

Suggested next work:

1. Measure with donation actually succeeding — build fresh inputs per iteration
   inside the timed region, or profile a serving-shaped loop — to get the
   kernel's end-to-end cost without the copy. That requires a profiler mode
   that regenerates inputs, which the current protocol deliberately does not do.
2. Sweep `num_slices_per_block` (fixed at 8 here; tpu-inference's own heuristic
   would pick up to 64 from its VMEM budget).
3. Sweep `page_size`: sglang-jax defaults to 1, which would make every slice a
   single token and change the DMA pattern completely.
4. Migrate the three related fusion kernels audited under this category —
   tpu-inference's `compress_and_store` and `proj_and_save_state`, which fuse
   normalisation and RoPE into the store.

## 6. Trace and artifact locations

Raw traces remain on the TPU:

```text
profiles/native/kv_cache_update/<run>/trace/
```

Compact local artifacts are the three `result.json` files beside this report
and the `kv-*` keys in `profiles/native/xplane_summary.json`.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in kv-baseline kv-tpu-inference kv-sglang-jax; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --trace-chunk-size 1 --output-dir profiles/native
done

# The cache-size experiment that identified the harness copy.
for pages in 32 256 1024 4096; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation kv-tpu-inference --kv-slices 32 --kv-tokens 1024 \
    --kv-pages $pages --run-label pages$pages --iterations 20 \
    --trace-chunk-size 1 --output-dir /tmp/kv_cache_size
done

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native

uv run --frozen --with pytest python -m pytest \
  tests/test_kv_cache_update_tpu.py -q
```
