# Ragged Paged Attention Native-Shape TPU v6e Analysis

Generated: 2026-07-31 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 inputs, JIT compile, 5 warmups, 50 wall-clock diagnostics, then
50 device-profiled iterations, each captured in its own trace chunk to avoid
Perfetto event-count truncation.

MXU utilization follows the JAXBench paper convention of 918 bf16 TFLOP/s;
XPlane independently reports the device peak as 946.7 TFLOP/s and an HBM ridge
point of 577.96 FLOP/byte.

## 1. Two contracts, not one family-wide API

Four implementations are migrated across three repositories. They do **not**
share one signature, and forcing them into one would misrepresent all four.

| Contract | Source | Upstream path | Migrated |
|---|---|---|---|
| `rpa_v2` | JAXBench 7p | `benchmark/7p_Ragged_Paged_Attention/optimized.py` | `ragged_paged_attention` |
| `rpa_v2` | vLLM tpu-inference | `ragged_paged_attention/v2/kernel.py` | `ragged_paged_attention` |
| `rpa_v3` | vLLM tpu-inference | `ragged_paged_attention/v3/kernel.py` | `ragged_paged_attention` |
| `rpa_v3` | sglang-jax | `ragged_paged_attention/ragged_paged_attention_v3.py` | `ragged_paged_attention` |

**`rpa_v2`** — `f(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)`.
K and V are interleaved on one axis of a single paged cache (`[..., 0::2, :]`
and `[..., 1::2, :]`), `page_indices` is 2-D, and the live sequence count is a
one-element array. Masking is causal and *right-aligned*: a sequence's queries
are its last `q_len` tokens, so query row `r` attends to KV positions
`[0, kv_len - q_len + r]`.

**`rpa_v3`** — K and V arrive separately to be appended to a 5-D dtype-packed
cache, `page_indices` is flattened, and `distribution = (i, j, k)` marks
sequences `[0:i]` decode-only, `[i:j]` chunked-prefill-only, `[j:k]` mixed.
The cache layout is
`[total_num_pages, page_size, align_to(2*H_kv, packing)//packing, packing, align_to(D,128)]`
with `packing = 32 // dtype_bits`.

Within `rpa_v3` the two implementations agree on that cache layout — both
return `[64, 16, 2, 2, 128]` from
`get_kv_cache_shape(64, 16, 2, 128, bf16)` — but **not on the signature**:

| | required positional args |
|---|---:|
| tpu-inference v3 | 8, ending `cu_q_lens, distribution` |
| sglang-jax v3 | 10, adding `cu_kv_lens` and a required `custom_mask` |

and sglang-jax's own bundled `ref_ragged_paged_attention` is a *third* contract
again — `(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens,
num_seqs)`, with K and V as separate page arrays — so it is not a reference for
sglang-jax's own kernel. `rpa_v3` names a shared cache layout and semantics,
not a shared API.

### Native shape

Only `rpa_v2` has one. JAXBench's `CONFIG` defines a Llama-3.1-70B serving
workload:

```text
max_num_batched_tokens=4096  max_num_seqs=64   num_q_heads=64  num_kv_heads=8
head_dim=128                 page_size=16      pages_per_seq=256
```

giving 64 tokens and 4096 KV positions per sequence, a 1.07 GiB paged cache,
and GQA ratio 8. Neither tpu-inference nor sglang-jax ships a benchmark
configuration for `rpa_v3`, so those two are validated for correctness and
**deliberately not given a native-shape performance number** rather than being
profiled at an invented shape.

## 2. Baselines

`rpa_v2` uses `baseline.rpa_v2`, derived from the pure-JAX `workload` that
JAXBench ships as its own baseline for this benchmark — so the denominator is
the one the upstream benchmark itself uses. It contains no Pallas call.

`rpa_v3` has no corpus-written reference. tpu-inference v3 ships a pure-JAX
`ref_ragged_paged_attention` (verified: no `pallas` or `pl.` reference in the
function body) matching its own signature. Re-deriving the packed cache layout
by hand would risk a subtly wrong reference producing false failures, so
sglang-jax's kernel is validated against **tpu-inference's** reference — a
genuine cross-repository check, since the two were written independently.

## 3. Correctness

All PASS at the 0.9999 cosine threshold.

### rpa_v2, against `baseline.rpa_v2`

| Implementation | Shape | Cosine | Max abs | Mean abs | Status |
|---|---|---:|---:|---:|---|
| JAXBench | native | 0.9999962449 | 9.766e-04 | 4.632e-05 | PASS |
| tpu-inference | native | 0.9999962449 | 9.766e-04 | 4.632e-05 | PASS |

The two agree with each other to every reported digit at the native shape, and
also across five small-shape configurations: GQA ratio 4 and 8, MQA
(`num_kv_heads=1`), a partially filled batch (`num_seqs=2` of 4), and uneven
per-sequence KV lengths `[128, 100, 64, 33]`.

### rpa_v3, against tpu-inference v3's pure-JAX reference

| Implementation | Reference | Cosine | Max abs | Mean abs | Status |
|---|---|---:|---:|---:|---|
| tpu-inference v3 | own | 0.9999750257 | 7.813e-03 | 5.621e-04 | PASS |
| sglang-jax v3 | **cross-repo** | 0.9999755621 | 7.813e-03 | 5.568e-04 | PASS |
| sglang-jax v3 | vs tpu-inference v3 kernel | 0.9999848008 | 7.813e-03 | 3.980e-04 | PASS |

One trap worth recording: the v3 kernels default to `update_kv_cache=True` and
**donate the `kv_cache` buffer**. Reusing the same cache array for a second
call raises `Array has been deleted`; inputs must be rebuilt per call.

## 4. Timing and speedup — `rpa_v2` at the native shape

Because both implementations expose the same block-size API, they are profiled
in two regimes. Reporting only one would confuse tuning with kernel quality.

### Matched tuning — both given JAXBench's autotuned blocks

`num_kv_pages_per_block=64, num_queries_per_block=64, vmem_limit=32 MiB`

| Configuration | Median (ms) | Std (ms) | TFLOP/s | MXU @ 918 | Speedup |
|---|---:|---:|---:|---:|---:|
| **pure-JAX baseline** | **17.025451** | 0.003471 | 32.3 | 3.52% | **1.000×** |
| JAXBench | 3.545906 | 0.001888 | 155.0 | 16.89% | **4.801×** |
| tpu-inference | 3.583431 | 0.001980 | 153.4 | 16.71% | **4.751×** |

### Kernel-selected blocks — each kernel consults the JAX tuned-size table

| Configuration | Median (ms) | Std (ms) | TFLOP/s | MXU @ 918 | Speedup |
|---|---:|---:|---:|---:|---:|
| JAXBench | 4.229369 | 0.001482 | 130.0 | 14.16% | 4.026× |
| tpu-inference | 4.246164 | 0.001595 | 129.5 | 14.10% | 4.010× |

**Tuning matters more than the kernel here.** Under matched tuning the two
implementations are 1.06% apart; under kernel-selected blocks, 0.40% apart. But
switching from the table's choice to JAXBench's autotuned blocks is worth
**19.3%** on the same kernel (4.229 → 3.546 ms). Had tpu-inference been
profiled with table-selected blocks against JAXBench's tuned run — the first
comparison this analysis produced — it would have looked 20% slower, and all of
that would have been tuning, not code.

The two kernels differ in exactly two places, both visible in the source diff:
JAXBench uses `pl.cdiv` and a masked `pltpu.store`, while tpu-inference uses a
local `cdiv` and a read-modify-write `jnp.where` select. The masked store is
worth about 1% here and is preserved rather than normalized away.

## 5. FLOP accounting

| Convention | Value | Note |
|---|---:|---|
| `logical` (primary) | 549,755,813,888 | `max_seqs × H_q × 4 × q_len × kv_len × D`, matching JAXBench's `get_flops()` |
| `causal_useful` | 545,527,955,456 | subtracts the right-aligned causal mask |

The mask removes only **0.77%** of the rectangle at this shape: with
`q_len = 64` and `kv_len = 4096`, query row `r` still attends to `4033 + r` of
4096 positions. Unlike the flash-attention family, causal masking is not a
meaningful accounting question here, and no ragged padding is discounted from
either count because every sequence shares one `kv_len`.

## 6. Bottleneck analysis

Per-operation self time from the first trace chunk:

| Configuration | Operation | Occurrences | Time | Share |
|---|---|---:|---:|---:|
| pure-JAX baseline | `gather` | 1 | 5658.525 µs | 33.25% |
| pure-JAX baseline | `sub` | 1 | 3696.698 µs | 21.72% |
| pure-JAX baseline | `dot_general` | 1 | 3005.970 µs | 17.66% |
| pure-JAX baseline | `broadcast_in_dim` | 1 | 2574.451 µs | 15.13% |
| pure-JAX baseline | `reshape` | 1 | 1559.409 µs | 9.16% |
| JAXBench | `pallas_call` | 1 | 3544.280 µs | 100.00% |
| tpu-inference | `pallas_call` | 1 | 3585.121 µs | 100.00% |
| JAXBench, auto blocks | `pallas_call` | 1 | 4226.949 µs | 100.00% |

This is the clearest bottleneck story in the corpus so far:

- **The baseline is not matmul-bound at all.** It spends 17.66% of its device
  time in `dot_general` and the other 82% gathering pages, masking, and
  reshaping. Its 499 distinct device operations are the 64-sequence Python loop
  unrolled, each iteration re-gathering its pages and rebuilding a full
  `[64, 64, 4096]` mask. The 4.8× speedup is therefore mostly *overhead the
  kernel does not pay*, not a faster inner matmul — the kernel folds the page
  gather into its DMA schedule and the mask into the accumulator.
- **Both kernels are a single fused `pallas_call` at 100% of device time.**
  There is no metadata tail to fuse away, unlike the grouped-matmul family
  where 4% of runtime was un-fused `scatter-add`/`gather` work.
- At 16.9% paper-style MXU there is real headroom, but the ceiling is lower
  than it looks: a decode-heavy paged workload with `q_len = 64` against
  `kv_len = 4096` is dominated by streaming 1.07 GiB of KV pages, not by MXU
  throughput. The interesting next measurement is arithmetic intensity against
  the 577.96 FLOP/byte ridge point, not more tile search.

XProf's bandwidth fields are again not used for conclusions: the Pallas calls
report 4.4–14.9 GB/s, which is implausibly low for a kernel streaming a 1 GiB
cache in 3.5 ms (≈300 GB/s actual). Device durations and operation attribution
remain reliable.

Suggested next work:

1. Sweep `num_kv_pages_per_block` around 64 — tuning is worth ~19% here and
   JAXBench's value was autotuned for a different JAX version.
2. Profile a decode-only shape (`q_len = 1`), which is the case the v3 contract
   exists to optimize and where the v2 kernels should look worst.
3. Give `rpa_v3` a defensible native shape, or state permanently that it has
   none upstream and profile it only at declared validation shapes.
4. Migrate the remaining 5 audited RPA launch points: `v3/kernel_hd64.py`,
   `experimental/batched_rpa` (kernel + schedule), `experimental/rpa_v3_cp`,
   and sglang-jax's older `ragged_paged_attention.py`.

## 7. Trace and artifact locations

Raw traces remain on the TPU:

```text
profiles/native/ragged_paged_attention/<run>/trace/
```

where `<run>` is `rpa_<implementation>` for matched tuning and
`rpa_<implementation>__auto-blocks` for kernel-selected blocks. Compact local
artifacts are the five `result.json` files beside this report and the `rpa-*`
keys in `profiles/native/xplane_summary.json`.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

# Matched tuning: both kernels get JAXBench's autotuned blocks.
for impl in rpa-baseline rpa-jaxbench rpa-tpu-inference; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --native --trace-chunk-size 1 \
    --output-dir profiles/native
done

# Kernel-selected blocks, to separate tuning from kernel quality.
for impl in rpa-jaxbench rpa-tpu-inference; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --native --rpa-auto-blocks --run-label auto-blocks \
    --trace-chunk-size 1 --output-dir profiles/native
done

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native

# Correctness, including the rpa_v3 cross-repository check.
uv run --frozen --with pytest python -m pytest \
  tests/test_ragged_paged_attention_tpu.py -q
```
