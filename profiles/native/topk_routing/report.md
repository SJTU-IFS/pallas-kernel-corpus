# MoE Router Top-K TPU v6e Analysis

Generated: 2026-07-31 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: f32 router logits, JIT compile, 5 warmups, 50 wall-clock diagnostics,
then 50 device-profiled iterations, each in its own trace chunk.

**These are selection kernels.** They perform no meaningful arithmetic, so MXU
utilization is reported as `null` and correctness is **exact**, not a cosine
threshold — a nearly-right expert index is a wrong index.

## 1. Scope

| Source | Upstream path | Migrated | Audited |
|---|---|---:|---:|
| sglang-jax | `biased_topk/` | 2 launches | 2 |
| sglang-jax | `grouped_topk/` | 1 launch | 1 |
| Tokamax | `experimental/tpu/topk/` | **0** | 1 |
| vLLM tpu-inference | `experimental/deepseek_v4/streamindex_topk.py` | 0 | 1 |
| sglang-jax | `dsa/streamindex_topk.py` | 0 | 1 |

**3 of the family's 6 audited launch points are migrated.** The other three are
covered in §6.

The two sglang packages are kept as two corpus files rather than merged: they
define three colliding names (`get_interpret`, `NEG_INF`, `SAFE_AUTO_BT`) and
concatenating them would silently shadow the first definitions.

Three contracts, all returning batch-major `(weights[batch, topk], ids[batch, topk])`:

```text
router_topk         f(router_logits[batch, num_experts], topk)
router_biased_topk  + correction_bias[num_experts]; ranks on logits + bias but
                      returns the PRE-bias weights of the winners
router_grouped_topk + num_expert_group, topk_group; experts split into groups,
                      each scored by its two best biased logits, best
                      topk_group groups kept, topk chosen within them
```

Inside the kernels the Pallas grid works transposed — the batch dimension is the
lane dimension — and the wrappers transpose back, which is why their internal
variables are named `weights_t`/`ids_t`. **`num_experts` must be a multiple of
128**; the kernels reject anything else.

### No native shape

No upstream benchmark defines one. The profiled shape is a **declared
validation shape** (`native_source_shape: false`): `batch=4096`,
`num_experts=256`, `topk=8`, `num_expert_group=8`, `topk_group=4` — a
DeepSeek-V3-style router.

## 2. Correctness — exact, not approximate

| Contract | Expert ids | Selected weights | Status |
|---|---|---|---|
| `router_topk` | bitwise identical | max abs 0.0 | PASS |
| `router_biased_topk` | bitwise identical | max abs 0.0 | PASS |
| `router_grouped_topk` | bitwise identical | max abs 0.0 | PASS |

Verified at `(batch, num_experts, topk)` of `(4096,256,8)`, `(1024,128,4)`,
`(2048,384,6)`, and for grouped at two group configurations. Every profiled run
reports `exact_match_failures: 0`.

A dedicated test also pins the bias semantics: with a bias large enough to
change the winner, the kernel must return the winner's *pre-bias* logit. That
asymmetry is the point — the bias steers routing without contaminating the
combine weights — and a kernel that returned `logit + bias` would still pass a
naive cosine check.

## 3. Timing

| Contract | Configuration | Median (ms) | Std (ms) | Speedup |
|---|---|---:|---:|---:|
| plain | pure-JAX baseline | 0.115054 | 0.000106 | 1.000× |
| plain | sglang-jax | **0.016457** | 0.000049 | **6.99×** |
| biased | pure-JAX baseline | 0.402626 | 0.000150 | 1.000× |
| biased | sglang-jax | **0.017599** | 0.000056 | **22.88×** |
| grouped | pure-JAX baseline | 2.478722 | 0.002392 | 1.000× |
| grouped | sglang-jax | **0.022181** | 0.000056 | **111.75×** |

The kernel is nearly shape-insensitive across the three contracts — 16.5, 17.6,
22.2 µs — while the pure-JAX baseline degrades by 21× from plain to grouped.
That gap is the whole story, and the trace explains it.

## 4. Bottleneck analysis

Per-operation self time from the first trace chunk:

| Configuration | Operation | Occurrences | Time | Share |
|---|---|---:|---:|---:|
| baseline, grouped | `top_k` | 1 | 1996.014 µs | 80.52% |
| baseline, grouped | `gather` | 1 | 278.575 µs | 11.24% |
| baseline, grouped | `scatter_custom_fusion` | 1 | 88.635 µs | 3.58% |
| baseline, biased | `gather` | 1 | 278.591 µs | 69.24% |
| baseline, biased | `top_k` | 1 | 107.730 µs | 26.77% |
| baseline, biased | `add` | 1 | 4.037 µs | 1.00% |
| **sglang-jax, biased** | `pallas_call` | 1 | **17.285 µs** | 100.00% |
| **sglang-jax, grouped** | `pallas_call` | 1 | **21.891 µs** | 100.00% |

- **Each kernel is a single fused `pallas_call` at 100% of device time.** No
  metadata tail, no separate gather — the whole routing decision is one launch.
- **The baseline's cost is XLA's `top_k` plus the gather it forces.** In the
  biased case, `jnp.take_along_axis` to recover pre-bias weights costs 278.6 µs
  — 69% of the total, and *more than the `top_k` itself*. The kernel gets those
  weights for free because it already has them in registers when it picks.
- **Grouped is where the baseline collapses.** It needs 33 distinct device
  operations: a reshape, `top_k` over groups, a scatter to build the group mask,
  a `repeat` to expand it, a masked `where`, then a second `top_k`. Two `top_k`
  passes plus a scatter is 2.48 ms against the kernel's single 21.9 µs pass.
- The 111.75× is therefore **structural, not micro-optimization**: the kernel
  fuses a two-level selection that XLA cannot fuse across.

Bandwidth is reported (`achieved_gbytes_per_second`) over the logits the kernel
must read and the selection it writes — 4.46 MB at this shape, giving 200–271
GB/s for the kernels. Note the baseline rows use the same *algorithmic* byte
count while actually moving far more through intermediates, so their GB/s
figures understate their true traffic and should not be compared directly.

Suggested next work:

1. Sweep `block_tokens` (fixed at `"auto"` here).
2. Profile bf16 logits with `packed=True`, the grouped kernel's bit-exact
   packed-key path, which this report does not cover.
3. Resolve the Tokamax SparseCore kernel (§6) — it would be the corpus's first
   non-TensorCore kernel.

## 5. Trace and artifact locations

```text
profiles/native/topk_routing/<run>/trace/
```

Compact local artifacts are the six `result.json` files beside this report and
the `topk-*` keys in `profiles/native/xplane_summary.json`.

## 6. The three unmigrated launch points

### Tokamax SparseCore top-k — runs, but semantics unresolved

`tokamax/_src/ops/experimental/tpu/topk/pallas_mosaic_tpu_kernel.py` flattens
cleanly (its only non-corpus dependency is `absl` logging) and **runs on this
v6e** — `plsc.get_sparse_core_info()` reports 2 cores, 16 subcores, 8 lanes. It
would be the corpus's only SparseCore kernel.

It does not reproduce `jax.lax.top_k` under the obvious calling convention,
measured at rows=8, n=1024, k=8 with `use_approx_top_k=False`:

- with all-negative keys it returns the **most** negative elements — consistent
  with ranking by raw float bit pattern, where the sign bit makes negatives
  sort high;
- with non-negative keys its selected *set* still differs (it returned 99.6164
  where the reference took 99.778) and its output is not sorted.

This may be a misuse of `num_seq_windows` / `digit_width` / the expected input
layout rather than a kernel defect — the API has tuning knobs this corpus has
not explored. Until that is resolved, counting it as migrated and validated
would be wrong, so **it is not in the corpus**. The evidence is preserved in
the family's `baseline.py`.

### streamindex_topk ×2 — different family in practice

`tpu-inference/experimental/deepseek_v4/streamindex_topk.py` and
`sglang-jax/dsa/streamindex_topk.py` share a signature
(`q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
distribution`) and are 7-of-10 AST-identical, so they are the same lineage with
three diverged functions. But that signature is a *sparse-attention indexer*
over a paged KV cache, not MoE routing — it is closer to the
ragged-paged-attention family in input construction and cost. Migrating it is a
separate effort of RPA scale and is not attempted here.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in topk-baseline-plain topk-sglang-plain \
            topk-baseline-biased topk-sglang-biased \
            topk-baseline-grouped topk-sglang-grouped; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --trace-chunk-size 1 --output-dir profiles/native
done

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native

uv run --frozen --with pytest python -m pytest \
  tests/test_topk_routing_tpu.py -q
```
