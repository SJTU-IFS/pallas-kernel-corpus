# Grouped Matmul (Megablox GMM) Native-Shape TPU v6e Analysis

Generated: 2026-07-31 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 inputs, JIT compile, 5 warmups, 50 wall-clock diagnostics, then
50 device-profiled iterations, each captured in its own trace chunk to avoid
Perfetto event-count truncation.

The primary time is the complete `jit_*()` device event from Perfetto. MXU
utilization follows the JAXBench paper convention of 918 bf16 TFLOP/s; XPlane
independently reports the device peak as 946.7 TFLOP/s.

## 1. Scope and contract

All five upstream grouped-matmul lineages are now represented. From the three
v1 lineages only the `gmm` **forward** launch is migrated; their `tgmm` (dW)
launches remain audited-not-migrated. From MaxText and Tokamax both the v2
forward and backward launches are migrated.

| Source | Upstream path | Migrated | Audited in that file |
|---|---|---:|---:|
| JAXBench | `benchmark/11p_Megablox_GMM/optimized.py` | `gmm` | `gmm`, `tgmm` |
| vLLM tpu-inference | `megablox/gmm.py` | `gmm` | `gmm` |
| sglang-jax | `gmm/megablox_gmm_kernel/gmm.py` | `gmm` | `gmm` |
| MaxText | `megablox/pallas_mosaic_tpu_v2_{gmm,tgmm}_kernel.py` | `gmm_v2`, `tgmm_v2` | both |
| Tokamax | `ragged_dot/pallas_mosaic_tpu_v2_{gmm,tgmm}_kernel.py` | `gmm_v2`, `tgmm_v2` | both |

**The v2 kernels do not depend on qwix.** Only the *v1* megablox kernels in
MaxText and Tokamax do, which is why those two repositories were initially
recorded as blocked. Their v2 kernels need no quantization framework and no
`flax`, so they are migrated here with no change to the dependency set. See
§8.

All of them share one contract, `grouped_matmul_2d`:

```text
lhs[m, k], rhs[num_groups, k, n], group_sizes[num_groups] -> out[m, n]
out rows [offsets[i], offsets[i] + group_sizes[i]) = lhs[those rows] @ rhs[i]
offsets = exclusive cumsum(group_sizes);  REQUIRED: sum(group_sizes) == m
```

Native shape from JAXBench's `CONFIG` (Qwen3-235B-A22B MoE):
`m = seq_len × experts_per_tok = 4096 × 8 = 32768`, `k = emb_dim = 4096`,
`n = moe_mlp_dim = 1536`, `num_groups = num_experts = 128`, bf16.
Working set is about 2.1 GiB (lhs 268 MiB, rhs 1.61 GiB, out 201 MiB), so
nothing in this family OOMs on a single v6e.

### Two implementations are the same kernel

sglang-jax's `gmm.py` carries the header "Adapted from tpu-inference". They are
not merely similar: after discarding docstrings, their ASTs differ only in the
spelling of the `jax.jit` decorator, both compile to the identical HLO
(`jit_gmm(16611991136737096903)` in both traces), and their measured medians
differ by 0.02%. They are counted as two launch points because two repositories
ship them, but they are **one implementation**. Their `common.py` and tuned
block-size tables do differ (100 vs 145 entries), which could select different
tiling at other shapes — at this shape both miss the table and fall back to the
same heuristic.

## 2. `jax.lax.ragged_dot` is not a baseline

The obvious pure-JAX denominator would be `jax.lax.ragged_dot`. It is not one.
On TPU it lowers to two Mosaic custom calls:

```text
%ragged-dot-metadata = custom-call(...), custom_call_target="tpu_custom_call"
%ragged-dot-none     = custom-call(...), custom_call_target="tpu_custom_call"
                       frontend_attributes={mosaic_fusion_en...}
```

It is the Megablox kernel shipped inside JAX, and the first call builds the
same group metadata the migrated kernels build. It is therefore reported below
as a fourth *implementation*, never as the JAX baseline.
`tests/test_grouped_matmul_tpu.py` pins this, so a future JAX release that
lowers it to plain HLO fails the test rather than silently becoming a baseline.

The pure-XLA baselines actually used are:

| Baseline | Applies to | Why |
|---|---|---|
| `grouped_matmul_batched_dense` | equal group sizes | With uniform groups the ragged structure is a reshape, so `[G,m/G,k] @ [G,k,n]` is exact and performs exactly the logical FLOPs. No handicap. |
| `grouped_matmul_loop` | any group sizes | Masked accumulation over groups. Correct for arbitrary routing, but executes `num_groups ×` the necessary work. |

Both are plain `dot_general`; neither emits a custom call. They agree with each
other bit-exactly, at both the small validation shape and the native shape.

## 3. Correctness

All comparisons use `sum(group_sizes) == m`. Threshold is cosine > 0.9999.

| Implementation | Routing | Reference | Cosine | Max abs | Mean abs | Status |
|---|---|---|---:|---:|---:|---|
| JAXBench | balanced | `batched_dense` | 1.0000000000 | 1.907e-06 | 7.204e-08 | PASS |
| tpu-inference | balanced | `batched_dense` | 1.0000000000 | 1.431e-06 | 6.168e-08 | PASS |
| sglang-jax | balanced | `batched_dense` | 1.0000000000 | 1.431e-06 | 6.168e-08 | PASS |
| `ragged_dot` | balanced | `batched_dense` | 1.0000000000 | 0.000e+00 | 0.000e+00 | PASS |
| JAXBench | unbalanced | `loop` | 1.0000000000 | 1.431e-06 | 7.204e-08 | PASS |
| tpu-inference | unbalanced | `loop` | 0.9999999404 | 1.431e-06 | 6.170e-08 | PASS |
| sglang-jax | unbalanced | `loop` | 0.9999999404 | 1.431e-06 | 6.170e-08 | PASS |
| `ragged_dot` | unbalanced | `loop` | 1.0000000000 | 0.000e+00 | 0.000e+00 | PASS |

`ragged_dot` matching the pure-XLA baselines bit-exactly, while the migrated
Pallas kernels differ by ~1e-6, is consistent with the migrated kernels using
bf16 MXU passes with fp32 accumulation where XLA's path rounds differently.

### Rows covered by no group are not comparable

If `sum(group_sizes) < m`, the trailing rows are outside the contract and the
lineages diverge. Measured with `m=1024, G=8, group_sizes=[128]*7+[0]`:

- **JAXBench leaves them uninitialized.** The kernel never writes them, so they
  hold whatever was in the output buffer: exact zeros on the first call in a
  fresh process, stale values afterwards. Running the same jitted call before
  and after other same-shaped work flips the result, which is the proof that
  this is not a second convention but an absence of one.
- tpu-inference, sglang-jax, and `ragged_dot` write a deterministic partial
  product from the last m-tile, bit-identical to each other and stable across
  cold and warm runs.

Comparing across the two lineages there produced cosine 0.60 — a spurious FAIL
against code that is entirely correct on its actual domain. This is why the
contract requires full coverage, and why the profiler only measures covered
inputs. The native MoE workload always covers every row: each routed token
belongs to exactly one expert.

## 4. Timing and speedup

### Balanced routing — the native JAXBench configuration

Every expert receives exactly `32768 / 128 = 256` tokens.

| Implementation | Median (ms) | Std (ms) | Tiling `(tm,tk,tn)` | TFLOP/s | MXU @ 918 | Speedup |
|---|---:|---:|---|---:|---:|---:|
| **`batched_dense` baseline (pure XLA)** | **1.355097** | 0.000376 | — | 304.3 | 33.15% | **1.000×** |
| **MaxText v2** | **1.360626** | 0.000375 | auto | 303.0 | 33.01% | **0.996×** |
| **Tokamax v2** | **1.360667** | 0.000357 | auto | 303.0 | 33.01% | **0.996×** |
| `jax.lax.ragged_dot` (Mosaic, in JAX) | 1.536961 | 0.000463 | — | 268.3 | 29.22% | 0.882× |
| sglang-jax v1 | 1.604829 | 0.002068 | `(512, 2048, 1536)` | 256.9 | 27.99% | 0.844× |
| tpu-inference v1 | 1.605172 | 0.001964 | `(512, 2048, 1536)` | 256.9 | 27.98% | 0.844× |
| JAXBench v1 | 1.870049 | 0.002426 | `(256, 1024, 1024)` | 220.5 | 24.02% | 0.725× |

The v2 kernels essentially **close the gap to plain XLA** (0.996×), where the
best v1 kernel reached only 0.844×.

**Every Pallas grouped matmul is slower than the pure-XLA baseline here**, by
12% to 38%. This is a real result, not a tuning failure: when routing is
perfectly balanced the ragged structure carries no information, and the ragged
kernels pay for group metadata, a dynamic grid, and store masking that a plain
batched `dot_general` does not need.

The native JAXBench configuration hard-codes `group_sizes = full(G, m // G)`,
which is the dense baseline's best case and not what production MoE routing
looks like. Reporting only this table would misrepresent the family.

### Unbalanced routing — where the ragged kernels matter

Random group sizes still summing to `m`. `batched_dense` is not applicable, so
the only pure-XLA option is the masked loop.

| Implementation | Median (ms) | Std (ms) | TFLOP/s | MXU @ 918 | Speedup |
|---|---:|---:|---:|---:|---:|
| **`loop` baseline (pure XLA)** | **129.730081** | 0.050753 | 3.2 | 0.35% | **1.000×** |
| **MaxText v2** | **1.568957** | 0.001769 | 262.8 | 28.62% | **82.69×** |
| **Tokamax v2** | **1.570100** | 0.002106 | 262.6 | 28.61% | **82.63×** |
| `jax.lax.ragged_dot` | 2.000266 | 0.001784 | 206.1 | 22.45% | 64.86× |
| sglang-jax v1 | 2.279739 | 0.002427 | 180.9 | 19.70% | 56.91× |
| tpu-inference v1 | 2.279998 | 0.002504 | 180.8 | 19.70% | 56.90× |
| JAXBench v1 | 3.599451 | 0.003308 | 114.5 | 12.48% | 36.04× |

**This is the regime where a Pallas kernel finally wins outright.** The v2
kernels are 28% faster than `jax.lax.ragged_dot` and 45% faster than the best
v1 kernel. Uneven routing is also the realistic case: production MoE routing is
not balanced, and it is the only regime where the dense reshape baseline does
not apply at all.

The 36–65× figures must be read carefully. The loop baseline is not slow
because XLA is inefficient — on the work it actually issues it reaches
**406.8 TFLOP/s**, above every kernel in this report. It is slow because it
issues 128× too much work: 52.777 TFLOP against a logical 0.412 TFLOP. The
speedup is therefore almost entirely *algorithmic* — skipping work — and not a
same-instruction-count kernel comparison.

Each kernel is also slower under unbalanced routing than balanced routing
(2.280 vs 1.605 ms for tpu-inference; 3.599 vs 1.870 for JAXBench), because
group boundaries no longer align to the m-tile and more tiles must be visited.

### Honest summary of the two regimes

| Question | Answer |
|---|---|
| Fastest at balanced native routing | pure-XLA `batched_dense`, 1.355 ms |
| Fastest ragged implementation, either regime | `jax.lax.ragged_dot` |
| Best migrated kernel | tpu-inference / sglang-jax (one kernel, two repos) |
| Do the Pallas kernels beat plain XLA? | Only when routing is uneven, where the dense reshape does not apply |

## 5. FLOP accounting

Every row is covered exactly once, so the logical count is a dense matmul of
the same shape: `2 × m × k × n = 2 × 32768 × 4096 × 1536 = 412,316,860,416`.

| Configuration | Manual logical | XLA `cost_analysis().flops` | Interpretation |
|---|---:|---:|---|
| `batched_dense` baseline | 412.317G | 412.317G | exact agreement |
| `loop` baseline | 412.317G logical | 412.468G | XLA reports one loop body, not all 128 trips; executed work is 52.777T |
| JAXBench Pallas | 412.317G | 412.317G | explicit Pallas cost estimate |
| tpu-inference / sglang-jax | 412.317G | 412.317G | explicit Pallas cost estimate |

Two conventions are recorded per run in `result.json` under `flops`: `logical`
for every case, plus `executed_dense` for the loop baseline (`logical ×
num_groups`). Tile padding means the kernels can also *schedule* slightly more
than the logical count when group sizes are not multiples of `tm`; that is
visible in the unbalanced timings but is not separately counted here.

## 6. Bottleneck analysis

XPlane reports an HBM ridge point of 577.96 FLOP/byte for this device.
Per-operation self time from the first trace chunk, balanced routing:

| Configuration | Operation | Time | Share |
|---|---|---:|---:|
| `batched_dense` baseline | `dot_general` | 1355.325 µs | 100.00% |
| `ragged_dot` | `ragged-dot-none` | 1530.475 µs | 99.57% |
| `ragged_dot` | `ragged-dot-metadata` | 6.562 µs | 0.43% |
| tpu-inference | `pallas_call` | 1532.715 µs | 96.37% |
| tpu-inference | `scatter-add` | 33.131 µs | 2.08% |
| tpu-inference | `gather` ×7 + `while` + `reduce_sum` | 31.9 µs | 1.55% |
| JAXBench | `pallas_call` | 1812.415 µs | 97.86% |
| JAXBench | metadata ops | 48.3 µs | 2.60% |

This explains the ranking precisely:

- **tpu-inference's Pallas kernel is not the problem.** At 1532.7 µs it is
  within 0.15% of `ragged_dot`'s 1530.5 µs kernel — they are the same Megablox
  code. The entire 0.844× vs 0.882× gap is **group-metadata overhead**: 65.0 µs
  of JAX-level `scatter-add`, `gather`, `searchsorted`, and `while` ops
  (4.07% of total) versus `ragged_dot`'s single fused 6.6 µs
  `ragged-dot-metadata` custom call. Fusing metadata construction is the single
  highest-value optimization for the migrated kernels.
- **JAXBench is tiling-limited.** Its kernel takes 1812.4 µs against 1532.7 µs
  for the same algorithm, because its autotuned `(256,1024,1024)` is smaller
  than the `(512,2048,1536)` the tpu-inference heuristic picks. Its tuning was
  performed for a different target; re-tuning at this shape should close most
  of the 15% gap.
- **The dense baseline wins on balanced input** by having no metadata phase at
  all and a single MXU-saturating `dot_general`.
- Neither tuned table contains this shape's key
  `(32768, 4096, 1536, 128, 128, 'bfloat16', 'bfloat16', 4096)`, so both
  tpu-inference and sglang-jax fall through to `get_default_gmm_block_sizes`.
  The tables target fp8-quantized decode shapes.

XProf's bandwidth fields are not used for conclusions: the Pallas calls report
5.49e9 GB/s and the JAXBench call 7.41e10 GB/s, which are physically
impossible. Device durations and operation attribution remain reliable.

Suggested next work:

1. Fuse or precompute group metadata for the migrated kernels; it is 4.07% of
   tpu-inference's total and the entire gap to `ragged_dot`.
2. Re-tune JAXBench's `(tm,tk,tn)` at this shape; `(512,2048,1536)` is already
   known to be better on this device.
3. Add the native tuned-table entry for this shape rather than relying on the
   heuristic fallback.
4. Migrate the **v1** `tgmm` from JAXBench, so the training-side dW pass is
   covered for that lineage too; `tgmm_v2` is already migrated from MaxText and
   Tokamax (§8).
5. Profile `tgmm_v2`, which is migrated and correctness-tested but not yet
   benchmarked; it needs an explicit `preferred_element_type` (it asserts
   `out_dtype cannot be None`).
6. Resolve the `qwix` question for the remaining **v1** kernels of MaxText and
   Tokamax. This no longer blocks repository coverage -- their v2 kernels are
   migrated, so the family already spans 5 of 6 repositories -- and the v2
   measurements below suggest v1 is the less interesting target anyway.

## 8. The v2 kernels, and what unblocked them

MaxText and Tokamax were initially recorded as blocked on `qwix`, a third-party
quantization package. That was true only of their **v1** kernels. Their v2
kernels contain zero `qwix` references, so they need no quantization framework
and no `flax`, and were migrated with no change to the pinned dependency set.

Two mechanical edits were required, both recorded in each file's header:

1. `tgmm_v2` imports `gmm_v2` as a sibling module, and defines four helpers
   that shadow same-named `gmm_v2` helpers (`get_cost_estimate`,
   `get_scope_name`, `zero_out_start`, `zero_out_end`) with genuinely different
   bodies. Flattening the pair renames the tgmm copies with a `tgmm_` prefix.
2. Tokamax builds its mesh with `pltpu.TensorCoreMesh`, which jax 0.10.2 no
   longer exposes publicly; it is spelled `pltpu.create_tensorcore_mesh`, the
   public factory for the same object. This v6e has one TensorCore, so the
   MegaCore scaling that mesh exists for is inactive here.

### A silent precision regression from v1 to v2

`gmm` (v1) defaults `preferred_element_type` to `jnp.float32`. `gmm_v2`
defaults it to the **input dtype**. Left at the default, accumulation rounds to
bf16 per k-tile, and at `k=4096` the result falls below the corpus 0.9999
threshold:

| Setting | Cosine vs `batched_dense` | Max abs |
|---|---:|---:|
| v2 default (bf16 out) | 0.9998437762 | 0.015625 |
| `preferred_element_type=jnp.float32` | 1.0000001192 | **0.0** |

Both MaxText and Tokamax show identical numbers, confirming the shared lineage.
A drop-in v1 → v2 swap is therefore a silent accuracy regression unless the
caller pins the dtype. Everything reported above pins it to float32 so v2 is
measured on the same contract as v1, and
`tests/test_grouped_matmul_tpu.py::test_v2_default_output_dtype_loses_precision`
fails if that default ever changes.

### What remains blocked

Only the four **v1** launch points in MaxText and Tokamax still need `qwix`.
Their closure for the surface used (`QArray`, `dot_general`, `dot`,
`pallas_call`) is roughly 2,340 lines across seven `qwix/_src` modules, and
`QArray` is a `flax.struct` dataclass, so a faithful copy also pulls in `flax`.
`qwix.pallas.pallas_call` has no no-QArray fast path, so a thin shim would be a
behaviour change rather than a copy. Given that v2 is both newer and faster
than v1 in every regime measured here, migrating the v1 kernels looks like low
value for that cost.

## 7. Trace and artifact locations

Raw traces remain on the TPU:

```text
profiles/native/grouped_matmul/<run>/trace/
```

where `<run>` is `gmm_<implementation>` for balanced routing and
`gmm_<implementation>__unbalanced` for uneven routing. Compact local artifacts
are the ten `result.json` files beside this report and the
`grouped_matmul*` keys in `profiles/native/xplane_summary.json`.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

# Balanced routing, the native JAXBench configuration.
for impl in gmm-baseline-batched-dense gmm-jaxbench gmm-tpu-inference \
            gmm-sglang-jax gmm-xla-ragged-dot; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --native --trace-chunk-size 1 \
    --output-dir profiles/native
done

# Uneven routing, where the dense baseline does not apply.
for impl in gmm-baseline-loop gmm-jaxbench gmm-tpu-inference \
            gmm-sglang-jax gmm-xla-ragged-dot; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --native --unbalanced --run-label unbalanced \
    --trace-chunk-size 1 --output-dir profiles/native
done

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native
```
