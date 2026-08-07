# Flash Attention Native-Shape TPU v6e Analysis

Generated: 2026-07-30 (America/New_York)  
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)  
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1  
Protocol: bf16 inputs, JIT compile, 5 warmups, 50 wall-clock diagnostics,
then 50 device-profiled iterations. Every profiled iteration was captured in a
separate trace chunk to avoid Perfetto event-count truncation.

The primary time is the complete `jit_*()` device event from Perfetto, not
`time.perf_counter()`. MXU utilization follows the JAXBench paper convention
of 918 bf16 TFLOP/s. XPlane independently identifies the device peak as
946.7 TFLOP/s.

## 1. Computation flow and native configurations

These are three different upstream contracts, so each Pallas kernel is compared
only with its matching JAX implementation.

| Source | Contract | Original/native shape | Selected tiling |
|---|---|---|---|
| JAXBench | causal batched MHA | `B=4,H=64,S=4096,D=128` | `BQ=2048, BK-major=2048, BK=1024, BB=1` |
| PallasBench | dense educational 2-D attention | effectively `B=1,H=1,S=512,D=64` | `BQ=128`, full `K/V=512` |
| Tokamax | causal Splash MHA, pre-scaled Q | `B=8,H=128,Sq=Skv=4096,Dqk=192,Dv=128` | `BQ=BKV=BKV-compute=1024` |

All implementations compute `QKᵀ`, apply softmax (and a causal mask where
applicable), then compute `P·V`.

### Grid and matmul inventory

| Source | Logical grid | Matmul | Per-tile shape | Full-rectangle FLOPs |
|---|---|---|---|---:|
| JAXBench | `(B/BB,H,S/BQ,S/BK-major) = (4,64,2,2)`; causal cells are skipped in-kernel | QKᵀ | `(2048,128)@(128,1024)`, two inner K tiles per major tile | 1.100T |
| JAXBench | same | PV | `(2048,1024)@(1024,128)`, two inner K tiles per major tile | 1.100T |
| PallasBench | `(S/BQ) = (4)` | QKᵀ | `(128,64)@(64,512)` | 0.0336G |
| PallasBench | same | PV | `(128,512)@(512,64)` | 0.0336G |
| Tokamax | per batch: `(128 heads,10 active causal block pairs)` | QKᵀ | `(1024,192)@(192,1024)` | 6.597T |
| Tokamax | same | PV | `(1024,1024)@(1024,128)` | 4.398T |

Tokamax's dynamic causal grid executes ten lower-triangular 1024×1024 block
pairs instead of all sixteen pairs. Its tile-level executed matmul count is
therefore about 6.872T FLOPs, while its Pallas `CostEstimate` and the upstream
full-rectangle convention report 10.995T. The latter is retained as the primary
paper-compatible utilization numerator.

### FLOP accounting cross-check

| Configuration | Source/manual logical FLOPs | XLA `cost_analysis().flops` | Interpretation |
|---|---:|---:|---|
| JAX causal baseline | 2.199T | 1.121T | XLA exploits/charges roughly the causal triangle |
| JAXBench Pallas | 2.199T | 2.216T | Pallas cost estimate, 0.78% above source `get_flops()` |
| JAX dense baseline | 67.109M | 34.603M | XLA fusion accounting omits or discounts part of the pipeline |
| PallasBench | 67.109M | unavailable | Pallas custom call has no explicit cost estimate |
| JAX blockwise Splash baseline | 10.995T | 43.489G | XLA reports one loop body, not all dynamic loop trips |
| Tokamax Splash | 10.995T | 10.995T | explicit Pallas cost estimate |

The profiler stores all three relevant facts rather than assuming that XLA
cost accounting is uniform across standard JAX, loops, and Pallas custom calls.

## 2. Correctness

| Pallas implementation | Native test shape | Cosine similarity | Max abs diff | Mean abs diff | Status |
|---|---|---:|---:|---:|---|
| JAXBench | `4×64×4096×128` | 0.99997127 | 0.0625 | 0.00029511 | PASS |
| PallasBench | `512×64` | 0.99997836 | 0.00390625 | 0.00034806 | PASS |
| Tokamax | `8×128×4096×(192/128)` | 0.99999422 | 0.015625 | 0.00013090 | PASS |

Reference functions are respectively `causal_bhsd`, `dense_2d`, and the
memory-bounded pure-JAX `splash_mha_bhsd_blockwise`. The pass threshold is
cosine similarity greater than 0.9999.

PallasBench's upstream task assumes fp32. The standalone TPU corpus version
uses explicit fp32 dot accumulators and casts the result back to bf16; this is
required by the JAX 0.10.2 TPU verifier.

## 3. Timing and speedup

### Device timing

| Configuration | Min (ms) | Median (ms) | Mean (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| JAX causal baseline | 15.161027 | 15.290344 | 15.288313 | 0.052829 | 15.191758 | 15.367243 |
| JAXBench Pallas | 6.348314 | 6.352849 | 6.352921 | 0.001665 | 6.350424 | 6.355997 |
| JAX dense baseline | 0.002503 | 0.002536 | 0.002578 | 0.000073 | 0.002506 | 0.002706 |
| PallasBench | 0.004211 | 0.004325 | 0.004323 | 0.000089 | 0.004213 | 0.004479 |
| JAX blockwise Splash baseline | 260.389261 | 260.871027 | 260.985522 | 0.524268 | 260.493629 | 261.919532 |
| Tokamax Splash | 28.389704 | 28.433733 | 28.433202 | 0.011712 | 28.416525 | 28.449062 |

### Matching-baseline comparison

| Pallas implementation | Baseline median | Pallas median | Speedup | Result |
|---|---:|---:|---:|---|
| JAXBench | 15.290344 ms | 6.352849 ms | **2.407×** | faster |
| PallasBench | 2.536 µs | 4.325 µs | **0.586×** | 1.706× slower |
| Tokamax | 260.871027 ms | 28.433733 ms | **9.175×** | faster |

Tokamax's original dense pure-JAX expression could not compile at the native
shape: XLA requested 69.00 GiB of temporaries with only 31.25 GiB available.
The baseline used here remains pure JAX and mathematically equivalent, but
blocks Q by 128 and maps batches sequentially. Tokamax's speedup includes both
better tiling/fusion and the Pallas dynamic grid skipping fully masked causal
blocks; it is not a same-instruction-count microkernel comparison.

For PallasBench, wall medians are about 116 µs for both implementations even
though device medians are 2.5 and 4.3 µs. This demonstrates why wall timing is
invalid for this original, genuinely small workload.

### Tokamax tile preselection

| `BQ/BKV/BKV-compute` | One-iteration device time |
|---|---:|
| `512/512/512` | 40.315 ms |
| `1024/1024/1024` | **28.434 ms** |
| `2048/2048/1024` | 29.037 ms |
| `2048/2048/2048` | 31.840 ms |

These preflight measurements selected 1024³ before the formal 5+50 run.

## 4. Hardware utilization

Primary values use each source's full-rectangle logical FLOPs, matching the
JAXBench paper-style numerator requested for this experiment.

| Configuration | FLOPs used | Device median | Achieved TFLOP/s | MXU @ 918 |
|---|---:|---:|---:|---:|
| JAX causal baseline | 2.199T logical | 15.290344 ms | 143.82 | 15.67% |
| JAXBench Pallas | 2.199T logical | 6.352849 ms | 346.15 | 37.71% |
| JAX dense baseline | 67.109M logical | 0.002536 ms | 26.47 | 2.88% |
| PallasBench | 67.109M logical | 0.004325 ms | 15.51 | 1.69% |
| JAX blockwise Splash baseline | 10.995T logical | 260.871027 ms | 42.15 | 4.59% |
| Tokamax Splash | 10.995T XLA/manual | 28.433733 ms | 386.69 | 42.12% |

Two alternative causal interpretations are useful:

- Using the JAX baseline's 1.121T XLA cost gives 73.31 TFLOP/s and 7.99% MXU.
- Using Tokamax's approximately 6.872T actually scheduled block matmuls gives
  241.68 TFLOP/s and 26.33% MXU. Exact useful triangular-token FLOPs are lower
  still because diagonal blocks compute masked entries.

## 5. Bottleneck analysis

XPlane reports an HBM ridge point of 577.96 FLOP/byte and a VMEM-read ridge
point of 40.64 FLOP/byte.

- **JAXBench:** the Pallas wrapper is essentially one 6.352 ms custom call.
  Its 2.4× gain comes from large 2048-token tiles, online softmax, and avoiding
  the baseline's separate 9.413 ms online-softmax and 5.944 ms QK path.
  At 37.7% paper-style MXU, masking, softmax, data movement, and pipeline
  bubbles still leave substantial headroom.
- **PallasBench:** the original shape launches only four Q programs. At this
  scale, Pallas launch/copy overhead exceeds the benefit of its 128×512 tiles;
  the fused JAX baseline is faster. This is an upstream workload-size property,
  not evidence that wall-clock timing should be substituted.
- **Tokamax:** the Pallas call occupies 22.124 ms (77.93%) of the wrapper.
  Two argument/data events occupy about 3.135 ms and 3.129 ms (22.06%
  combined). The chosen 1024 tile is materially better than 512 and 2048 in
  this configuration. Further gains should target argument/layout movement and
  overlap around the custom call before expanding the tile search.
- **Pure-JAX Splash baseline:** QK, PV, and softmax reduction consume about
  40.82%, 33.24%, and 22.31% respectively. Query blocking solves HBM capacity
  but repeatedly streams K/V and serializes batch/query loops, explaining its
  low 4.59% logical MXU utilization.

XProf's Pallas bandwidth and operational-intensity fields are not used for
conclusions: the Tokamax custom call reports a physically impossible bandwidth
and near-zero OI. Device durations and operation attribution remain useful.

Suggested next work:

1. Expand Tokamax autotuning around asymmetric Q/KV tiles and scheduler flags,
   using the 1024³ result as the incumbent.
2. Add a larger, production-representative PallasBench companion shape while
   preserving 512×64 as the upstream-native result.
3. Add causal-effective and scheduled-block FLOPs as first-class fields in the
   profiler so utilization tables can present multiple conventions
   automatically.

## 6. Trace and per-operation breakdown

Raw traces remain on the TPU at:

```text
profiles/native/flash_attention/<implementation>/trace/
```

Compact local artifacts are the six `result.json` files and
`profiles/native/xplane_summary.json`.

| Configuration | XPlane operation | Device time in first trace chunk | Share |
|---|---|---:|---:|
| JAX causal baseline | online softmax | 9.413 ms | 61.25% |
| JAX causal baseline | QK `dot_general` | 5.944 ms | 38.68% |
| JAXBench | Pallas call | 6.352 ms | 100.00% |
| JAX Splash baseline | QK `dot_general` | 106.624 ms | 40.82% |
| JAX Splash baseline | PV `dot_general` | 86.845 ms | 33.24% |
| JAX Splash baseline | softmax reduction | 58.279 ms | 22.31% |
| Tokamax | Pallas call | 22.124 ms | 77.93% |
| Tokamax | two argument events | 6.264 ms | 22.06% |
| PallasBench | Pallas calls, 10 iterations | 26.156 µs total | 60.71% self time |
| JAX dense baseline | online softmax, 10 iterations | 13.390 µs total | 53.72% self time |

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation baseline-causal --native --trace-chunk-size 1 \
  --output-dir profiles/native
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation jaxbench --native --trace-chunk-size 1 \
  --output-dir profiles/native
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation baseline-dense-2d --native --trace-chunk-size 1 \
  --output-dir profiles/native
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation pallasbench --native --trace-chunk-size 1 \
  --output-dir profiles/native
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation baseline-splash --native --trace-chunk-size 1 \
  --output-dir profiles/native
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation tokamax --native --trace-chunk-size 1 \
  --tokamax-block-q 1024 --tokamax-block-kv 1024 \
  --tokamax-block-kv-compute 1024 --output-dir profiles/native

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native
```
