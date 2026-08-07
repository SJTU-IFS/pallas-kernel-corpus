# Flash Attention TPU v6e Analysis Report

Generated: 2026-07-31  
Device: single-chip TPU v6e (`TPU v6 Lite`)  
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1  
Inputs: bf16, sequence 1024, head dimension 128  
Protocol: JIT compile, 5 warmups, 50 wall-clock diagnostics, and 50
device-profiled iterations. Profiler iterations were captured as five groups of
10 because a single JAXBench Pallas trace exceeded Perfetto's one-million-event
limit when low-level Pallas region tracing was enabled.

The primary time below is the complete `jit_*()` device wrapper duration from
the Perfetto trace. Wall-clock time is not used for speedup or utilization.

## 1. Computation flow

The directory contains three distinct contracts:

| Implementation | Contract | Input |
|---|---|---|
| JAXBench | causal batched MHA | `[B,H,S,D] = [1,8,1024,128]` |
| Tokamax Splash | causal MHA with pre-scaled Q | `[H,S,D] = [8,1024,128]` |
| PallasBench | non-causal educational attention | `[S,D] = [1024,128]` |

Each computes a QKᵀ matmul, softmax, and probability-value matmul. The
JAXBench and Tokamax results are not compared to each other because their
public scaling contracts differ. PallasBench is not compared to either because
it is 2-D and non-causal.

### Matmul inventory

| Implementation | Grid / tiling | Operation | Tile shape | Calls | Logical FLOPs |
|---|---|---|---|---:|---:|
| JAXBench | `1×8×8×8`, `BQ=BK=128` | QKᵀ | `(128,128)@(128,128)` | up to 512 | 2.147G |
| JAXBench | same | PV | `(128,128)@(128,128)` | up to 512 | 2.147G |
| Tokamax | 128-token Q/KV blocks, 8 heads | QKᵀ | `(128,128)@(128,128)` | causal block grid | 2.147G |
| Tokamax | same | PV | `(128,128)@(128,128)` | causal block grid | 2.147G |
| PallasBench | 8 Q blocks; full K/V per block | QKᵀ | `(128,128)@(128,1024)` | 8 | 0.268G |
| PallasBench | same | PV | `(128,1024)@(1024,128)` | 8 | 0.268G |

Logical FLOPs use the JAXBench paper convention:

```text
causal BHSD / HSD: 4·B·H·S²·D = 4,294,967,296 FLOPs
dense 2-D:         4·S²·D     =   536,870,912 FLOPs
```

This convention counts the full attention rectangle. Causal implementations
can skip upper-triangular tiles, so it is an effective/logical FLOP count rather
than an instruction count.

### FLOP cross-check

| Implementation | Logical/manual | XLA `cost_analysis().flops` |
|---|---:|---:|
| JAX causal baseline | 4.295G | 2.190G |
| JAXBench Pallas | 4.295G | 4.329G |
| JAX dense-2D baseline | 0.537G | 0.273G |
| PallasBench | 0.537G | unavailable |
| JAX Splash baseline | 4.295G | 4.364G |
| Tokamax Splash | 4.295G | 4.295G |

XLA's accounting convention is not consistent across standard-JAX fusions and
Pallas custom calls. Hardware utilization therefore reports both the
paper-style logical FLOPs and, when available, XLA-cost FLOPs.

## 2. Correctness

| Implementation | Reference contract | Cosine similarity | Max abs diff | Mean abs diff | Status |
|---|---|---:|---:|---:|---|
| JAXBench | `causal_bhsd` | 0.99997723 | 0.015625 | 0.00041633 | PASS |
| PallasBench | `dense_2d` | 0.99998063 | 0.00292969 | 0.00026487 | PASS |
| Tokamax | `splash_mha_hsd` | 0.99998778 | 0.0078125 | 0.00024295 | PASS |

Thresholds: cosine > 0.9999 is PASS, > 0.99 is MARGINAL, otherwise
FAIL.

PallasBench originally assumed fp32 inputs. Its standalone corpus version now
uses explicit fp32 accumulators for both bf16 matmuls and casts the final result
to the output dtype; without this, JAX 0.10.2's TPU verifier rejects the kernel.

## 3. Efficiency

### Device timing

| Configuration | Min (ms) | Median (ms) | Mean (ms) | Std (ms) | P5 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| JAX causal baseline | 0.023215 | 0.023274 | 0.023308 | 0.000073 | 0.023232 | 0.023453 |
| JAXBench Pallas | 0.207301 | 0.208006 | 0.208201 | 0.000677 | 0.207392 | 0.209611 |
| JAX dense-2D baseline | 0.004082 | 0.004188 | 0.004219 | 0.000065 | 0.004173 | 0.004356 |
| PallasBench | 0.007659 | 0.007824 | 0.007835 | 0.000118 | 0.007665 | 0.008017 |
| JAX Splash baseline | 0.028850 | 0.029016 | 0.028999 | 0.000072 | 0.028876 | 0.029092 |
| Tokamax Splash | 0.173841 | 0.174474 | 0.174645 | 0.000656 | 0.173987 | 0.176270 |

### Speedup against the matching baseline

| Optimized implementation | Baseline median | Kernel median | Speedup | Kernel slowdown |
|---|---:|---:|---:|---:|
| JAXBench Pallas | 0.023274 ms | 0.208006 ms | 0.112× | 8.94× |
| PallasBench | 0.004188 ms | 0.007824 ms | 0.535× | 1.87× |
| Tokamax Splash | 0.029016 ms | 0.174474 ms | 0.166× | 6.01× |

These measurements characterize the shared correctness shape, not each
upstream implementation's published tuned shape. In particular, JAXBench's
upstream tuned parameters target `B=4,H=64,S=4096`; the corpus's small-shape
entry point uses 128-token tiles.

### Why wall-clock timing is misleading

| Configuration | Device median | Wall median | Wall/device |
|---|---:|---:|---:|
| JAX causal baseline | 0.023274 ms | 0.138570 ms | 5.95× |
| JAXBench Pallas | 0.208006 ms | 0.335100 ms | 1.61× |
| JAX dense-2D baseline | 0.004188 ms | 0.133915 ms | 31.98× |
| PallasBench | 0.007824 ms | 0.121775 ms | 15.56× |
| JAX Splash baseline | 0.029016 ms | 0.139410 ms | 4.80× |
| Tokamax Splash | 0.174474 ms | 0.298800 ms | 1.71× |

The 4.2-µs dense baseline would appear to take 134 µs under wall timing. This
is exactly the host-dispatch/synchronization distortion described by JAXBench.

## 4. Hardware utilization

The primary MXU percentage below follows the requested JAXBench paper constant
of 918 bf16 TFLOP/s. XPlane's roofline metadata reports a 946.7-TFLOP/s device
peak; using that value would reduce the percentages by about 3%.

| Configuration | FLOPs used | Device median | Achieved TFLOP/s | MXU @ 918 |
|---|---:|---:|---:|---:|
| JAX causal baseline | XLA 2.190G | 0.023274 ms | 94.12 | 10.25% |
| JAXBench Pallas | XLA 4.329G | 0.208006 ms | 20.81 | 2.27% |
| JAX dense-2D baseline | XLA 0.273G | 0.004188 ms | 65.11 | 7.09% |
| PallasBench | manual 0.537G | 0.007824 ms | 68.62 | 7.47% |
| JAX Splash baseline | XLA 4.364G | 0.029016 ms | 150.41 | 16.38% |
| Tokamax Splash | XLA 4.295G | 0.174474 ms | 24.62 | 2.68% |

For an apples-to-apples algorithmic count, full logical FLOPs give 20.10% for
the causal JAX baseline and 13.97% for the dense baseline; the Pallas values are
unchanged or nearly unchanged.

## 5. Bottleneck analysis

XPlane reports an HBM ridge point of approximately 578 FLOP/byte and a VMEM
read ridge point of 40.6 FLOP/byte.

- A 128×128 QK tile has about 32 FLOP/byte when fp32 score storage is counted.
- A 128×128 PV tile has roughly 25.6–32 FLOP/byte depending on accumulator
  lifetime.
- PallasBench's large 128×1024 matmuls are around 39–41 FLOP/byte at the tile
  level. Re-reading full K and V for all eight Q tiles gives about 114 logical
  FLOP/byte at HBM, still far below the HBM ridge.
- Even an ideal single-read full-attention calculation is approximately
  512 logical FLOP/byte for JAXBench/Tokamax, below the HBM ridge. Repeated KV
  tile traffic reduces it further.

Therefore these configurations are primarily memory/tile-overhead bound, not
MXU-peak-bound. The gap between device time and the compute-peak lower bound is
97.8% for JAXBench, 92.5% for PallasBench, and 97.3% for Tokamax; that gap
includes HBM/VMEM movement, softmax, mask work, pipeline bubbles, and launch
overhead, so it must not be labeled pure non-matmul time.

Suggested next steps:

1. Benchmark the JAXBench implementation at its actual tuned
   `B=4,H=64,S=4096` configuration before judging the upstream optimization.
2. Sweep `block_q`, `block_k_major`, and `block_k` instead of using the generic
   128-token safe entry point.
3. For PallasBench, avoid loading the complete K/V matrices independently for
   every Q program where the compiler cannot retain/reuse them.
4. For Tokamax, tune the Splash block configuration for S=1024; its defaults
   are generic and the XPlane custom call accounts for essentially the entire
   174-µs wrapper.
5. Add shape-specific benchmark records to the corpus inventory so correctness
   shapes and upstream production/tuned shapes are never conflated.

XProf's duration and op naming agree with the Perfetto wrapper events. Its
custom-call bandwidth/OI counters are not reliable for all Pallas calls:
Tokamax, for example, reports a physically impossible bandwidth and near-zero
OI. Those counters are retained in `xplane_summary.json` for auditability but
are not used for the conclusions above.

## 6. Trace and per-op breakdown

Raw traces remain on the TPU under:

```text
profiles/flash_attention/<implementation>/trace/
```

Each optimized run has five chunks of 10 iterations. Compact `result.json`
files and `xplane_summary.json` are stored locally in this profile directory.

| Configuration | XPlane operation | Avg device time |
|---|---|---:|
| JAX causal baseline | online softmax | 13.377 µs |
| JAX causal baseline | QK/mask fusion | 9.501 µs |
| JAX dense-2D baseline | online softmax | 2.313 µs |
| JAX dense-2D baseline | matmul fusion | 1.812 µs |
| JAX Splash baseline | output matmul fusion | 13.011 µs |
| JAX Splash baseline | QK/mask fusion | 9.284 µs |
| JAX Splash baseline | softmax reduction fusion | 6.264 µs |
| JAXBench Pallas | `flash_attention.1` custom call | 207.817 µs |
| PallasBench | `pallas_flash_attention.1` custom call | 7.825 µs |
| Tokamax | `splash_mha_fwd_no_residuals.1` custom call | 174.302 µs |
| Tokamax | mask/grid iota | 0.041 µs |

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation jaxbench
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation pallasbench
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation tokamax

uv run --frozen --group profile python tools/summarize_xplane.py
```
