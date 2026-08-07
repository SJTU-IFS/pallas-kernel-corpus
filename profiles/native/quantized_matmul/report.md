# Quantized Matmul TPU v6e Analysis

Generated: 2026-07-31 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 activations, int8 weights, JIT compile, 5 warmups, 50 wall-clock
diagnostics, then 50 device-profiled iterations, each in its own trace chunk.

## 1. Scope and contracts

| Source | Upstream path | Migrated | Audited there |
|---|---|---:|---:|
| vLLM tpu-inference | `quantized_matmul/{kernel,blockwise_kernel}.py` | 2 launches | 2 |
| sglang-jax | `quantized_matmul/quantized_matmul_kernels/{kernel,blockwise_kernel}.py` | 2 launches | 2 |

**The whole family is migrated** — 4 of 4 audited TPU launch points.

Each repository ships two kernels that share one `util.py` and one
`tuned_block_sizes.py`, so each is flattened into a single corpus file rather
than duplicating the helpers. Both kernel modules define a public
`quantized_matmul_kernel`; concatenating them would silently shadow the first,
so the block-wise copy is exposed as `blockwise_quantized_matmul_kernel`. That
rename is the only edit to either kernel body.

Two contracts:

```text
quantized_matmul_per_channel
  x       [n_batch, n_in]   w_q [n_out, n_in]   w_scale [n_out]
  ->      (x_q @ w_q.T) * w_scale * x_scale

quantized_matmul_blockwise
  block_size is a scalar over n_in;  w_scale [n_in // block_size, 1, n_out]
  -> the contraction is split per block and each partial product scaled
```

Activation quantization, when requested, is symmetric and per token
(`x_scale = rowwise max|x| / dtype_max`). **Weight zero points are not
implemented by either upstream kernel** — both raise `NotImplementedError` — so
the corpus reference is symmetric-only too, and a test pins that.

The inner `matmul_kernel` is **AST-identical** between the two repositories;
only the public wrappers and the tuned tables differ. Their measured medians
agree to within 0.1%.

### No native shape

Neither repository ships a benchmark configuration. The profiled shape is a
**declared validation shape** (`native_source_shape: false`) chosen because the
upstream tuned-size table covers it for v6e:

```text
n_batch=1024  n_in=4096  n_out=14336   (a Llama-3-70B MLP up-projection)
key (tpu_version=6, 1024, 14336, 4096, 'int8', 'int8') -> (1024, 1024, 4096)
```

## 2. Baselines

Two pure-JAX references, for two different jobs:

| Reference | Role |
|---|---|
| `quantized_matmul_per_channel` | exact fp32 contraction — the **correctness** reference |
| `quantized_matmul_per_channel_xla` | dequantize weights into the activation dtype, one ordinary matmul — the **speed** denominator |

Comparing against the fp32 one would flatter every kernel by an upcast nobody
would ship, so the XLA path is the denominator below. Neither uses Pallas.

Worth recording: sglang-jax's `xla_quantized_matmul_local` *looks* like an XLA
reference but dispatches into the block-wise Pallas kernel, so it is not a valid
baseline — the same trap as `jax.lax.ragged_dot` in the grouped-matmul family.

## 3. Correctness

All PASS at the 0.9999 cosine threshold, against the matching reference.

| Path | Implementation | Cosine | Max abs | Mean abs |
|---|---|---:|---:|---:|
| w8a8 | tpu-inference | 0.9999766946 | 2.000 | 1.481e-01 |
| w8a8 | sglang-jax | 0.9999766946 | 2.000 | 1.481e-01 |
| w8a16 | tpu-inference | 0.9999949336 | 1.000 | 5.226e-02 |
| w8a16 | sglang-jax | 0.9999949336 | 1.000 | 5.226e-02 |

Also verified at small shapes: per-channel at `(256,2048,1024)` and
`(512,1024,2048)` with and without activation quantization, and block-wise at
`block_size` 128 and 256 — 14 checks, all PASS, with the two repositories
agreeing to every digit.

The residual error is quantization, not rounding: the reference here is the XLA
dequantize path, so the reported difference includes each kernel's own
quantization choices. Mean relative error is 0.66% for w8a8 and 0.4% for
block-wise.

## 4. Timing

### w8a8 — int8 weights, int8 activations (the tuned path)

| Configuration | Median (ms) | Std (ms) | TOP/s | Speedup |
|---|---:|---:|---:|---:|
| **pure-XLA baseline** | **0.088507** | 0.000225 | 1358.7 | **1.000×** |
| sglang-jax | 0.094214 | 0.000267 | 1276.4 | 0.939× |
| tpu-inference | 0.094301 | 0.000368 | 1275.3 | 0.939× |

Both kernels select the table's `(1024, 1024, 4096)` tiling and land 6% behind
plain XLA. XLA's int8 matmul is already strong at this shape.

### w8a16 — int8 weights, unquantized bf16 activations

| Configuration | Median (ms) | Std (ms) | TOP/s | Speedup |
|---|---:|---:|---:|---:|
| **pure-XLA baseline** | **0.186746** | 0.000398 | 644.0 | **1.000×** |
| tpu-inference | 12.357151 | 0.004185 | 9.7 | **0.015×** |
| sglang-jax | 12.357422 | 0.004789 | 9.7 | **0.015×** |

**66× slower than plain XLA**, in both repositories, identically.

### Why: a tuned-table miss, confirmed from the recorded config

`result.json` records the tiling each kernel selected:

| Path | Selected `(batch, out, in, lane)` |
|---|---|
| w8a8 | `(1024, 1024, 4096, 1)` — table hit |
| w8a16 | `(128, 128, 128, 1)` — **fallback** |

The fallback tiling is 8× smaller in batch, 8× in output and 32× in the
contraction, which is the entire 66×.

The cause is in the tables themselves, and the two are not identical:

| Table | Entries | Activation dtypes present |
|---|---:|---|
| tpu-inference | 488 (319 for v6e) | `int8`, `float8_e4m3fn` only |
| sglang-jax | 608 (394 for v6e) | `int8`, `float8_e4m3fn`, **and 120 `bfloat16`** |

sglang-jax does tune bfloat16 activations — but **every one of those 120
entries pairs bfloat16 activations with float8 weights**, none with int8. So the
specific combination "int8 weights, unquantized activations" misses the table in
*both* repositories and falls back. `tests/test_quantized_matmul_tpu.py` asserts
this precisely, so if either table gains an `(bfloat16, int8)` entry the claim
gets revisited rather than going stale.

This is a real limitation of the kernels as shipped, not a corpus artifact:
these are w8a8 / w8-fp8a kernels, and the weight-only-quantized path exists in
the API without tuning behind it.

## 5. FLOP accounting and a caveat on MXU%

`logical = 2 × n_batch × n_in × n_out = 120,259,084,288` (120.3 GFLOP).

The reported `mxu_utilization_pct` is computed against the **918 TFLOP/s bf16**
peak, for consistency with the rest of the corpus. That is the wrong
denominator here: with int8 operands the hardware peak is roughly double, which
is why the w8a8 rows report **148% and 139%** "utilization". Those numbers are
not errors and are not above hardware peak — they are FLOPs divided by a bf16
peak that does not apply. Read the TOP/s column instead; the percentages are
retained only so every family's `result.json` has the same fields.

## 6. Bottleneck analysis

Per-operation self time from the first trace chunk:

| Configuration | Operation | Time | Share |
|---|---|---:|---:|
| XLA baseline, w8a8 | `dot_general` | 73.828 µs | 83.14% |
| XLA baseline, w8a8 | `convert_element_type` | 7.982 µs | 8.99% |
| XLA baseline, w8a8 | `reduce_max` | 6.899 µs | 7.77% |
| tpu-inference, w8a8 | `pallas_call` | 87.460 µs | 92.84% |
| tpu-inference, w8a8 | `reduce_max` | 6.732 µs | 7.15% |
| XLA baseline, w8a16 | `dot_general` | 156.952 µs | 83.99% |
| XLA baseline, w8a16 | `convert_element_type` | 29.879 µs | 15.99% |
| tpu-inference, w8a16 | `pallas_call` | 12353.558 µs | 99.94% |

- **w8a8 is a near-tie decided outside the matmul.** The kernel's
  `pallas_call` is 87.5 µs against XLA's 73.8 µs `dot_general`; both then pay
  the same ~6.8 µs `reduce_max` for the per-token abs-max, which the Pallas
  kernel cannot fold in because a Pallas program only sees one block of the
  input. XLA additionally pays 8.0 µs converting dtypes. The kernel's headroom
  is in the matmul itself, not in overhead.
- **w8a16 is entirely inside the Pallas call** (99.94%), consistent with a
  tiling problem rather than a launch or metadata problem.
- The XLA w8a16 baseline pays 29.9 µs (16%) materializing a dequantized bf16
  weight matrix — 14336×4096×2 = 117 MiB of extra traffic. That is the cost the
  Pallas kernel is designed to avoid by dequantizing in-register, which makes
  the 66× regression the more striking: the kernel gives up a real structural
  advantage to an untuned tiling.

Suggested next work:

1. Autotune the `(bfloat16, int8)` combination and add it to both tables; the
   structural argument says the kernel should *win* this case, not lose 66×.
2. Sweep tilings manually for w8a16 at this shape to confirm the fallback is
   the whole story — `(1024, 1024, 4096)` is already known good for w8a8.
3. Profile the block-wise entry point, which is migrated and correctness-tested
   here but not yet benchmarked.
4. Check the fp8 activation path (`float8_e4m3fn`), which both tables tune
   heavily and which this report does not cover.

## 7. Trace and artifact locations

Raw traces remain on the TPU:

```text
profiles/native/quantized_matmul/<run>/trace/
```

where `<run>` is `qm_<implementation>__w8a8` or `__w8a16`. Compact local
artifacts are the six `result.json` files beside this report and the `qm-*`
keys in `profiles/native/xplane_summary.json`.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in qm-baseline-xla qm-tpu-inference qm-sglang-jax; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --run-label w8a16 \
    --trace-chunk-size 1 --output-dir profiles/native
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --qm-quantize-activation --run-label w8a8 \
    --trace-chunk-size 1 --output-dir profiles/native
done

uv run --frozen --group profile python tools/summarize_xplane.py \
  --profiles profiles/native

uv run --frozen --with pytest python -m pytest \
  tests/test_quantized_matmul_tpu.py -q
```
