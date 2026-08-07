# Splash Attention TPU v6e Analysis

Generated: 2026-08-02 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 inputs, JIT compile, 5 warmups, 50 wall-clock diagnostics, then
50 device-profiled iterations, each in its own trace chunk.

**This is the corpus's first family with migrated backward kernels.**

## 1. Scope

| Source | Upstream path | Migrated | Audited |
|---|---|---:|---:|
| JAXBench | `benchmark/2p_GQA_Attention/optimized.py` | fwd + bwd-dQ + bwd-dKV | 3 |
| MaxText | `attention/splash_attention_kernel.py` | fwd + bwd-dQ + bwd-dKV | 3 |
| JAXBench | `benchmark/4p_Sparse_Attention/optimized.py` | **0** | 3 |
| MaxText | `tokamax_splash_attention/splash_attention_kernel.py` | 0 | 2 |
| Tokamax | `experimental/tpu/splash_attention/splash_attention_kernel.py` | fwd (elsewhere) | 2 |

**7 of the family's 13 audited launch points are migrated** — 6 here, plus
Tokamax's forward which lives in `kernels/attention/flash_attention/` because
its contract (`splash_mha_hsd`, pre-scaled Q, independent `Dqk`/`Dv`) differs
from the one here. Corpus directory and semantic family are not the same thing;
`inventory.json` models them separately.

Contract `splash_attention_mha`:

```text
q     [num_q_heads, seq_len, head_dim]
k, v  [num_kv_heads, seq_len, head_dim]      num_q_heads % num_kv_heads == 0
mask  a mask_lib.MultiHeadMask, one entry per query head
->    [num_q_heads, seq_len, head_dim]
```

Splash is **block-sparse**: the mask is compiled to block metadata ahead of the
call and the kernel visits only live blocks. There is no batch dimension — the
kernel is per-device and callers `vmap`, so the profiled shape is one device's
share of JAXBench's Llama-3.1-405B GQA config (which is 128 q-heads across
batch 4). Declared validation shape: `num_q_heads=32, num_kv_heads=8,
seq_len=4096, head_dim=128`, causal mask.

Both files are self-contained on the pinned dependency set: `mask_lib` and
`mask_info_lib` resolve inside `jax.experimental.pallas.ops.tpu.splash_attention`
in jax 0.10.2, not inside either upstream repository. Neither needed flattening.

## 2. Baseline

Both files ship a pure-JAX `attention_reference` upstream taking the same mask
object, verified Pallas-free. Those are used rather than a corpus-written
reference, for the same reason as in the ragged-paged-attention and MLA
families: re-deriving block-sparse masking by hand risks a subtly wrong
reference producing false failures. With `backward_impl="vanilla"` the
reference also provides a plain-autodiff gradient to check the backward
kernels against.

Because the two implementations come from different repositories, each is also
checked against the other's output.

## 3. Correctness

| Pass | Implementation | Cosine | Status |
|---|---|---:|---|
| forward | JAXBench | 0.99999875 | PASS |
| forward | MaxText | 0.99999875 | PASS |
| forward + backward | JAXBench | 0.99998677 | PASS |
| forward + backward | MaxText | 0.99998677 | PASS |

Per-gradient, at `8/2 heads, seq 512`:

| Gradient | Cosine | Mean-relative | RMS-relative |
|---|---:|---:|---:|
| dQ | 0.99998629 | 5.043e-03 | 5.291e-03 |
| dK | 0.99998617 | 4.776e-03 | 5.224e-03 |
| dV | 0.99999988 | 6.760e-04 | 6.820e-04 |

Cosine is the criterion, as everywhere else in the corpus. The absolute
differences look large (1.5–2.5) only because the gradients themselves are
large at this shape.

The relative errors are ~5e-3 on dQ and dK — two orders of magnitude looser
than the forward pass. That is expected rather than alarming: a causal mask
leaves many gradient entries at or near zero, and any relative statistic
divides by those. dV, which has no such structure, sits at 6.8e-4. The test
bounds RMS-relative error at 2e-2, set from these measurements with headroom.

**JAXBench and MaxText agree to 8 digits on every measurement** despite sharing
only ~15% of their top-level definitions — different code, same math.

## 4. Timing

| Configuration | Median (ms) | Std (ms) | TFLOP/s | MXU @ 918 |
|---|---:|---:|---:|---:|
| JAXBench, forward | **0.76177** | 0.00046 | 360.8 | 39.31% |
| MaxText, forward | **0.76163** | 0.00046 | 360.9 | 39.31% |
| JAXBench, forward + backward | 20.48830 | 0.00147 | 40.2 | 4.38% |
| MaxText, forward + backward | 20.48779 | 0.00163 | 40.2 | 4.38% |

The forward pass reaches **39.3% MXU**, in line with the corpus's other tuned
attention kernels (JAXBench flash 37.7%, Tokamax splash 42.1%).

The two implementations are within 0.02% of each other in both passes.

## 5. Block sizes dominate everything else here

`BlockSizes.get_default()` is 128 in every dimension and carries an upstream
`TODO(apaszke,sharadmv): Select better parameters based on a heuristic`.
Measured against JAXBench's autotuned forward blocks (`block_q=2048`,
`block_kv=2048`, `block_kv_compute=1024`):

| Forward blocks | Median (ms) | Std (ms) | MXU |
|---|---:|---:|---:|
| `get_default()` (128³) | 13.59408 | 0.00385 | 2.20% |
| JAXBench autotuned | **0.76177** | 0.00046 | **39.31%** |

Both rows are the full 5+50 protocol. The default-block run takes about an hour
to profile on this device because at 128³ the kernel emits far more trace events
per iteration; that cost is a property of tracing it, not of the kernel.

That is a **17.8× difference on identical code** — larger than any
implementation-vs-implementation gap in this family, and consistent with the
pattern seen in ragged paged attention (tuning worth 19%) and quantized matmul
(a tuned-table miss worth 66×). Profiling splash with default blocks would have
reported these kernels at 2% MXU and been badly misleading.

A test pins the default at 128³ and the autotuned values at 2048, so if either
changes upstream the profiling choice gets revisited.

### The backward pass is untuned upstream

JAXBench autotuned the **forward only**. Every backward block is `None`:

```python
# Not autotuned (backward-only).
'block_q_dkv': None, 'block_kv_dkv': None, 'block_kv_dkv_compute': None,
'block_q_dq': None, 'block_kv_dq': None,
```

The backward pass rejects `None` outright (`ValueError: Need to specify backward
blocks`), so the profiles above fall back to the upstream default of 128 for
those five, and say so in `result.json`. That is why forward+backward costs
**26.9× the forward** rather than the ~3× a tuned backward would suggest: the
forward runs at 2048 tiles and the backward at 128.

The 3× FLOP convention used for the `-grad` rows (a backward costs roughly twice
its forward) is therefore a *convention*, not a measurement — it is recorded in
the result notes as such, and the resulting 4.38% MXU should be read as "the
backward is untuned", not as a property of the algorithm.

## 6. What is not migrated, and why

- **JAXBench `4p_Sparse_Attention`** (3 launches) is the *same kernel* as
  `2p_GQA_Attention`: 28 of their 31 top-level definitions are AST-identical,
  differing only in `create_inputs`, `get_flops` and `workload`. Migrating it
  would double-count, the same treatment given to `3p_MLA_Attention` in the MLA
  family.
- **MaxText `tokamax_splash_attention/`** (2 launches) is MaxText's vendored
  copy of Tokamax's kernel. It has diverged from both its own splash kernel and
  from Tokamax's — only 35% of shared definitions are AST-identical to
  Tokamax's — and it needs three repo-local modules (base, mask, mask_info).
- **Tokamax's `_splash_attention_bwd_dkv`**: only its forward is in the corpus.

Suggested next work:

1. Autotune the backward blocks. Nothing upstream has, and at 26.9× the forward
   this is the largest single lever in the family.
2. Migrate Tokamax's splash backward-dKV, completing that lineage.
3. Flatten MaxText's vendored `tokamax_splash_attention` copy — it is a
   three-way divergence worth measuring, since it sits between the two kernels
   already migrated.

## 7. Trace and artifact locations

```text
profiles/native/splash_attention/<run>/trace/
```

Compact local artifacts are the `result.json` files beside this report.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in splash-jaxbench splash-maxtext \
            splash-jaxbench-grad splash-maxtext-grad; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --trace-chunk-size 1 --output-dir profiles/native
done

# The untuned-block comparison in §5.
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation splash-jaxbench --splash-default-blocks \
  --run-label default-blocks --trace-chunk-size 1 --output-dir profiles/native

uv run --frozen --with pytest python -m pytest \
  tests/test_splash_attention_tpu.py -q
```
