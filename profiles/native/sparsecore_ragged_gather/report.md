# SparseCore Ragged Gather TPU v6e Analysis

Generated: 2026-08-02 (America/New_York)
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB HBM, 128 MiB VMEM)
SparseCore: 2 cores × 16 subcores, 8 SIMD lanes (`pltpu.get_tpu_info().sparse_core`)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: JIT compile, 5 warmups, 50 wall-clock diagnostics, then 50
device-profiled iterations, each in its own trace chunk.

**These are the corpus's first SparseCore kernels.** Every other family runs on
the TensorCore; these launch through `pl.kernel` over a
`plsc.VectorSubcoreMesh` and run on the v6e's SparseCores instead. MXU
utilization is meaningless here — a gather does no arithmetic — so the metric is
achieved bandwidth.

Gathers are measured in **both fp32 and bf16**. These kernels move rows without
computing on them and are dtype-generic (bit-exact in both, verified). fp32 is
what upstream's tests use and is the dtype for §§3-5; bf16 is what an MoE layer
actually carries, and §8 shows the choice reverses the overlap verdict.

## 1. Scope

| Source | Upstream path | Migrated | Audited |
|---|---|---:|---:|
| Tokamax | `ops/ragged_gather/pallas_mosaic_tpu_kernel.py` | fwd | 1 |
| Tokamax | `ops/ragged_gather/pallas_mosaic_v2_tpu_kernel.py` | fwd | 1 |
| Tokamax | `ops/ragged_gather_reduce/pallas_mosaic_tpu_kernel.py` | fwd | 1 |
| MaxText | `ragged/ragged_gather.py` | fwd | 1 |
| Tokamax | `ops/ragged_scatter/pallas_mosaic_tpu_kernel.py` | **0** | 1 |
| MaxText | `ragged/ragged_gather_reduce{,_v2}.py`, `gather_reduce_pallas.py` | 0 | 3 |
| tpu-inference | `sparse_core/*` | 0 | 4 |

**4 of the family's 12 audited launch points are migrated.** All four are
self-contained upstream — they import nothing outside `jax` — so
`tools/flatten_sparsecore_gather.py` only attaches provenance; no flattening or
import rewriting was needed.

Contract `ragged_gather`:

```text
x        [num_rows, hidden]
indices  [out_rows]           int32
start, end                    int32[1], the live range of `indices`
->       [padded_out_rows, padded_hidden]

out[i] == x[indices[i]]  for i in [start, end)
```

The output is padded to the SparseCore block size and column tile, so only the
live rows and the first `x.shape[-1]` columns carry data. MaxText adds an
optional `weights` argument giving `weights[:, None] * x[indices]`.

Contract `ragged_gather_reduce` is the MoE combine step: gather, scale rows by
`topk_weights`, zero the rows `valid_rows_mask` excludes, then sum consecutive
groups of `reduce_group_size` rows.

## 2. Baseline

Unusually for this corpus, the references are not reverse-engineered. Each
upstream kernel states its own semantics by falling back to exactly them when
no SparseCore is present:

```python
sc_info = pltpu.get_tpu_info().sparse_core
if sc_info is None:
  return x[indices]          # upstream's own fallback
```

So `baseline.py` uses `x[indices]`, and a test asserts that string is still in
each kernel's source, so the reference gets revisited rather than silently
drifting if upstream's fallback ever changes.

## 3. Correctness

| Contract | Implementation | Cosine | Max abs diff | Status |
|---|---|---:|---:|---|
| `ragged_gather` | Tokamax | 1.00000000 | **0** | PASS |
| `ragged_gather` | Tokamax v2 | 1.00000000 | **0** | PASS |
| `ragged_gather` | MaxText | 1.00000000 | **0** | PASS |
| `ragged_gather_reduce` | Tokamax | 1.00000000 | 7.15e-07 | PASS |

The three gathers are **bit-exact**, which is the right expectation: a gather
moves data, it does not compute. Verified over the full output and over a
partial live range (`end = out_rows // 2`). MaxText's weighted path was checked
separately against `weights[:, None] * x[indices]`.

`gather_reduce` is not bit-exact because it sums groups of rows in a different
order than the reference; 7.15e-07 on fp32 is reassociation, nothing more.

## 4. The gather_reduce kernel refuses to run on small inputs

This nearly produced a wrong result, and it is the most useful thing in this
family.

`ragged_gather_reduce_pallas` carries a **second** fallback the plain gathers
do not, and it is silent:

```python
dtype_bytes = jax.dtypes.itemsize_bits(x.dtype) // 8
if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
  return _fallback_implementation(...)     # plain XLA
```

On a v6e (128 MiB VMEM) fp32 `x` must exceed **38.4 MiB** before the SparseCore
path is used at all. Upstream's judgement is that XLA already wins when `x` fits
comfortably in VMEM. It additionally asserts
`num_cores // num_column_partitions <= num_lanes`, which needs `hidden >= 2048`
on this device.

The first profile of this family ran at `num_rows=4096, hidden=1024` — 16 MiB,
well under the threshold. Re-measured under the full protocol and kept as an
artifact (`*__xla-fallback/`, `measures_xla_fallback: true`):

| Implementation | Median (ms) | Std (ms) | `pallas_launches` |
|---|---:|---:|---:|
| baseline (XLA) | 0.019074 | 0.000050 | 0 |
| "Tokamax gather_reduce" | 0.019109 | 0.000058 | **0** |

A 0.18% difference. That is not a kernel that matches XLA; **it *is* XLA**. The
lowered HLO contained zero `tpu_custom_call`s. Correctness passed, of course:
upstream's `_fallback_implementation` is the same expression as the corpus
reference, differing only by two casts that are no-ops at fp32.

Two changes came out of this:

- The two contracts now have **different declared shapes**, and for a
  correctness reason rather than a tuning one: `ragged_gather` at
  `num_rows=4096, hidden=1024`, `ragged_gather_reduce` at `num_rows=8192,
  hidden=2048`.
- `tools/profile_kernel.py` now counts `tpu_custom_call`s in the lowered HLO
  before every run and **refuses** to profile a non-reference implementation
  that lowers to zero, or a reference baseline that lowers to more than zero.
  Every `result.json` in the corpus now records `pallas_launches`. This is a
  corpus-wide guard: several upstream kernels fall back at runtime, and until
  now nothing would have caught it.

A test pins the threshold from both sides — XLA below it, Pallas above it.

## 5. Timing

Contract `ragged_gather`, `num_rows=4096, hidden=1024, out_rows=2048`:

| Implementation | Median (ms) | Std (ms) | GB/s | vs XLA |
|---|---:|---:|---:|---:|
| baseline `x[indices]` (XLA) | **0.010270** | 0.000043 | **1634.3** | 1.00× |
| Tokamax | 0.042032 | 0.000255 | 399.3 | 4.09× slower |
| MaxText | 0.042251 | 0.000204 | 397.3 | 4.11× slower |
| Tokamax v2 | 0.068644 | 0.000168 | 244.5 | 6.68× slower |

Contract `ragged_gather_reduce`, `num_rows=8192, hidden=2048, out_rows=2048`,
`reduce_group_size=4`:

| Implementation | Median (ms) | Std (ms) | GB/s | vs XLA |
|---|---:|---:|---:|---:|
| baseline (XLA) | **0.071845** | 0.000142 | 467.2 | 1.00× |
| Tokamax | 0.100153 | 0.000429 | 335.1 | 1.39× slower |

### Reading these numbers

XLA's gather runs at **1634 GB/s**, which is essentially the v6e's HBM
bandwidth (~1640 GB/s). There is no headroom to win at this dtype, and no
SparseCore kernel beats it in isolation.

That is *not*, however, because the operation is purely memory-bound — §8
measures the same gathers in bf16 and finds every one of them gets **slower**
despite moving half the bytes. At `hidden=1024` there is a per-row and
per-index floor that fp32 happens to sit right at the bandwidth limit of. Read
the GB/s column as "what fp32 achieves here", not as a statement about what
limits the kernel.

That is not the argument for these kernels. **A SparseCore gather's value is
that it runs on hardware the TensorCore is not using.** In an MoE layer the
gather overlaps with matmuls, so the cost that matters is the *marginal* cost
once TensorCore work is running alongside it — not the 0.042 ms measured here
with the rest of the chip idle. Measuring these kernels standalone measures the
thing they are designed not to be.

So the numbers above are true and incomplete: **in isolation the SparseCore
gathers are 4–6.7× slower than XLA's.** Sections 6-9 measure the case they are
actually for.

The v1-vs-v2 gap is real and measurable, though: **v1 is 1.63× faster than v2**
at this shape, on identical semantics and bit-identical output. Both are current
upstream.

## 6. Overlap: what the gather costs while a matmul is running

`tools/overlap_harness.py` times five configurations and takes differences:

```text
marginal cost = T(matmul + gather) - T(matmul)      what the gather adds
hidden        = 1 - marginal / T(gather alone)      1.0 free, 0.0 serialized
```

Each configuration runs in its own process under the same 5+50 protocol. The
matmul and the gather are compiled into one jitted function with **no data
dependency**, returning both outputs so neither is dead code; whether XLA
overlaps them is measured, not assumed. The TensorCore workload is a square
bf16 matmul, whose arithmetic intensity is N/3 — so N=1024 sits below the v6e's
ridge point (577.96 FLOP/byte) and N≥2048 above it.

Alongside the timing, the harness reads the trace directly. The TPU trace keeps
the TensorCore (`XLA Ops`) and each SparseCore (`Sparse Core Ops`, plus 16
`TEC n` subcore tracks) on separate tracks, so `overlap` below is the fraction
of SparseCore busy time that fell *inside* TensorCore busy time — measured, not
inferred from timing.

### Sweeping the matmul, gather fixed at 2048 rows (16 MiB)

| N | TFLOP/s | SC marginal | SC hidden | SC overlap | XLA marginal | XLA hidden |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 313.2 | 0.03735 | 0.11 | 0.21 | **0.00948** | 0.08 |
| 2048 | 578.8 | 0.01825 | 0.57 | 0.97 | **0.01253** | −0.22 |
| 4096 | 807.6 | 0.03840 | 0.09 | 1.00 | **0.03660** | −2.56 |
| 8192 | 727.7 | 0.02139 | 0.49 | 1.00 | **0.01202** | −0.17 |

**Temporal overlap works.** By N=2048 the trace shows 97–100% of SparseCore
busy time falling inside TensorCore busy time. At N=1024 it is only 21%, for
the obvious reason: that matmul takes 6.9 µs and the gather takes 42 µs, so
there is not enough TensorCore work to hide behind.

**But full overlap is not free.** SparseCore `hidden` never exceeds 0.68
anywhere in this study, and XLA still wins every row above. Running
concurrently and running for free are different things.

### The decomposition that explains it

The harness splits marginal cost into the TensorCore running longer
(`slowdown`, read from the trace) and everything else (`edge` — launch and
drain):

| Configuration | marginal | slowdown | edge |
|---|---:|---:|---:|
| SparseCore, N=2048 | 0.01825 | 0.00115 | 0.01709 |
| SparseCore, N=4096 | 0.03840 | 0.02173 | 0.01667 |
| SparseCore, N=8192 | 0.02139 | 0.00471 | 0.01667 |
| XLA, N=2048 | 0.01253 | 0.01240 | 0.00013 |
| XLA, N=4096 | 0.03660 | 0.03598 | 0.00062 |
| XLA, N=8192 | 0.01202 | 0.01055 | 0.00148 |

Two clean facts:

- **XLA's `edge` is ~0.** Its entire marginal cost is the TensorCore running
  longer, which is exactly right — an XLA gather *is* TensorCore work.
- **SparseCore's `edge` is a fixed ~17–19 µs**, essentially constant across
  every matmul size and (below) every gather size. That is SparseCore program
  launch and drain, and it never overlaps with anything.

A fixed 17 µs against a 42 µs gather is 40% overhead, which is the whole reason
SparseCore loses at this size. It also predicts where it should stop losing.

### Sweeping the gather, matmul fixed at N=8192

| Gather rows | Bytes moved | SC alone | SC marginal | XLA alone | XLA marginal | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 2,048 | 16 MiB | 0.04206 | 0.02139 | 0.01029 | **0.01202** | XLA, 1.78× |
| 8,192 | 64 MiB | 0.10331 | 0.03754 | 0.03697 | **0.03557** | XLA, 1.06× |
| 16,384 | 128 MiB | 0.18484 | **0.06257** | 0.07265 | 0.07113 | **SparseCore, 1.14×** |
| 32,768 | 256 MiB | 0.34744 | **0.11031** | 0.14420 | 0.14332 | **SparseCore, 1.30×** |

**The crossover is real, and it is between 64 MiB and 128 MiB moved** — roughly
12,000 gathered rows at `hidden=1024`, fp32. Above it the SparseCore kernel is
the cheaper way to gather while a matmul runs, by 14% at 16K rows and 30% at
32K rows.

The mechanism is visible in the slowdown term: SparseCore costs the TensorCore
consistently **~35% less** than XLA does (0.01837 vs 0.03327 at 8K rows;
0.09197 vs 0.14148 at 32K). That discount is the offload actually working. It
buys nothing until it exceeds the fixed ~18 µs launch cost, which is why
break-even lands where it does — 0.35 × XLA-slowdown = 18 µs implies an XLA
slowdown around 51 µs, and that is what a ~12K-row gather costs.

A caveat that turned out to matter: the workload above is **one dense matmul**,
not the grouped matmul an MoE layer actually runs. The next section re-runs the
crossover against the real thing, and the answer changes.

N=4096 is anomalous for both gathers (XLA `hidden` = −2.56). That matmul runs at
807.6 TFLOP/s, 88% of peak and the most efficient point in the sweep, so
concurrent memory traffic costs it the most. It affects both gathers similarly
and does not disturb the comparison.

## 7. The same crossover against a grouped matmul — it disappears

The dense matmul above is a convenient TensorCore workload, not a representative
one. Re-run against the corpus's grouped matmul — JAXBench's kernel at the shape
already profiled in `profiles/native/grouped_matmul/` (`rows=32768,
groups=128, k=4096, n=1536`, 1.87016 ms, 220.5 TFLOP/s), which conveniently
takes about as long as the dense N=8192 matmul:

| Gather rows | SC marginal | SC hidden | SC overlap | XLA marginal | XLA hidden | XLA/SC |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.01174 | 0.72 | 0.98 | **0.01010** | 0.02 | 0.860 |
| 8,192 | 0.04706 | 0.54 | 0.99 | **0.03683** | 0.00 | 0.783 |
| 16,384 | 0.09331 | 0.50 | 1.00 | **0.07249** | 0.00 | 0.777 |
| 32,768 | 0.19211 | 0.45 | 1.00 | **0.14479** | −0.00 | 0.754 |

**XLA is cheaper at every gather size, and the gap widens with size** (XLA/SC
falls 0.860 → 0.754). Against the dense matmul the same sweep crossed over at
16,384 rows and reached 1.30× in SparseCore's favour. Against the grouped
matmul there is no crossover at all, and the trend runs the other way.

The §6 conclusion was measured correctly and is not representative of an MoE
workload. Note that §7 is itself superseded by §8, which changes the gather to
the dtype an MoE layer actually carries: **§8 has the answer to use.**

### Why it reverses

The decomposition explains it cleanly, and the two terms move in opposite
directions:

| | dense N=8192, 32K rows | GMM, 32K rows |
|---|---:|---:|
| SparseCore edge (fixed launch) | 0.01834 | **−0.00108** |
| SparseCore TensorCore slowdown | 0.09197 | 0.19318 |
| XLA TensorCore slowdown | 0.14148 | 0.14542 |
| SparseCore slowdown vs XLA | **35% less** | **33% more** |

- **The fixed ~18 µs launch cost vanishes.** Against the GMM the edge term is
  ~0 at every size. That was SparseCore's whole handicap in §6, and it is gone —
  because the GMM has already spun up SparseCore machinery for its own work
  (below), so the gather's launch is no longer paying to start it cold.
- **But the slowdown term flips sign.** Offloading the gather stops being a
  discount and becomes a penalty: 0.19318 against XLA's 0.14542.

The candidate explanation was HBM headroom, and the two workloads differ sharply
on it:

| Workload | Bytes moved | Achieved | % of 1640 GB/s peak | Headroom |
|---|---:|---:|---:|---:|
| dense N=8192 | 384 MiB | 266.5 GB/s | 16.3% | **83.7%** |
| grouped matmul | 1888 MiB | 1058.6 GB/s | 64.5% | **35.5%** |

A gather is pure bandwidth, so the story writes itself: the dense matmul is
compute-bound (AI 2731 against a 578 ridge point) and leaves five-sixths of the
bus free for a concurrent gather, while the grouped matmul is memory-bound (AI
208) and already consumes two-thirds of it.

**That explanation is wrong.** Section 9 sweeps the GMM shape across a 14× range
of arithmetic intensity and a 4× range of HBM utilization, and the result does
not move at all. It is recorded here as the hypothesis it was, because the
measurement that killed it only exists because it was written down.

### The SparseCore is not idle during an MoE matmul

Worth stating separately, because it undercuts the premise rather than the
measurement: **XLA already offloads work to the SparseCore inside the grouped
matmul.** With no gather present at all, the GMM's trace shows ~10.3 µs of
`scatter_offload_custom_fusion` and `copy` ops on the SparseCore track.

So "the SparseCore is sitting idle, use it" is not quite true of the workload
these kernels target — the compiler got there first, and a SparseCore gather
contends for the SparseCore as well as for HBM. The harness separates the two
(`workload_sparsecore_busy_us` vs `sparsecore_busy_us`) so the gather is never
credited with the workload's own offload.

Note that the gather is still **fully overlapped in time** (0.98–1.00) in every
row above. Concurrency was never the problem. This is the §6 lesson in its
sharpest form: *running concurrently and running for free are different things*,
and against a bandwidth-bound workload the difference is the entire result.

### What this establishes, with fp32 gathers

- Against the workload these kernels are written for, at fp32 the SparseCore
  gathers are **not** the cheaper option at any size measured — 16% worse at 2K
  rows, 33% worse at 32K.
- The dense sweep shows the SparseCore advantage existing under *some*
  conditions; §9 establishes that HBM headroom is not the condition.

The `fp32` qualifier turns out to carry the whole result. Section 8 switches the
gather to bf16 — what an MoE layer actually carries — and the verdict reverses
again.

## 8. bf16 gathers reverse it — and not for the expected reason

fp32 was inherited from upstream's own tests, not from how these kernels get
used: MoE activations are bf16. Re-running every §7 configuration with a bf16
gather, same shapes, same protocol:

| Gather rows | SC marginal | SC hidden | XLA marginal | XLA/SC | (fp32 was) |
|---:|---:|---:|---:|---:|---:|
| 2,048 | **0.00973** | 0.77 | 0.01233 | **1.267** | 0.860 |
| 8,192 | **0.03621** | 0.67 | 0.04270 | **1.179** | 0.783 |
| 16,384 | **0.07419** | 0.62 | 0.08186 | **1.103** | 0.777 |
| 32,768 | **0.14341** | 0.62 | 0.15815 | **1.103** | 0.754 |

**In bf16 the SparseCore gather is the cheaper option at every size**, by
10–27%, against the grouped matmul. In fp32 it lost at every size. Nothing
changed but the element width.

The dense matmul moves the same way, further:

| Gather rows | SC marginal | SC hidden | XLA marginal | XLA/SC | (fp32 was) |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.01649 | 0.61 | **0.01419** | 0.861 | 0.562 |
| 8,192 | **0.02503** | 0.77 | 0.04227 | **1.689** | 0.948 |
| 16,384 | **0.03828** | 0.81 | 0.08066 | **2.107** | 1.137 |
| 32,768 | **0.06803** | 0.82 | 0.15976 | **2.348** | 1.299 |

### The reason is not "half the bytes"

The obvious explanation would be that a bf16 gather moves half the data. That
is not what happened. **Standalone, every gather got _slower_ in bf16:**

| Gather rows | SC alone fp32 → bf16 | XLA alone fp32 → bf16 |
|---:|---|---|
| 2,048 | 0.04206 → 0.04272 (**+2%**) | 0.01029 → 0.01303 (**+27%**) |
| 8,192 | 0.10331 → 0.10926 (**+6%**) | 0.03697 → 0.04215 (**+14%**) |
| 16,384 | 0.18484 → 0.19706 (**+7%**) | 0.07265 → 0.08094 (**+11%**) |
| 32,768 | 0.34744 → 0.37372 (**+8%**) | 0.14420 → 0.15850 (**+10%**) |

Halving the element width made the isolated gather 2–27% *more* expensive. So
**a gather at `hidden=1024` is not bandwidth-bound in isolation** — at 4 KiB
(fp32) or 2 KiB (bf16) per row the cost is dominated by per-row and per-index
work, and narrowing the elements only adds sub-word packing. This also
retroactively explains §5: those standalone numbers were never measuring
bandwidth limits either.

But the *marginal* cost — what the gather adds alongside a workload — is
contention, and contention does scale with bytes. The two move in opposite
directions, and only one of the gathers captures the benefit:

| Against the GMM, 32K rows | fp32 | bf16 | change |
|---|---:|---:|---|
| SparseCore marginal | 0.19211 | 0.14341 | **−25%** |
| XLA marginal | 0.14479 | 0.15815 | **+9%** |

The SparseCore gather converts the halved byte count into a 17–25% cheaper
marginal cost. XLA's gets 9–22% *more* expensive, because it runs on the
TensorCore, where the sub-word packing penalty lands on the same unit that is
also doing the matmul — it pays the bf16 cost without collecting the
contention saving.

That asymmetry, not bandwidth alone, is what flips the result.

### What this establishes

- **In bf16 — the realistic dtype — the SparseCore gathers win against both
  workloads**, by 10–27% against the grouped matmul and up to 2.35× against a
  dense matmul. Taken together with §7: the fp32 answer is real but is not the
  one that applies to an MoE layer.
- The kernels are **bit-exact in bf16** as well as fp32, verified for all three
  implementations. A gather moves rows without computing on them, so this is
  the right expectation and it holds; a test pins it.
- The one place XLA still wins is the smallest gather against the dense matmul
  (2K rows, XLA/SC 0.861) — the fixed SparseCore launch cost from §6 has not
  gone anywhere, it is just amortized sooner in bf16.
- Two dtypes, two workloads, one chip, one GMM shape — §9 removes the last of
  those by sweeping the GMM shape, and in doing so refutes §7's stated
  mechanism.

Artifacts: `overlap/summary.json` and the per-run `overlap/*/result.json`. Raw
traces are discarded by default — this study is 55 runs — and `--keep-traces`
retains them for one run.

## 9. Sweeping the GMM shape: arithmetic intensity does not explain anything

§7 attributed its reversal to HBM headroom. This section tests that by holding
the gather fixed (32,768 rows) and sweeping the grouped matmul across a **14×
range of arithmetic intensity** and a **4× range of HBM utilization**. Shapes
vary `rows` and `groups` at fixed `k=4096, n=1536`; because the expert weights
are 81–95% of the traffic, arithmetic intensity here is essentially **tokens per
expert**, which is a knob MoE designs actually turn.

Sorted by HBM utilization, fp32 gathers:

| GMM rows×groups | AI | HBM used | own SC | SC marginal | XLA marginal | XLA/SC |
|---|---:|---:|---:|---:|---:|---:|
| 32768×128 (native) | 208 | 64.5% | 10.3 µs | 0.19211 | 0.14479 | 0.754 |
| 32768×256 | 115 | 60.5% | 41.7 µs | 0.18439 | 0.14382 | 0.780 |
| 131072×128 | 534 | 25.8% | 10.5 µs | 0.19163 | 0.14289 | 0.746 |
| 32768×32 | 534 | 25.3% | 9.6 µs | 0.19632 | 0.14523 | 0.740 |
| 32768×8 | 878 | 15.3% | 9.6 µs | 0.19419 | 0.14563 | 0.750 |
| *dense N=8192* | *2731* | *16.3%* | *none* | *0.11031* | *0.14332* | ***1.299*** |

And bf16:

| GMM rows×groups | AI | HBM used | SC marginal | XLA marginal | XLA/SC |
|---|---:|---:|---:|---:|---:|
| 32768×128 (native) | 208 | 64.5% | 0.14341 | 0.15815 | 1.103 |
| 32768×256 | 115 | 60.5% | 0.13218 | 0.15870 | 1.201 |
| 131072×128 | 534 | 25.8% | 0.14171 | 0.15784 | 1.114 |
| 32768×32 | 534 | 25.3% | 0.14337 | 0.15822 | 1.104 |
| 32768×8 | 878 | 15.3% | 0.14388 | 0.15837 | 1.101 |
| *dense N=8192* | *2731* | *16.3%* | *0.06803* | *0.15976* | ***2.348*** |

### The headroom hypothesis is refuted, two ways

**Directly.** Compare the last two rows of each table. The dense matmul at
**16.3%** HBM gives XLA/SC 1.299; the grouped matmul at **15.3%** HBM — *less*
bandwidth used, more headroom — gives 0.750. Matched headroom, opposite
verdicts. Bandwidth availability cannot be what separates them.

**By trend.** Across the GMM sweep, HBM utilization falls from 64.5% to 15.3%
and arithmetic intensity rises 115 → 878, while XLA/SC stays flat: **0.754,
0.780, 0.746, 0.740, 0.750** in fp32 and **1.103, 1.201, 1.114, 1.104, 1.101**
in bf16. There is no trend to speak of, in either dtype.

The two shapes that reach AI 534 by different routes — 32768×32 and 131072×128,
a 4× difference in total size — agree to within 1% (0.740 vs 0.746; 1.104 vs
1.114). The sweep is measuring what it thinks it is.

### So what does separate a dense matmul from a grouped one?

Not the gather side: **XLA's marginal cost is essentially constant across every
workload and shape** (0.143–0.146 fp32, 0.158–0.159 bf16). It is the SparseCore
arm that is ~2× more expensive against a GMM (≈0.19) than against a dense matmul
(0.11) at matched headroom.

The visible structural difference is that the grouped matmul is itself a Mosaic
kernel that **uses the SparseCore on its own account**, while the dense matmul is
a single XLA op that does not. But the amount of that usage does not correlate
either: 41.7 µs of own-SparseCore work gives 0.780, and 9.6 µs gives 0.750.

So this section establishes what the mechanism *is not* — not bandwidth, not
arithmetic intensity, not the volume of the workload's own SparseCore use — and
does not establish what it is. Pinning it down would mean varying the workload
kind rather than its shape: a Mosaic kernel that never touches the SparseCore
would separate "is a Pallas kernel" from "uses the SparseCore".

### One shape behaves entirely differently

`rows=8192, groups=128` (64 tokens per expert) is an outlier large enough that
averaging it in would be misleading, so it is excluded from the trend above and
reported separately:

| dtype | SC marginal | SC hidden | XLA marginal | XLA/SC |
|---|---:|---:|---:|---:|
| float32 | **+0.00450** | 0.99 | 0.14345 | 31.855 |
| bfloat16 | **−0.04269** | 1.11 | 0.15816 | −3.705 |

In bf16 the marginal cost is **negative**: adding the SparseCore gather made the
whole module 43 µs *faster* than the grouped matmul alone. `hidden` above 1.0 is
not physically meaningful under the "how much was hidden" reading, and the ratio
column's sign is an artifact. `summarize` now flags any row with a
non-positive marginal rather than letting the ratio be read as a spectacular
result.

The measurement is reproducible and the XLA arm at this shape is completely
ordinary (0.14345 / 0.15816, in line with every other shape), so this is not a
broken workload baseline — something specific happens when a SparseCore gather
runs alongside *this* grouped matmul. It is the shape with by far the fewest
tokens per expert, where the kernel is dominated by weight loading. Beyond that
the cause is unknown and it is left as an open anomaly, not explained away.

## 10. What is not migrated, and why

- **`ragged_scatter`** (1 launch) runs on this device, but under the obvious
  calling convention it returns *gather* results. Measured with a cyclic-shift
  permutation — chosen because a reversal is an involution and cannot tell the
  two apart:

  ```text
  out == x[indices]      -> True
  out[indices] == x      -> False
  ```

  Its `_preprocess_indices` derives separate `src_indices` and `dst_indices`, so
  the likelier explanation is that the intended convention is not the one tried
  here, rather than the kernel being mislabelled. Counting it as validated would
  be wrong, so it is left out — the same call made for Tokamax's SparseCore
  top-k in the `topk_routing` family.
- **MaxText's three other gather_reduce launches** and **tpu-inference's four
  `sparse_core` kernels** are audited but not migrated; the tpu-inference ones
  need a `core_map_helper` module.

Suggested next work, in the order §9 leaves them:

1. **Separate "is a Pallas kernel" from "uses the SparseCore."** §9 rules out
   bandwidth, arithmetic intensity, and the volume of the workload's own
   SparseCore use, leaving the workload *kind* as the difference between a
   dense matmul (SparseCore wins) and a grouped one (it does not). Running the
   sweep against a Mosaic kernel that never touches the SparseCore — splash
   attention is already in the corpus — would isolate which of the two matters.
2. **Chase the `rows=8192` anomaly.** A negative marginal cost is either a
   scheduling effect worth understanding or a measurement artifact worth
   knowing about, and at 64 tokens per expert it sits in a regime real MoE
   inference reaches.
3. Re-measure §5 as a per-row cost. §8 shows the standalone gathers are not
   bandwidth-bound at `hidden=1024`, so the GB/s figures in §5 describe a regime
   the kernel is not actually in; a rows-per-second number would be the honest
   metric, and sweeping `hidden` would locate where bandwidth does take over.
4. Find out why v2 is 1.63× slower than v1 and whether it wins at some other
   shape; if it does not, that is worth reporting upstream.
5. Resolve the `ragged_scatter` convention, which would add a fifth kernel.

## 11. Trace and artifact locations

```text
profiles/native/sparsecore_ragged_gather/<run>/trace/
```

Compact local artifacts are the `result.json` files beside this report.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in scg-baseline-gather scg-tokamax scg-tokamax-v2 scg-maxtext \
            scg-baseline-gather-reduce scg-tokamax-gather-reduce; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --trace-chunk-size 1 --output-dir profiles/native
done

uv run --frozen --with pytest python -m pytest \
  tests/test_sparsecore_ragged_gather_tpu.py tests/test_overlap_harness.py -q
```

The §6 overlap study, one configuration per process:

```bash
for c in gather-sparsecore gather-xla; do
  uv run --frozen --group profile python tools/overlap_harness.py \
    --config $c --output-dir profiles/native
done
for n in 1024 2048 4096 8192; do
  for c in matmul matmul+gather-sparsecore matmul+gather-xla; do
    uv run --frozen --group profile python tools/overlap_harness.py \
      --config $c --matmul-n $n --output-dir profiles/native
  done
done

# The gather-size sweep behind §6.
for g in 8192 16384 32768; do
  for c in gather-sparsecore gather-xla matmul+gather-sparsecore matmul+gather-xla; do
    uv run --frozen --group profile python tools/overlap_harness.py \
      --config $c --matmul-n 8192 --scg-out-rows $g --output-dir profiles/native
  done
done

# The grouped-matmul sweep behind §7. Defaults are the shape already profiled
# in profiles/native/grouped_matmul/.
uv run --frozen --group profile python tools/overlap_harness.py \
  --config gmm --output-dir profiles/native
for g in 2048 8192 16384 32768; do
  for c in gmm+gather-sparsecore gmm+gather-xla; do
    uv run --frozen --group profile python tools/overlap_harness.py \
      --config $c --scg-out-rows $g --output-dir profiles/native
  done
done

# Derive the tables; needs no TPU.
# The bf16 sweep behind §8 -- same configurations, --gather-dtype bfloat16.
for g in 2048 8192 16384 32768; do
  for c in gather-sparsecore gather-xla gmm+gather-sparsecore gmm+gather-xla; do
    uv run --frozen --group profile python tools/overlap_harness.py \
      --config $c --gather-dtype bfloat16 --scg-out-rows $g \
      --output-dir profiles/native
  done
  for c in matmul+gather-sparsecore matmul+gather-xla; do
    uv run --frozen --group profile python tools/overlap_harness.py \
      --config $c --matmul-n 8192 --gather-dtype bfloat16 --scg-out-rows $g \
      --output-dir profiles/native
  done
done

# The GMM shape sweep behind §9: 14x of arithmetic intensity, both dtypes.
for spec in "8192 128" "32768 256" "32768 32" "32768 8" "131072 128"; do
  set -- $spec
  uv run --frozen --group profile python tools/overlap_harness.py \
    --config gmm --gmm-rows $1 --gmm-groups $2 --output-dir profiles/native
  for dt in float32 bfloat16; do
    for c in gmm+gather-sparsecore gmm+gather-xla; do
      uv run --frozen --group profile python tools/overlap_harness.py \
        --config $c --gmm-rows $1 --gmm-groups $2 --gather-dtype $dt \
        --scg-out-rows 32768 --output-dir profiles/native
    done
  done
done

uv run --frozen --group profile python tools/overlap_harness.py \
  --summarize --output-dir profiles/native
```

To reproduce the fallback measurement in §4, force the gather shape and opt out
of the guard:

```bash
uv run --frozen --group profile python tools/profile_kernel.py \
  --implementation scg-tokamax-gather-reduce --scg-rows 4096 --scg-hidden 1024 \
  --allow-xla-fallback --run-label xla-fallback \
  --trace-chunk-size 1 --output-dir profiles/native
```
