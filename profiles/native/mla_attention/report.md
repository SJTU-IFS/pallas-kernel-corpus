# MLA Attention TPU v6e Analysis

Generated: 2026-08-04 (America/New_York), superseding the 2026-07-31 run
Device: one TPU v6e chip (`TPU v6 Lite`, 31.25 GiB available HBM)
Software: Python 3.12.13, JAX/JAXlib 0.10.2, libtpu 0.0.42.1
Protocol: bf16 inputs, JIT compile, 5 warmups, then 50 device-profiled
iterations each in its own trace chunk, then the wall-clock diagnostics and
correctness check. That order is load-bearing — see §7.

## 1. Scope

| Source | Upstream path | Migrated | Audited |
|---|---|---:|---:|
| Tokamax | `_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py` | 1 | 1 |
| vLLM tpu-inference | `mla/v1/kernel.py` | 1 | 1 |
| vLLM tpu-inference | `mla/v2/kernel.py` | 1 | 1 |
| sglang-jax | `mla/v2/kernel.py` | 1 | 1 |
| vLLM tpu-inference | `deepseek_v4/mla.py`, `deepseek_v4/mla_swa.py` | 2 | 2 |
| JAXBench | `benchmark/3p_MLA_Attention/optimized.py` | **0** | 3 |

**6 of the family's 9 audited launch points are migrated.** Four are profiled
below; the two DeepSeek-V4 kernels are migrated and correctness-validated but
deliberately unprofiled (§6). See §6 for the three that are not migrated.

MLA compresses the KV cache into a single latent stream shared by every query
head, plus a small rotary part. That sharing is what makes the cache small, and
it is why the contract looks unlike ordinary attention:

```text
ql_nope       [num_tokens, num_q_heads, lkv_dim]   latent query part
q_pe          [num_tokens, num_q_heads, r_dim]     rotary query part
new_kv_c      [num_tokens, lkv_dim]                latent KV to append
new_k_pe      [num_tokens, r_dim]                  rotary K to append
cache_kv      [total_num_pages, page_size // packing, packing, kv_dim]
kv_lens       i32[max_num_seqs]
page_indices  i32[max_num_seqs * pages_per_seq]    flattened
cu_q_lens     i32[max_num_seqs + 1]
distribution  i32[3]  decode / prefill / mixed, as in rpa_v3
->            [num_tokens, num_q_heads, lkv_dim]
```

Three contracts among the profiled four, none drop-in compatible:

| Contract | Sources | Required args |
|---|---|---:|
| `mla_ragged_paged_attention` | Tokamax, tpu-inference v1 | 9 |
| `mla_ragged_paged_attention_cu_kv` | sglang-jax v2 | 10, adding `cu_kv_lens` |
| `mla_ragged_paged_attention_head_major` | tpu-inference v2 | 9, `ql_nope` transposed |

The `cu_kv_lens` divergence is the same pattern seen between `rpa_v2` and
`rpa_v3`. The head-major one is nastier: tpu-inference **v2** takes the same
nine argument *names* as v1, but `ql_nope` is `[num_q_heads, num_tokens,
lkv_dim]` and the output comes back head-major too, while `q_pe` stays
token-major. Nothing rejects a token-major call when `num_tokens ==
num_q_heads` — it returns a wrong answer at cosine 0.534.

### Two shape facts that cost real debugging time

- **`kv_dim = align_to(lkv_dim, 128) + align_to(r_dim, 128)`.** The cache holds
  the latent and rotary parts *concatenated*, each padded to a 128-lane
  multiple. At DeepSeek-V3 dims (`lkv=512, r=64`) that is `512 + 128 = 640` —
  not 512, and not 576.
- **`cache_kv` is donated.** The entry points are decorated
  `@jax.jit(donate_argnames=("cache_kv"))`, so the buffer is consumed and there
  is no flag to disable it. Reusing one across calls raises
  `Array has been deleted`. The profiler pre-builds a fresh cache per call
  outside the timed region; `result.json` records `donated_argnum`. This is the
  only family that donates, which made it the obvious suspect when profiling
  broke — wrongly, as §7 shows.

### No native shape

No upstream benchmark defines one. The profiled shape is a **declared
validation shape** (`native_source_shape: false`) at DeepSeek-V3 MLA
dimensions: `num_seqs=8, q_len=64, kv_len=1024, num_q_heads=16, lkv_dim=512,
r_dim=64, page_size=16`.

## 2. Baseline — correctness only, deliberately no speed denominator

vLLM tpu-inference v1 ships a pure-JAX `ref_mla_ragged_paged_attention`
(verified Pallas-free) with the matching nine-argument signature. The corpus
uses it as the family's correctness reference, including as a
**cross-repository** check for Tokamax and sglang-jax, and as a cross-*version*
check for its own repository's v2 — a stronger independence property than a
hand-written baseline, since the implementations were written by different
teams.

**It cannot serve as a timing denominator.** It calls `dynamic_validate_inputs`,
which tests traced values with Python `if`, so `jax.jit` raises
`TracerBoolConversionError`. Rather than invent a denominator, this family
reports speedups *between implementations*, and every `result.json` records
`reference_is_jittable: false`. A test fails if it ever becomes jittable, so
that decision gets revisited rather than silently inherited.

## 3. Correctness

All PASS at the 0.9999 cosine threshold, against tpu-inference v1's pure-JAX
reference.

| Implementation | Reference | Cosine | Max abs | Status |
|---|---|---:|---:|---|
| Tokamax | **cross-repo** | 0.9999933243 | 3.906e-03 | PASS |
| tpu-inference v2 | **cross-repo** | 0.9999972582 | 1.953e-03 | PASS |
| sglang-jax v2 | **cross-repo** | 0.9999972582 | 1.953e-03 | PASS |
| tpu-inference v1 | own | 0.9999972582 | 1.953e-03 | PASS |

Also verified at a smaller shape (`num_seqs=4, q_len=32, kv_len=256`).

## 4. Timing

All four profiled with matched blocks — `num_kv_pages_per_block=2`,
`num_queries_per_block=16`, the smallest configuration all four accept — so
the comparison isolates the kernels rather than their tiling.

| Implementation | Median (ms) | Std (ms) | TFLOP/s | MXU @ 918 | vs fastest |
|---|---:|---:|---:|---:|---:|
| **tpu-inference v2** | **1.119854** | 0.000215 | 16.3 | 1.78% | **1.000×** |
| Tokamax | 1.141174 | 0.000305 | 16.0 | 1.74% | 1.019× |
| sglang-jax v2 | 1.183758 | 0.000137 | 15.4 | 1.68% | 1.057× |
| tpu-inference v1 | 3.754144 | 0.006284 | 4.9 | 0.53% | 3.352× |

The three second-generation kernels land within 6% of each other, and
tpu-inference's own **v2 is 3.35× faster than its v1** at this shape — the
generational gap inside one repository is far larger than any gap between
repositories. That answers the open question from the 2026-07-31 run, which
could only observe that v1 trailed Tokamax and sglang-jax without knowing
whether the cause was the repository or the generation. It was the generation.

The low MXU figures are expected and not a defect: at `q_len=64` against
`kv_len=1024` this is a decode-shaped workload, and the matched block sizes are
the *smallest* the four agree on, chosen for comparability rather than speed.
Tuning each kernel to its own preferred blocks would raise all four.

These medians reproduce the 2026-07-31 numbers to within 0.05% for all three
implementations recorded then (Tokamax 1.140662 → 1.141174, sglang-jax
1.183468 → 1.183758, tpu-inference v1 3.753764 → 3.754144), which is the
evidence that §7's fix changed how the trace is *captured* and nothing about
what is measured.

## 5. FLOP accounting

```text
logical = 2 · tokens · kv_len · num_q_heads · (lkv_dim + r_dim)   [QK]
        + 2 · tokens · kv_len · num_q_heads · lkv_dim             [PV]
        = 18,253,611,008
```

QK contracts over the latent *and* rotary dims, PV over the latent only. Every
query head attends the same shared latent KV stream, so unlike ordinary MHA
there is no per-head K/V to count — that sharing is precisely what MLA buys.

## 6. What is not migrated, and why

### JAXBench `3p_MLA_Attention` — no new Pallas kernel

Its three audited launch points are the **flash-attention kernel already in the
corpus** at `kernels/attention/flash_attention/jaxbench_optimized.py`. Comparing
the two files: 24 of 27 shared definitions are AST-identical to
`1p_Flash_Attention`, differing only in `benchmark`, `create_inputs` and
`workload`, plus one added `_apply_rope`.

Its MLA-ness lives entirely in the surrounding JAX — the LoRA down/up
projections and RoPE — not in the Pallas launch. Migrating it as an "MLA
kernel" would double-count a kernel the corpus already has, which is exactly
the file-count-versus-kernel-count trap. It is recorded as audited-but-not-
migrated with this reason.

### The DeepSeek-V4 variants — migrated, deliberately unprofiled

`deepseek_v4/mla.py` and `mla_swa.py` are both migrated and correctness-
validated against ported upstream references, but neither is profiled, and not
for want of trying: they are the two halves of a **two-pass** attention. The
sliding-window kernel runs first and its `L` and `m` outputs are the carry-in
the sparse-top-k kernel consumes. Timing either alone would measure something
that never runs alone, and upstream defines no benchmark shape for the pair.
They also implement different contracts (`mla_sliding_window` and
`mla_sparse_topk`) from the four profiled here, so they would not belong in
§4's table even if they were timed.

JAXBench is therefore the only MLA source with nothing migrated from it.
`mla/v2/kernel.py`, listed here as pending in the 2026-07-31 report, has since
been migrated — four repo-local imports resolved, see `inventory.json` for what
was inlined and why — which is what made §4's v1-versus-v2 comparison possible.

## 7. Why this family stopped being profilable, and what changed

Between 2026-07-31 and 2026-08-04 profiling every implementation in this
family — including the three that already had a `result.json` — failed with

```text
RuntimeError: trace chunk 0 only found 3 jit device events for 10 iterations
```

`flash_attention` and `ragged_paged_attention` profiled fine in the same
session, and this is the only family that donates an argument, so the donated
`cache_kv` was the natural suspect. **It is not involved.** With the
donated-spare path completely untouched, trace chunks of 1 through 5
iterations return exactly one `jit_*` module event per iteration, and a
10-iteration chunk returns all ten once the cap below is raised. Two unrelated
tracer limits were responsible, and this kernel is dense enough to hit both.

### 7a. The trace-viewer event cap

xprof's converter — the step that writes the `perfetto_trace.json.gz` the
profiler reads — keeps at most **1,000,000** events carrying a duration and
drops the rest. `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS` overrides the limit.
`--xla_enable_custom_call_region_trace=true` and
`--xla_xprof_register_llo_debug_info=true` make every VLIW bundle inside a
Pallas kernel one of those events. Measured on the validation host:

| Trace | Iterations | Duration events |
|---|---:|---:|
| MLA v1, first trace in the process | 1 | 461,870 |
| MLA v1, first trace in the process | 10 | 4,211,980 |
| MLA v1, a later trace in the same process | 1 | 151,749 |
| MLA v1, a later trace in the same process | 5 | 577,996 |
| RPA v1 at its native shape, first trace | 10 | 308,796 |
| MLA v1, both tracing flags removed | 10 | 77,134 |

The instruction-class events (`Vector Load`, `Scalar ALU`, `Vector Store`, …)
are emitted only during the **first** profiler session of a process, which is
why a later trace costs a third as much and why the failure always named
chunk 0.

At the old default of ten iterations per chunk, chunk 0 needs about 4.2M events
and gets 1.0M. The drop is not selective, so seven of the ten `jit_*` XLA
module events — the only events the profiler measures — go out with the
instruction-level noise. Raising the cap to 20M returns all ten, which is what
identifies the cap as the cause rather than a symptom.

### 7b. One profiler session per process

Fixing the cap exposed a second, unrelated failure: chunk 0 captured device
data and every chunk after it captured **none** — about 1,600 host events and a
300 KB XPlane against the usual 18 MB — with the second `jax.profiler.trace`
blocking for sixty seconds before returning empty.

Bisecting what `tools/profile_kernel.py` did before its trace loop, one piece
at a time, on Tokamax MLA at one iteration per chunk:

| Work done before the trace loop | Chunks that captured |
|---|---|
| nothing | 14 of 14 |
| the two cost/HLO compilations | 3 of 3 |
| the correctness check | 3 of 3 |
| 50 wall-clock iterations | 3 of 3 |
| cost analysis + correctness + wall loop | **1 of 8** |
| **all of the above together** | **1 of 14** |
| all of it, tracing flags removed | 14 of 14 |
| all of it, moved *after* the trace loop | 14 of 14 |

No single step does it; the accumulated work does, and only with
instruction-level tracing on. Once wedged it stays wedged — the first failed
session blocks for the full sixty seconds, every one after it returns empty in
600 ms.

**This is not a change in the profiler.** The fifth row is the prelude exactly
as it stood for the 2026-07-31 run, before `count_pallas_launches` added its
compilation, and it wedges the tracer today. So whatever changed between then
and now is in the host or the TPU runtime, not in `tools/profile_kernel.py`;
`pyproject.toml` and `uv.lock` are byte-identical across the two runs. What
changed has not been identified, and the reordering below does not depend on
knowing.

`tools/profile_kernel.py` now runs the traced measurement **first** — build,
spare copies, warmups, trace loop — and does the wall-clock diagnostic, the
Pallas-launch guard, the correctness check and the cost analyses afterwards. It
also defaults `--trace-chunk-size` to 1 (what every recorded recipe in this
corpus already passed), records the per-chunk event count and the cap in
`result.json`, and, when a chunk does come up short, says which of the two
limits it hit instead of only reporting the shortfall.

### 7c. What the tracing flags cost

Measured on Tokamax MLA, 14 traced iterations each: **1.1414 ms** with the two
flags, **0.8965 ms** without them. Instruction-level tracing adds about **27%**
to what this kernel appears to cost.

The flags stay on. Every other family's recorded result was measured with them,
and turning them off for this family alone would make its numbers incomparable
with the rest of the corpus. But §4's medians are kernel time *under
instruction-level tracing*, not the kernel's standalone cost, and a
corpus-wide decision to drop the flags would move every recorded number.

## 8. Trace and artifact locations

```text
profiles/native/mla_attention/<run>/trace/
```

Compact local artifacts are the four `result.json` files beside this report.

## Reproduce

```bash
uv sync --frozen --group profile
export LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"

for impl in mla-tpu-inference-v2 mla-tokamax mla-tpu-inference mla-sglang-jax; do
  uv run --frozen --group profile python tools/profile_kernel.py \
    --implementation $impl --output-dir profiles/native
done

uv run --frozen --with pytest python -m pytest \
  tests/test_mla_attention_tpu.py -q
```

`--trace-chunk-size 1` is no longer passed because it is now the default; a
larger chunk still trips §7a on this family.
