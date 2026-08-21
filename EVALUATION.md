# Rigor Evaluation

An independent audit of whether this corpus's correctness claims are supported.

**Method.** Seven auditors each re-derived one dimension from primary sources —
the six pinned upstream trees, not this repository's own assertions. Every
finding that asserted a problem was then handed to a skeptic instructed to
refute it, and told explicitly that a limitation this repository already
discloses is not a defect. 47 problem-asserting findings were raised; 33 (70%)
were refuted and discarded. Only survivors appear below.

**Conditions.** Run entirely without a TPU, on CPU JAX 0.10.2 — the same version
this corpus pins. Nothing in the repository was modified: all regeneration ran
in temporary directories, and `git status --porcelain` was empty before and
after. Section 5 states plainly what that leaves unverified.

**Independently re-checked before publication:** the scale-invariance of
`cosine` (a 10^6-scaled output still passes the 0.9999 bar; 2% noise fails), the
file counts (113 optimized / 49 with a copyright notice / 12 with a smoke
runner), the presence of `xla_quantized_matmul` upstream in two places and
already carried in this repo at `sglang_jax_optimized.py:853`, and the clean
repository state.

---

**Repository:** `/Users/shizhan/Desktop/Research/tpu-pallas-agent/pallas-kernel-corpus` @ `3205137` (branch `prepare-remaining-launch-points`)
**Repository state after audit:** `git status --porcelain` empty. Nothing was modified. All regeneration ran in `mktemp -d` directories.

---

## 1. The answer

**Yes — the corpus's structural and provenance claims are substantially supported, and its correctness claims are supported but weaker than the README implies in two specific, identifiable ways.**

This is an unusually well-evidenced repository. Across seven audit dimensions and 47 problem-asserting findings, independent skeptics refuted 33 (70%). The refutations were not charitable readings — most were empirical, and several showed the auditor had gotten the direction of an error backwards. The claims that reproduce, reproduce *exactly*:

- The launch-point audit regenerates **byte-identically** (`181` launch points / `156` TPU-compatible / `25` GPU / `151` source files / `40` families), and so do `inventory.json` and `REMAINING.md`.
- The accounting closes at every level: `136 + 5 + 11 + 4 + 0 = 156`; per-family `audited_tpu` sums to 156; per-implementation `migrated_launch_points` sums to 136 across 113 implementations. No launch point is double-claimed — verified by an independent re-implementation of the assignment plus 300 random permutations of implementation order, all producing an identical matching.
- **~110 of 113 kernel files regenerate byte-identically** from their recorded upstream commits: all 43 PallasBench kernels + 19 baselines, all 34 files from the 11 recent flatten scripts, and 48 of 50 from the 14 legacy scripts.
- The CPU suite reproduces to the test: I measured **44 passed, 431 skipped of 475**, matching README:244 exactly.
- Licensing metadata is consistent: all 113 `SOURCE` commits match the six pinned trees, all recorded paths exist upstream.
- References are genuinely upstream's in the large majority of cases — all 19 PallasBench families' `jax_*` functions and `generate_inputs` are **byte-identical** to `pallasbench/baselines/jax_baseline.py`, as are MaxText's `reference_mqa/mha/gqa`, Tokamax's cross-entropy reference, sglang-jax's `naive_recurrent_kda`, and all 13 in-kernel `ref_*` functions the tests compare against.

The two real weaknesses: **(a)** the suite's dominant comparator in its largest attention families is blind to output magnitude, and **(b)** one family's reference was written by the corpus, shares the kernel-under-test's own quantizer, and diverges from upstream's shipped reference by essentially the test's entire tolerance budget. Neither shows a kernel to be wrong. Both mean the safety net is thinner than the README's "Verification status" section leads a reader to believe.

Everything else that survived is bookkeeping and provenance-labelling error — real, cheap to fix, and costly mainly because provenance precision is this corpus's entire value proposition.

---

## 2. What I re-verified myself

I did not take the auditors or the skeptics on trust. I independently reproduced every finding below marked ✔:

| Check | Command / result |
|---|---|
| Repo + pinned trees clean | 6 pinned commits match `inventory.json`; only untracked `__pycache__` in accelerator-agents |
| RPA v3 flatten drift ✔ | Regenerated → **16-byte diff**; control `sglang_jax_v2` under identical staging → **BYTE-IDENTICAL** |
| Copyright header set ✔ | AST-resolved every `SOURCE` into pinned trees → **exactly 6** files with no notice whose upstream constituents all carry one |
| cosine blindness ✔ | 8 files define `cosine`; 62 call sites; **35 of 41** cosine-using test functions carry no magnitude assertion |
| quantized_matmul ✔ | Measured cosine(corpus baseline, upstream reference) = **0.9999014** against a `> 0.9999` bar |
| gdn shadowing ✔ | `inspect.getsourcelines` → resolves to **line 493** (Tokamax), shadowing line 94 (tpu-inference) |
| Counts ✔ | 113 optimized files / 49 with copyright / 12 smoke runners / 147 `result.json` of which **96** carry `pallas_launches` / **11** files declare >1 launch point |
| CPU suite ✔ | `44 passed, 431 skipped`; the 44 = 15 flatten-tool + 10 spmm + 8 harness + 6 ledger + 5 ragged_mqa |
| Sparsecore rebuttal ✔ | Confirmed the *rejected* finding was inverted (see §4) |

---

## 3. Findings that survived, ranked by cost

### Tier 1 — Weakens the evidence that a kernel is correct

**F1. `cosine(actual, expected) > 0.9999` is scale-invariant, and it is the only comparison in 35 of 41 test functions across eight attention/matmul families.** *(major — claim true as written, but far weaker than it reads)*

The helper is byte-identical in all 8 files: `a @ e / (norm(a)*norm(e) + 1e-12)`. Measured directly:

```
kernel output x0.5   -> cosine = 0.9999999  passes: True
kernel output x2     -> cosine = 0.9999999  passes: True
kernel output x1e+06 -> cosine = 1.0000001  passes: True
2% random perturbation -> cosine = 0.999805 passes: False
```

The bar is *tight* on direction and *exactly zero* on magnitude. Distribution of magnitude-blind test functions: `test_ragged_paged_attention_tpu.py` 10/10, `test_gated_delta_net_tpu.py` 9/10, `test_flash_attention_backward_tpu.py` 7/8, `test_mla_attention_tpu.py` 3/3, `test_quantized_matmul_tpu.py` 2/2, `test_gated_linear_attention_tpu.py` 2/3, splash forward 2/3. In those tests the only companion assertion is `assert actual.shape == expected.shape`. A kernel with a wrong dequantization scalar, fp8 scale factor, or softmax normalization constant — a plausible failure mode for exactly these kernels — passes green.

Corroborated by mutation: with every float Pallas output multiplied by 2.0 under interpret mode, `tests/test_quantized_matmul_tpu.py` gives `17 passed` (also at ×1e6), and all four `test_forward_matches_reference` splash cases pass.

This is not disclosed. `README.md:172` states the sweep covers "every numeric comparison the suite makes" and Verification status reports "311 comparisons, all clean." The scale-invariance appears only as a parenthetical inside `tools/assertion_strength.py`'s docstring, framed as tool methodology. **Fix is one line per test** — the corpus already uses an RMS-relative bar in three places, so the authors knew the check was needed and did not generalize it.

*Fair mitigation:* 17 of 25 test files use magnitude-sensitive comparators exclusively; the 0.9999 bar is genuinely tight on direction; and gross bugs (indexing, masking, ragged bounds) perturb direction and are caught.

---

**F2. `kernels/quantization/quantized_matmul/baseline.py` is corpus-written although three upstream references exist — one of which the corpus already carries in the same directory — and the stated reason names the wrong function.** *(major — the claim "references are upstream's own wherever upstream ships one" is false here)*

Upstream references, all pure JAX, all confirmed present in the pinned trees:
- `pinned/tpu-inference/tpu_inference/layers/common/linear.py:29` `xla_quantized_matmul` — docstring: *"Reference (pure JAX) implementation of the quantized matmul kernel below"*
- `pinned/tpu-inference/tests/kernels/quantized_matmul_kernel_test.py:21` `reference_block_quantized_matmul` — *"Pure JAX reference for Block-wise Quantized Matmul"*
- `pinned/sglang-jax/.../quantized_matmul_kernels/util.py:68` `xla_quantized_matmul` — needs no signature adapter

**The third is already inside the corpus**, carried verbatim by the flattener at `kernels/quantization/quantized_matmul/sglang_jax_optimized.py:853`. The corpus wrote its own reference instead, and `baseline.py:29-31` justifies this by disqualifying only `xla_quantized_matmul_local` (correctly — that one dispatches into `blockwise_kernel`), never mentioning the three valid ones. Read as written, it says no valid upstream reference exists. It says so three times over wrong.

The substitution is not free. `baseline.py:62-74` `quantize_per_token` is copied from the kernel's own helper (its docstring says *"Mirrors the kernels' `quantize_array`"*) and does a **truncating** cast `(x * scale_inv).astype(x_q_dtype)`, where every upstream reference rounds via `quantize_block`'s `jnp.round(data / scale)`. Measured on CPU (n_batch=64, n_in=256, n_out=128, int8):

```
activation quantization ON : cosine(corpus baseline, upstream reference) = 0.9999014   max|d| = 1.044 (peak 71.4)
activation quantization OFF: cosine = 1.0000000, max|d| = 0.0
```

`tests/test_quantized_matmul_tpu.py:82,107` assert `cosine > 0.9999`. The corpus reference and upstream's differ by **the test's entire margin**, and the divergence is *entirely* the quantizer — which the reference copies from the kernel it is validating. A defect in `quantize_array` is invisible to this test. That is the self-fulfilling-reference failure the README argues against elsewhere ("the reference is part of the contract", README:977).

Not disclosed anywhere: not in README's quantized-matmul paragraph, not in Verification status, not in `inventory.json`'s four rows for the family.

---

### Tier 2 — Provenance and licensing accuracy

**F3. The copyright-attribution claim is false on two of its three parts, and `THIRD_PARTY_NOTICES.md` routes readers to headers that were deleted.** *(major — claim false)*

`README.md:1265-1266`: *"Where an upstream file carried a copyright header, that header is retained verbatim — 45 of the 108 do, which is exactly the set whose originals had one; the flattening tools preserve leading comments and never add a notice that upstream did not write."*

- **Counts stale:** 49 of 113, not 45 of 108 (the HEAD commit added five files without revising the sentence).
- **"Exactly the set" is false.** Six files carry *no* notice while 100% of their upstream constituents carry the full Apache-2.0 header — independently reproduced by AST-resolving every `SOURCE` into the pinned trees:

| Corpus file | Upstream constituents with a header |
|---|---|
| `state_space/gated_delta_net/tpu_inference_v1_optimized.py` | 4/4 |
| `state_space/gated_delta_net/tpu_inference_v2_optimized.py` | 4/4 |
| `state_space/gated_delta_net/tpu_inference_v3_optimized.py` | 7/7 |
| `state_space/gated_delta_net/tokamax_v3_optimized.py` | 7/7 |
| `attention/ragged_paged_attention/tpu_inference_batched_optimized.py` | 9/9 |
| `memory/kv_cache_update/tpu_inference_dsv4_optimized.py` | 6/6 |

- **"The flattening tools preserve leading comments" is false by construction.** `tools/flatten_gdn.py:614` `strip_module_preamble` — *"Remove the license block, module docstring and every import statement"* — is called by `flatten_gdn.py`, `flatten_batched_rpa.py`, `flatten_dsv4_compressor.py` and `flatten_gla.py`, precisely the tools behind those six files. Other flatten tools *do* carry a `license_header` through, so the behaviour is inconsistent, not policy.
- **Aggravating:** `THIRD_PARTY_NOTICES.md:4-5` tells readers to consult *"each copied file's retained header for the controlling license and notices."* For these six there is no retained header.

The third sub-claim ("never add a notice that upstream did not write") holds.

*Mitigation:* `LICENSE` is Apache-2.0, `THIRD_PARTY_NOTICES.md` names all six upstreams with commits, and every file's `SOURCE` records repository/commit/path — so attribution is not absent from the work as a whole.

---

**F4. `kernels/state_space/gated_delta_net/baseline.py` defines `_recurrent_gated_delta_rule_step` twice; the contract-1 entry point silently executes the wrong repository's copy.** *(minor — claim false, numerically inert today)*

Both sections are verbatim from their pinned commits. But `grep -n "^def _recurrent_gated_delta_rule_step"` returns lines **94** (tpu-inference) and **493** (Tokamax), and `inspect.getsourcelines` on the loaded module resolves to **line 493** — Tokamax's copy — with body marker `batch_size, num_heads, _, d_k = query.shape`. `ragged_gated_delta_rule` (contract 1) calls the global at line 294, so the tpu-inference copy is dead code.

The file's marker at line 323 reads *"Everything above this line is upstream's file verbatim"* while the import block above it carries four imports upstream's file does not have (`dataclasses`, `enum`, `functools`, `from jax import lax` — they come from the second reference). The sibling `gated_linear_attention/baseline.py:33` words the same situation correctly: *"both files copied verbatim; only their imports were merged."*

Zero numerical consequence at the pinned commits (the bodies are alpha-equivalent; bitwise-identical outputs either way). The cost is latent: `tools/flatten_gdn.py`'s own header records that this pair shares 24 top-level names of which only ~42% are AST-identical, so a re-pin could silently substitute a diverged helper. `flatten_gdn.py` already applies renames to prevent exactly this shadowing when concatenating the v1 modules — a guard `baseline.py` never receives because no tool assembles it end to end.

---

**F5. `kernels/sampling/speculative_decoding/baseline.py:190` labels a function "unmodified" after dropping upstream's `jax.jit` decorator and all its comments.** *(minor — claim false, behaviourally inert)*

Upstream `eagle_util.py:181-183` carries `@functools.partial(jax.jit, static_argnames=["num_verify_tokens", "batch_size", "speculative_num_steps"])`. The corpus copy has no decorator, drops all five inline comments — including the only documentation of the expected `score_list`/`token_list`/`parents_list` shapes — and adds a corpus docstring, under the unqualified label *"Upstream's own, from eagle_util.py -- unmodified."*

The corpus's own convention elsewhere is precise about this (*"unmodified apart from this header and the same jaxtyping substitution"*), and `README.md:1029-1041` treats an identical lost `@jax.jit(static_argnames=[...])` as a defect worth a regression test. Compounding: the function is a hardcoded string literal (`BASELINE_BODY`, `tools/flatten_speculative.py:236`), not extracted from the pinned tree — so unlike the adjacent test vectors, nothing enforces that it stays in step with upstream.

---

**F6. `kernels/attention/ragged_paged_attention/sglang_jax_optimized.py` is stale relative to the tool that claims to produce it.** *(minor — claim false, functionally inert)*

I reproduced this in one command. Regenerating from the pinned commit yields a 16-byte diff:

```
@@ -36,6 +36,8 @@
     "target": "tpu",
     "contract": "rpa_v3",
 }
+
+import logging
```

Cause: `tools/flatten_rpa_v3.py:274` prepends `import logging` whenever any chunk contains `logging.getLogger`, and both `tuned_block_sizes_v3.py:33` and `ragged_paged_attention_v3.py:49` do. **No staging arrangement can suppress it.** The control settles it: `sglang_jax_v2` regenerates **BYTE-IDENTICAL** under the identical staging, and its committed file *does* carry the prepend at line 39. All other 270,346 bytes match.

Inert (the flattened chunks carry their own `import logging`), but undisclosed — README, the inventory note and the file docstring are silent, whereas the corpus *does* document the equivalent MLA v2 byte difference at README:1040.

---

### Tier 3 — Bookkeeping claims that are literally false but cost little

All independently reproduced. Each is a falsifiable asserted number in a repository whose stated discipline is that its counts are re-derived, not asserted.

| # | Claim | Reality |
|---|---|---|
| **F7** | `README:172` — the sweep intercepts "every numeric comparison the suite makes" | `tools/assertion_strength.py:137` wraps only `("cosine", "_close")` + the two numpy asserts. `tests/test_ragged_mqa_attention.py` has **zero** calls to any of them — all six of its correctness comparisons go through a local `compare()` at line 74. Also unseen: `test_sparsecore_ragged_gather_tpu.py:166,252`, each the sole comparison in its test. *(Split verdict: the claims-dimension skeptic upheld this at minor; the tests-dimension skeptic rejected it as informational on the grounds that the un-swept comparisons are demonstrably non-vacuous — which is true. Reported at minor because the claim is unambiguously false.)* |
| **F8** | `README:794` — "every `result.json` records `pallas_launches`" | **96 of 147** do. Excluding baselines and `xla_ragged_dot`, **34** non-reference implementation profiles lack it. True of the *tool* (`profile_kernel.py:523` writes it unconditionally); false of the stored corpus. The 51 misses also lack the four other fields added at the same time, i.e. they are older-schema artifacts. |
| **F9** | `tests/test_fused_moe_tpu.py:147-150` peak-scales `ATOL` | Peak-scaling *tightens* four of six comparisons but **loosens two past upstream's own shipped bar**: reference peaks 62.25 and 11.06 give atol 12.45 and 2.21 versus upstream's flat `atol=2e-1` (62× and 11× looser), and in both the absolute term alone exceeds the mean output magnitude. Upstream passes at the tighter bar. The in-file comment honestly explains the change as a vacuity fix but never says it went looser than upstream; `inventory.json` says "at its own atol=rtol=2e-1". |
| **F10** | `README:663` — `batched_rpa` is "the only one carrying **two launch points in one file**" | **11 files** declare >1 launch point (kda 4; layout_transpose, gdn v1, simple_gla 3 each; six others at 2). README's own migration table lists ten "forward (2 launches)" rows — the page contradicts itself. |
| **F11** | `README:108-112` — "Backward passes are covered in **three families**" then lists four; "the only unmigrated ones are the v1 `tgmm`s" | Four families (splash, flash, grouped matmul, cross entropy). Of 18 TPU-compatible backward launch points: 12 migrated, 2 qwix, **4 excluded** (JAXBench `4p_Sparse_Attention` and `3p_MLA_Attention` bwd_dq/dkv). "0 open" is true; "only unmigrated" is not. *Mitigation:* every `*_bwd*` definition in those files is AST-identical to a migrated counterpart, so no backward kernel code is missing. |
| **F12** | `README:36` — "Eleven of the 108 also carry a `python file.py` smoke runner" | **12 of 113.** Both numbers wrong; "eleven" was already wrong at `HEAD~1` when there were 108 files. |
| **F13** | `README:868` / `flatten_gdn.py:20` — gdn v3 pair shares "24 definition names ... only ~42% AST-identical" | "Seven module names" holds exactly. **24 reproduces under no normalization** (21 functions, or 33 functions+classes). The paired figures cannot both come from one measurement: 10-identical requires functions-only-no-docstrings (10/21 = 48%); ~42% requires functions+classes-with-docstrings (14/33). Range across methods: 33%–52%. The qualitative conclusion (diverged) holds under all of them. Notable: 42% is reused as a cross-family migration threshold. |
| **F14** | `family_evidence` taxonomy | 175 of 181 rows are `path-grouped`, whose documented meaning is *"its contract has not been verified yet"* — including **130 of the 136 migrated** launch points, all of which carry a named contract, a recorded baseline and `correctness: PASS` in the same file. Only 3 of 106 rules are `source-read`. The error runs entirely in the **conservative** direction (the ledger understates its own verification) and nothing references the field. *Informational.* |

---

## 4. What the skeptics knocked down — and the most instructive false alarm

**33 of 47 problem-asserting findings (70%) were refuted.** Two thirds of those failed one of three ways: (a) the finding tested a claim the repository does not make; (b) the "gap" was disclosed in the README, the inventory notes, or the file's own docstring; (c) the finding was an artifact of the auditor's CPU-only instrumentation.

**Most instructive false alarm — a `major` finding whose error direction was exactly inverted.**

An auditor reported that `kernels/memory/sparsecore_ragged_gather/baseline.py`'s `ragged_gather_reduce` "multiplies and sums in the input dtype" where upstream's `_fallback_implementation` casts to `jnp.float32`, and that this contributed 0.45% relative bf16 error to the "one bf16 ulp" gap the README attributes to the kernels — i.e. that the corpus had weakened its own oracle.

It is textually true that the corpus reference omits both `.astype(jnp.float32)` and the trailing `.astype(x.dtype)`. But `topk_weights` is created as `float32` in both `create_inputs` and the test helper, so JAX promotes bf16 × f32 → f32. I measured it:

```
x dtype: bfloat16   weights dtype: float32
corpus reference out dtype: float32    upstream fallback out dtype: bfloat16
upstream == round(corpus -> bf16): True
max|corpus - upstream| = 0.01554
```

The corpus reference accumulates in fp32 exactly as upstream does; the *entire* discrepancy is upstream's final downcast. **The corpus reference is strictly the more precise oracle**, and substituting upstream's would have let a kernel be wrong by a bf16 ulp for free. A `major` finding alleging a weakened test in fact identified a place where the corpus is stricter than upstream.

**One refutation was itself wrong, which is worth recording.** A skeptic dismissed the flatten-drift finding partly on the ground that "all six `sglang_jax*_optimized.py` files regenerate byte-identically." They regenerated six MoE-family files (`gated_mlp`, `fused_moe`, `grouped_matmul`) and never ran `flatten_rpa_v3.py` against the `ragged_paged_attention` file at issue. I reproduced the 16-byte drift in one command (F6). Skeptic verdicts here are strong but not infallible; the ones that measured the specific artifact held up, the ones that generalized from a neighbouring artifact did not.

Other notable refutations, all of which strengthen the corpus's standing: an auditor declared upstream's `embedding_lookup` exclusion unverifiable without a TPU, and the skeptic verified it on CPU via AOT lowering under an abstract mesh (`ValueError: Cannot do int indexing on TPU`, traced to `jax/_src/pallas/mosaic/lowering.py:2200`); a `PALLAS_INTERPRET` "silent green suite" finding required monkeypatching `jax.devices()` to lie about hardware and produced a visibly **red** run (`10 failed`); and four separate findings alleging corpus-written references (`flash_attention`, `grouped_matmul`, `kv_cache_update`, `topk_routing`) were refuted by measurement showing the corpus versions bitwise-identical or mathematically equivalent to upstream's, on upstream's own input distributions.

---

## 5. What could not be checked without a TPU

This is the honest boundary, and it is large.

**Executed on CPU: 44 of 475 tests.** Verified breakdown of the 44:

| File | Passes | Executes a Pallas kernel? |
|---|---|---|
| `test_flatten_tools.py` | 15 | No — helper unit tests |
| `test_structured_sparse_matmul.py` | 10 | **Yes** (interpret mode) |
| `test_overlap_harness.py` | 8 | No — pure Python |
| `test_corpus.py` | 6 | No — ledger/invariant checks |
| `test_ragged_mqa_attention.py` | 5 | **Yes** (interpret mode) |

So **15 tests execute a kernel, covering 2 implementations — both "prepared" (`migrated_launch_points = 0`)**. In the shipped suite, **0 of the 136 migrated launch points are exercised on this machine.** 19 test modules are hard-gated on `"TPU" in device.device_kind`.

The one CPU-executable path to a *migrated* kernel is the documented CLI, which I ran:

```
$ python kernels/attention/flash_attention/pallasbench_optimized.py --interpret
{"implementation": "pallasbench", "contract": "dense_2d", "shape": [128, 64],
 "compile_and_run_ms": 79.58, "max_abs_error": 1.19e-07}
```

Five kernel files ship such a CLI. **49 of 113 optimized files (43%)** use `make_async_copy`, semaphores, `emit_pipeline`, `core_map` or `ANY` memory space — structurally unreachable by the interpreter on any host.

**Consequently unverified, and it must not be read as either a pass or a failure:**

1. **Every TPU-measured number in README.** `433 passed, 1 skipped`; the assertion sweep's `274 → 301 → 311`; the multi_head_attention bf16 measurement; every cosine figure (0.534 / 0.99999 / 0.72 / 0.983 / 0.097); `~1634 GB/s`; the fused_moe `~0.6% relative error (max ratio 0.0057)`. Most cross-check against a second corpus artifact (test-file comments, `profiles/*/result.json` with jax version and device recorded, committed XPlane traces). One figure — *"9 elements of 131072, max difference 0.0019"* at README:927 — appears nowhere else and is uncorroborated.
2. **`tools/assertion_strength.py`'s full report.** It runs on CPU but reaches only **8** of 311 comparisons. The vacuity criterion itself I confirmed sound by probe (deliberately degenerate `assert_allclose` and `_close` cases both flagged VACUOUS; a flat-reference cosine flagged WEAK; the documented ml_dtypes bfloat16 fix works).
3. **`tools/launch_coverage.py`.** Its static half (implementation enumeration, `launch_sites()`) works on CPU; its runtime half needs a TPU. It is the corpus's primary defence against a launch point that never fires, and README discloses its last run predates three changes.
4. **Whether the 136 migrated kernels are numerically correct on hardware.** This audit verified *provenance, regenerability, accounting, reference fidelity and assertion strength*. It did **not** and could not verify that the kernels compute the right answers on a TPU. F1 and F2 concern how strong the evidence for that would be if the suite were run — not evidence that any kernel is wrong.

---

## 6. Credit where the repository's own disclosures pre-empted findings

This is not a marginal point — it is the single biggest reason two thirds of findings collapsed, and it reflects genuinely good practice.

- **The "Verification status" section is accurate to the digit.** `README:244` states `44 passed, 431 skipped` and names exactly what the CPU suite covers: *"the ledger checks, the flatten-tool tests, and the two prepared families that interpret mode can reach."* I measured exactly that, including the correct claim that only two prepared families are CPU-reachable. It also states *"That machine is gone, so the state is recorded here rather than implied"* and *"Nothing here is asserted as green that was not observed to be green."*
- **Staleness is dated, not hidden.** Each audit tool row carries "Last run before those same three changes," naming the fused_moe tolerance fix, speculative_decoding and paged_attention.
- **The corpus documents its own past failures.** README §"Auditing the ledger" opens by saying `migrated_launch_points` and `correctness: PASS` "are only as good as the check behind them, and twice a launch point was counted while nothing exercised it," then names all three incidents (`tgmm_v2`; PallasBench flash attention passing only from a profiling run; the SparseCore gather-reduce kernels passing against their own XLA fallbacks) and the tool built to catch them. It also records finding and fixing a blind spot in its *own* vacuity tool — `np.issubdtype(dtype, np.number)` silently dropping ml_dtypes bfloat16, hiding 11 comparisons (274 → 301).
- **Prepared kernels are labelled honestly and machine-readably.** Three of five have no dedicated test — but README:272 says the two `collective_matmul` kernels *"have never been executed at all"*, the per-kernel table marks them *"Not checkable on any hardware this project had"*, `inventory.json` says causal_conv1d *"has had no correctness check at all yet"*, and `collective_matmul/baseline.py` opens *"**Neither of these has ever been executed.**"* with `SOURCE = {..., "validated": False, "requires_devices": 8}`. An auditor filed this as a finding; the disclosure defeated it.
- **The corpus-written-reference convention is documented and machine-checkable.** README:41-48 states baselines are corpus-written by default and names the exactly four `<source>_reference.py` files that carry upstream verbatim — `find kernels -name "*_reference.py"` returns precisely those four. An absent `repository`/`commit` in a baseline's `SOURCE` is the corpus's own marker for "corpus-written." This defeated four separate reference findings. **F2 stands only because it is the case where the corpus's stated reason for going its own way points at the wrong upstream function.**
- **Exclusions carry measured evidence, not assertions.** The `topk_routing` SparseCore exclusion records the actual experiment (rows=8, n=1024, k=8, all-negative keys returning most-negative elements); the `embedding_lookup` exclusion is rechecked by a live test rather than asserted.
- **Anti-vacuity discipline is real and enforced in-test.** A zero-output mutation across seven families flipped every reachable numeric comparison to failure (pallasbench 49→6 with all 43 correctness comparisons dying; structured_sparse 18→2; layout_transpose 36→21). Five test files hard-assert the criterion inline (e.g. `assert not np.allclose(np.zeros_like(expected), expected, ...), "a zeroed kernel would pass this"`).

---

## 7. Ordered actions

### No TPU required — do these first

1. **Add a magnitude bar beside every `cosine > 0.9999`** (F1). One line per test in the 35 affected functions: an RMS-relative or norm-ratio check, the pattern the corpus already uses in three places. This is the single highest-value change in the list and needs no hardware to write.
2. **Replace `quantized_matmul/baseline.py`'s per-channel reference with the upstream `xla_quantized_matmul` already sitting at `sglang_jax_optimized.py:853`** (F2), or at minimum make the reference quantize with `jnp.round` rather than truncation so it stops sharing the kernel's quantizer. Correct the justification text in `baseline.py:29-31` and `profiles/native/quantized_matmul/report.md:69-71` to name the three references that do exist.
3. **Fix the copyright claim and the stripping** (F3). Either restore the license headers in the six files (make `strip_module_preamble` preserve the leading comment block, as five other flatten tools already do) or rewrite `README.md:1265-1266` to say plainly that four flatten tools strip module preambles including license blocks — and amend `THIRD_PARTY_NOTICES.md:4-5`, which currently directs readers to headers that do not exist.
4. **Rename one of the two `_recurrent_gated_delta_rule_step` definitions** in `gated_delta_net/baseline.py` (F4) — the same guard `flatten_gdn.py` already applies to the v1 modules — and correct the line-323 marker to match the sibling file's accurate wording.
5. **Regenerate `ragged_paged_attention/sglang_jax_optimized.py`** from the pinned commit (F6). One command; removes a 16-byte discrepancy between the artifact and the tool that claims to produce it.
6. **Add a `PINNED_CHECKOUTS`-gated regeneration-and-diff test** beside `tests/test_corpus.py:187`. This is what would have caught F6 automatically. It is CI-skippable since the upstream trees are not vendored, and it is the structural fix for the whole regenerability claim.
7. **Correct the bookkeeping** (F7–F13): scope the `assertion_strength` sentence and add `"compare"` to its helpers tuple; qualify the `pallas_launches` sentence to "every result.json written since this change"; `12 of 113` smoke runners; drop the `batched_rpa` uniqueness superlative; "four families" for backward passes and "the only *unmigrated*" → "the only *deferred*"; record the normalization method behind the 42% figure (or restate it as approximate) since it is reused as a cross-family threshold; drop the stale second clause of `path-grouped`'s definition.
8. **Clamp the fused_moe tolerance** (F9) to `min(upstream_flat_atol, ATOL * peak)` so peak-scaling can only tighten, and correct the inventory note.
9. **Restore the `jax.jit` decorator** in `speculative_decoding/baseline.py`, or qualify the "unmodified" label (F5). Consider extracting `BASELINE_BODY` from the pinned tree rather than transcribing it, so it cannot drift.

### Needs a TPU

10. **Re-run the full suite and record the result**, closing the regression gap for the fused_moe fix, speculative_decoding, paged_attention, and the five prepared families (28 tests) added since the last green run.
11. **Re-run `tools/assertion_strength.py`** after action 1 — the current 311-comparison report predates three changes, and the new magnitude bars will change the numbers.
12. **Re-run `tools/launch_coverage.py`.** This is the only mechanism that establishes each of the 136 migrated launch points actually fires, and its last run predates the same three changes.
13. **Backfill `pallas_launches`** into the 34 non-reference profiles that lack it (F8), which requires re-profiling.
14. **Add HLO-level launch assertions** to `test_mla_attention_tpu.py`, `test_quantized_matmul_tpu.py`, `test_splash_attention_tpu.py` and `test_topk_routing_tpu.py` — the four families with no in-file `count_pallas_launches` check. Their coverage currently rests entirely on the suite-wide `launch_coverage.py` (item 12), which is exactly the tool whose last run is stale.

---

## 8. Actions taken (2026-08-21)

All nine no-TPU actions from §7 are done. Each was verified on CPU; none required
hardware. The CPU suite went from **44 to 59 passing**.

| # | Action | Verification |
|---|---|---|
| 1 | `assert_matches` replaces 47 bare `cosine > 0.9999` sites across 8 files, adding an RMS-relative bar of 2e-2 | Probe: ×2, ×0.5 and ×10⁶ outputs now **fail**; ≤1% scale and small noise still pass. The ×10⁶ case scored `cosine = 1.0000000` before |
| 2 | `quantized_matmul/baseline.py` carries upstream's `xla_quantized_matmul` verbatim; false justification rewritten | 4 new CPU tests in `test_quantized_matmul_reference.py` |
| 3 | Copyright claim corrected: 49 of 113, and the six files whose headers are stripped are named | Counted directly |
| 4 | `_recurrent_gated_delta_rule_step` de-shadowed | `inspect.getsourcelines` now resolves each reference to its own repository's helper (94 and 505) |
| 5 | `speculative_decoding/baseline.py` regains upstream's `jax.jit`; label qualified. Fixed in the generator, not the artifact | Regenerated from the pinned commit |
| 6 | `ragged_paged_attention/sglang_jax_optimized.py` regenerated | Now byte-identical to `flatten_rpa_v3.py`'s output |
| 7 | `tests/test_regenerates_from_pinned.py` — 11 regeneration-and-diff tests | Confirmed to **fail** on a deliberately injected drift, then pass after revert |
| 8 | fused_moe tolerance clamped to `ATOL * min(1, peak)` | Restores upstream's bar at peaks 11.06 and 62.25 (11× and 62× tighter); zeros still fail at every peak |
| 9 | README bookkeeping: F7, F8, F10, F11, F12, F13 | Each number re-measured before editing |

Action 7 is the structural one. The drift in F6 existed because nothing checked
the corpus's foundational claim; there is now a test that does, and it is proven
to fail when the claim is violated rather than merely asserted to.

**F1 and F2 carry a caveat.** The 47 rewritten assertions live in TPU-gated
tests that could not be run here. The 2e-2 bar is this corpus's own existing
choice, and it has roughly 3x headroom against the worst cosine the suite
records (0.99997), but *the first TPU run should be treated as verification of
that bar, not a formality*. Likewise the `quantized_matmul` kernel comparison
was deliberately **not** switched to upstream's reference: the two references
disagree at cosine 0.9998809, below the tests' own 0.9999 bar, so switching
changes pass/fail semantics for a kernel that truncates. That swap remains a
TPU action.

The five TPU actions (§7, items 10-14) are unchanged and still outstanding.
