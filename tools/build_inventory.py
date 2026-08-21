"""Build ``inventory.json`` from the re-derived launch-point ledger.

``tools/audit_launch_points.py`` produces the mechanical part: which files in
the pinned upstream trees contain Pallas launches, and whether each launch is
TPU-compatible.  This script attaches the semantic part -- category, family,
forward/backward role -- and the corpus migration state.

Family assignment is curated, not inferred from filenames alone.  Each mapping
entry carries an ``evidence`` level:

  ``source-read``  the implementation was read and its semantic contract
                   recorded in the family ledger;
  ``path-grouped`` the launch was grouped by upstream module and function name
                   only, and its contract has not been verified yet.

Migration status is never inferred.  A family is ``migrated`` only when a
standalone corpus file exists for it, and ``profiled`` only when a native-shape
result.json exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]

# (repository, upstream path prefix) -> (category, family, evidence)
# Longest matching prefix wins, so a directory rule can be overridden by a file
# rule.  Every one of the 181 audited launch points must match a rule.
FAMILY_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    # ---- JAXBench -------------------------------------------------------
    ("JAXBench", "benchmark/1p_Flash_Attention/", "attention", "flash_attention", "source-read"),
    ("JAXBench", "benchmark/2p_GQA_Attention/", "attention", "splash_attention", "path-grouped"),
    ("JAXBench", "benchmark/3p_MLA_Attention/", "attention", "mla_attention", "path-grouped"),
    ("JAXBench", "benchmark/4p_Sparse_Attention/", "attention", "splash_attention", "path-grouped"),
    ("JAXBench", "benchmark/6p_Paged_Attention/", "attention", "paged_attention", "path-grouped"),
    ("JAXBench", "benchmark/7p_Ragged_Paged_Attention/", "attention", "ragged_paged_attention", "path-grouped"),
    ("JAXBench", "benchmark/8p_GEMM/", "matmul", "dense_matmul", "path-grouped"),
    ("JAXBench", "benchmark/11p_Megablox_GMM/", "moe", "grouped_matmul", "path-grouped"),
    # ---- PallasBench ----------------------------------------------------
    ("PallasBench", "pallasbench/kernels/level1/add_op.py", "elementwise", "elementwise_binary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/multiply_op.py", "elementwise", "elementwise_binary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/clamp.py", "elementwise", "elementwise_unary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/exp_op.py", "elementwise", "elementwise_unary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/log_op.py", "elementwise", "elementwise_unary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/rsqrt_op.py", "elementwise", "elementwise_unary", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/gelu.py", "elementwise", "activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/relu.py", "elementwise", "activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/sigmoid.py", "elementwise", "activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/silu.py", "elementwise", "activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/tanh_act.py", "elementwise", "activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/batched_matmul.py", "matmul", "dense_matmul", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/matmul.py", "matmul", "dense_matmul", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/outer_product.py", "matmul", "dense_matmul", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/cosine_sim.py", "loss", "similarity_loss", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/mse_loss.py", "loss", "similarity_loss", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/cross_entropy.py", "loss", "cross_entropy", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/embedding_lookup.py", "embedding", "embedding_lookup", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/one_hot.py", "embedding", "one_hot", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/layernorm.py", "normalization", "layer_norm", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/rmsnorm.py", "normalization", "rms_norm", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/log_softmax.py", "normalization", "softmax", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/softmax.py", "normalization", "softmax", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/reduce_max.py", "reduction", "reduction", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/reduce_mean.py", "reduction", "reduction", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/reduce_sum.py", "reduction", "reduction", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level1/nucleotide_onehot.py", "other", "genomics", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/pairwise_distance.py", "other", "genomics", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/pwm_scan.py", "other", "genomics", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level3/triangle_update.py", "other", "genomics", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/fused_softmax_cross_entropy.py", "loss", "cross_entropy", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/sigmoid_bce.py", "loss", "binary_cross_entropy", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/geglu.py", "moe", "gated_mlp", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/swiglu.py", "moe", "gated_mlp", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level3/gated_mlp.py", "moe", "gated_mlp", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/layernorm_residual.py", "normalization", "norm_residual", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/rmsnorm_residual.py", "normalization", "norm_residual", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/linear_bias_relu.py", "matmul", "matmul_activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/matmul_gelu.py", "matmul", "matmul_activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/matmul_relu.py", "matmul", "matmul_activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/matmul_silu.py", "matmul", "matmul_activation", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level2/qk_softmax.py", "attention", "qk_softmax", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level3/flash_attention.py", "attention", "flash_attention", "source-read"),
    ("PallasBench", "pallasbench/kernels/level3/multi_head_attention.py", "attention", "multi_head_attention", "path-grouped"),
    ("PallasBench", "pallasbench/kernels/level3/transformer_block.py", "other", "transformer_block", "path-grouped"),
    # ---- MaxText --------------------------------------------------------
    ("MaxText", "attention/ragged_attention.py", "attention", "ragged_mqa_attention", "path-grouped"),
    ("MaxText", "attention/splash_attention_kernel.py", "attention", "splash_attention", "path-grouped"),
    ("MaxText", "tokamax_splash_attention/", "attention", "splash_attention", "path-grouped"),
    ("MaxText", "megablox/", "moe", "grouped_matmul", "path-grouped"),
    ("MaxText", "ragged/", "memory", "sparsecore_ragged_gather", "path-grouped"),
    ("MaxText", "gather_reduce_pallas.py", "memory", "sparsecore_ragged_gather", "path-grouped"),
    # ---- Tokamax --------------------------------------------------------
    ("tokamax", "tokamax/_src/ops/attention/", "attention", "flash_attention", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/flex_attention/", "attention", "flex_attention", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/experimental/tpu/splash_attention/", "attention", "splash_attention", "source-read"),
    ("tokamax", "tokamax/_src/ops/experimental/mla/", "attention", "mla_attention", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/experimental/tpu/topk/", "sampling", "topk_routing", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule/", "state_space", "gated_delta_net", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/gated_linear_unit/", "moe", "gated_mlp", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/linear_softmax_cross_entropy_loss/", "loss", "cross_entropy", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/normalization/", "normalization", "layer_norm", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/ragged_dot/", "moe", "grouped_matmul", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/ragged_gather/", "memory", "sparsecore_ragged_gather", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/ragged_gather_reduce/", "memory", "sparsecore_ragged_gather", "path-grouped"),
    ("tokamax", "tokamax/_src/ops/ragged_scatter/", "memory", "sparsecore_ragged_gather", "path-grouped"),
    # ---- vLLM tpu-inference --------------------------------------------
    ("tpu-inference", "causal_conv1d/", "convolution", "causal_conv1d", "path-grouped"),
    ("tpu-inference", "collectives/", "collectives", "collective_matmul", "path-grouped"),
    ("tpu-inference", "experimental/batched_rpa/", "attention", "ragged_paged_attention", "path-grouped"),
    ("tpu-inference", "experimental/rpa_v3_cp/", "attention", "ragged_paged_attention", "path-grouped"),
    ("tpu-inference", "experimental/deepseek_v4/compress_and_store/", "memory", "kv_cache_update", "path-grouped"),
    ("tpu-inference", "experimental/deepseek_v4/proj_and_save_state.py", "memory", "kv_cache_update", "path-grouped"),
    ("tpu-inference", "experimental/deepseek_v4/mla.py", "attention", "mla_attention", "path-grouped"),
    ("tpu-inference", "experimental/deepseek_v4/mla_swa.py", "attention", "mla_attention", "path-grouped"),
    ("tpu-inference", "experimental/deepseek_v4/streamindex_topk.py", "sampling", "topk_routing", "path-grouped"),
    ("tpu-inference", "flash_attention/", "attention", "flash_attention", "path-grouped"),
    ("tpu-inference", "fused_moe/", "moe", "fused_moe", "path-grouped"),
    ("tpu-inference", "gdn/", "state_space", "gated_delta_net", "path-grouped"),
    ("tpu-inference", "megablox/", "moe", "grouped_matmul", "path-grouped"),
    ("tpu-inference", "mla/v2/transpose.py", "memory", "layout_transpose", "path-grouped"),
    ("tpu-inference", "mla/", "attention", "mla_attention", "path-grouped"),
    ("tpu-inference", "quantized_matmul/", "quantization", "quantized_matmul", "path-grouped"),
    ("tpu-inference", "ragged_paged_attention/v2/ragged_kv_cache_update.py", "memory", "kv_cache_update", "path-grouped"),
    ("tpu-inference", "ragged_paged_attention/", "attention", "ragged_paged_attention", "path-grouped"),
    ("tpu-inference", "sparse_core/", "memory", "sparsecore_ragged_gather", "path-grouped"),
    ("tpu-inference", "structured_sparse_matmul/", "matmul", "structured_sparse_matmul", "path-grouped"),
    # ---- sglang-jax -----------------------------------------------------
    ("sglang-jax", "biased_topk/", "sampling", "topk_routing", "path-grouped"),
    ("sglang-jax", "grouped_topk/", "sampling", "topk_routing", "path-grouped"),
    ("sglang-jax", "dsa/streamindex_topk.py", "sampling", "topk_routing", "path-grouped"),
    ("sglang-jax", "fused_mlp.py", "moe", "gated_mlp", "path-grouped"),
    ("sglang-jax", "fused_moe/", "moe", "fused_moe", "path-grouped"),
    ("sglang-jax", "gmm/", "moe", "grouped_matmul", "path-grouped"),
    ("sglang-jax", "kda/", "state_space", "gated_linear_attention", "path-grouped"),
    ("sglang-jax", "simple_gla/", "state_space", "gated_linear_attention", "path-grouped"),
    ("sglang-jax", "mla/", "attention", "mla_attention", "path-grouped"),
    ("sglang-jax", "paged_attention/", "attention", "paged_attention", "path-grouped"),
    ("sglang-jax", "quantized_matmul/", "quantization", "quantized_matmul", "path-grouped"),
    ("sglang-jax", "ragged_paged_attention/", "attention", "ragged_paged_attention", "path-grouped"),
    ("sglang-jax", "speculative/", "sampling", "speculative_decoding", "path-grouped"),
    ("sglang-jax", "update_kv_cache/", "memory", "kv_cache_update", "path-grouped"),
)

# Forward/backward role, keyed by enclosing function name substring.
ROLE_RULES: tuple[tuple[str, str], ...] = (
    ("_bwd_dkv", "backward_dkv"),
    ("_bwd_dq", "backward_dq"),
    ("bwd_dkv", "backward_dkv"),
    ("bwd_dq", "backward_dq"),
    ("_vjp", "backward"),
    ("_bwd", "backward"),
    ("backward", "backward"),
    ("tgmm", "backward"),  # megablox tgmm is the dW pass of gmm
    ("_fwd", "forward"),
    ("forward", "forward"),
)

# Corpus state.  These are facts about this repository, so they are asserted
# here rather than derived, and validated against the filesystem below.
#
# A corpus directory is not the same thing as a semantic family: the
# flash_attention directory holds three different upstream contracts, and the
# Tokamax one is a Splash kernel.  Each implementation therefore names the
# semantic family its migrated launch points count towards.
CORPUS_DIRECTORIES: dict[str, dict] = {
    "kernels/attention/flash_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "causal_bhsd": (
                "causal scaled dot-product attention over [B,H,S,D]; caller "
                "supplies unscaled Q, kernel applies sm_scale=1/sqrt(D)"
            ),
            "dense_2d": (
                "non-causal scaled dot-product attention over a single [S,D] "
                "head pair; educational tiling, no online softmax rescaling"
            ),
            "splash_mha_hsd": (
                "causal multi-head attention over [H,S,D] with pre-scaled Q "
                "and independent Dqk/Dv; batching is the caller's vmap"
            ),
        },
        "implementations": {
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "flash_attention/kernel.py",
                "family": "flash_attention",
                "contract": "causal_bhsd",
                "entry_points": ["flash_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "inference-only: one launch point, no custom_vjp and so no "
                    "backward pass, unlike JAXBench's file in the same "
                    "directory. The single repo-local helper "
                    "(tpu_inference.utils.align_to) is inlined. Validated "
                    "against baseline.causal_bhsd at cosine 0.99999744. "
                    "Not yet profiled"
                ),
            },
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/1p_Flash_Attention/optimized.py",
                "family": "flash_attention",
                "contract": "causal_bhsd",
                "entry_points": ["workload", "kernel", "flash_attention"],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward", "backward_dq", "backward_dkv"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "B=4,H=64,S=4096,D=128,causal,bf16",
                "profile": "profiles/native/flash_attention/jaxbench/result.json",
            },
            "pallasbench": {
                "file": "pallasbench_optimized.py",
                "repository": "PallasBench",
                "upstream_path": "pallasbench/kernels/level3/flash_attention.py",
                "family": "flash_attention",
                "contract": "dense_2d",
                "entry_points": ["kernel", "pallas_flash_attention"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "S=512,D=64,non-causal,bf16",
                "profile": "profiles/native/flash_attention/pallasbench/result.json",
            },
            "maxtext_tokamax_splash": {
                "file": "maxtext_tokamax_splash_optimized.py",
                "repository": "MaxText",
                "upstream_path": (
                    "tokamax_splash_attention/splash_attention_kernel.py"
                ),
                "family": "splash_attention",
                "contract": "splash_mha_hsd",
                "entry_points": ["build_kernel", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward_dkv"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "WAS EXCLUDED until 2026-08-04 as a 'MaxText vendored copy "
                    "of Tokamax splash'. The directory name says vendored copy; "
                    "the measurement does not. Comparing the two after "
                    "ast.unparse and dropping docstrings, only 7 of 20 shared "
                    "top-level definitions are identical (35%), and this copy "
                    "is 2098 lines against Tokamax's 2346. That is BELOW the "
                    "42% at which the corpus migrated both halves of the gdn v3 "
                    "pair as genuinely diverged, and nowhere near the 90% "
                    "(28/31) that justified excluding JAXBench's "
                    "4p_Sparse_Attention. The exclusion was inconsistent with "
                    "every other duplication call in the corpus, so it is "
                    "lifted and the measurement recorded instead. Same four "
                    "modules flattened as Tokamax's (mask, mask_info, base, "
                    "kernel) and the same public factories. TWO launch points, "
                    "not three: this lineage fuses dQ into the dKV kernel, "
                    "which returns dq_unreduced, dk, dv from one pallas_call, "
                    "where the JAXBench and MaxText attention/ splash kernels "
                    "launch a separate backward-dQ. Validated forward and "
                    "backward against baseline.splash_mha_hsd and cross-checked "
                    "against Tokamax's copy; a test pins the 35% measurement so "
                    "a future convergence re-opens the question. Unprofiled: "
                    "the profiled native shape belongs to Tokamax's copy"
                ),
            },
            "tokamax": {
                "file": "tokamax_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/experimental/tpu/splash_attention/"
                    "splash_attention_kernel.py"
                ),
                "family": "splash_attention",
                "contract": "splash_mha_hsd",
                "entry_points": ["build_kernel", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward_dkv"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "B=8,H=128,S=4096,Dqk=192,Dv=128,causal,bf16",
                "profile": "profiles/native/flash_attention/tokamax/result.json",
            },
        },
    },
    "kernels/moe/grouped_matmul": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "grouped_matmul_2d": (
                "lhs[m,k] @ rhs[num_groups,k,n] where contiguous row blocks of "
                "lhs given by group_sizes select which rhs matrix applies; "
                "requires sum(group_sizes) == m, because the three "
                "implementations disagree on rows covered by no group"
            ),
            "grouped_matmul_2d_transpose": (
                "the backward-dW pass: per group, out[g] = "
                "lhs[start_g:end_g, :].T @ rhs[start_g:end_g, :] giving "
                "[num_groups, k, n]. Contracts over m instead of k. TWO "
                "CALLING CONVENTIONS: JAXBench's tgmm takes lhs already "
                "transposed as [k, m]; megablox v2's tgmm_v2 takes [m, k]. "
                "Neither checks, and at m == k the difference is invisible"
            ),
        },
        "implementations": {
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/11p_Megablox_GMM/optimized.py",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["workload", "kernel", "gmm", "tgmm"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "m=32768,k=4096,n=1536,G=128,bf16",
                "profile": "profiles/native/grouped_matmul/gmm_jaxbench/result.json",
                "notes": (
                    "forward and backward both validated bit-exactly against "
                    "baseline.grouped_matmul_transpose. The backward launch "
                    "point was already in this file -- tgmm is flattened "
                    "alongside gmm -- but went untested until 2026-08-04; the "
                    "same was true of maxtext_v2's and tokamax_v2's tgmm_v2, "
                    "which were counted as migrated backward with no test "
                    "exercising them. CALLING-CONVENTION TRAP: JAXBench's tgmm "
                    "takes lhs ALREADY TRANSPOSED, [k, m], where the three "
                    "megablox v2 kernels take [m, k] and transpose inside. "
                    "Both are 2-D float arrays and neither checks which axis is "
                    "m, so at m == k the mistake is undetectable; a test pins "
                    "the pairing. Only the forward is profiled -- upstream's "
                    "benchmark defines a native shape for gmm, not tgmm"
                ),
            },
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "megablox/gmm.py",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "m=32768,k=4096,n=1536,G=128,bf16",
                "profile": (
                    "profiles/native/grouped_matmul/gmm_tpu_inference/result.json"
                ),
            },
            "maxtext_v2": {
                "file": "maxtext_optimized.py",
                "repository": "MaxText",
                "upstream_path": "megablox",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm_v2", "tgmm_v2", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "m=32768,k=4096,n=1536,G=128,bf16",
                "profile": "profiles/native/grouped_matmul/gmm_maxtext_v2/result.json",
                "notes": (
                    "megablox v2: unlike this repository's v1 kernels it does "
                    "NOT depend on qwix, so no flax or quantization framework "
                    "is needed. gmm_v2 and tgmm_v2 are flattened together "
                    "because tgmm imports gmm as a sibling. Profiled with "
                    "preferred_element_type=float32: v2 changed that default "
                    "from v1's float32 to the input dtype, which costs enough "
                    "precision at k=4096 to fall below the 0.9999 threshold"
                ),
            },
            "tokamax_v2": {
                "file": "tokamax_optimized.py",
                "repository": "tokamax",
                "upstream_path": "tokamax/_src/ops/ragged_dot",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm_v2", "tgmm_v2", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "m=32768,k=4096,n=1536,G=128,bf16",
                "profile": "profiles/native/grouped_matmul/gmm_tokamax_v2/result.json",
                "notes": (
                    "same as maxtext_v2, plus one compatibility fix: upstream "
                    "builds its mesh with pltpu.TensorCoreMesh, which jax "
                    "0.10.2 no longer exposes publicly; it is spelled "
                    "pltpu.create_tensorcore_mesh, the public factory for the "
                    "same object. This v6e has one TensorCore, so the MegaCore "
                    "scaling that mesh exists for is inactive here"
                ),
            },
            "sglang_jax_v2": {
                "file": "sglang_jax_v2_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "gmm/megablox_gmm_kernel/gmm_v2.py",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm_v2", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream -- no repo-local imports -- so a "
                    "straight copy with provenance, unlike MaxText's and "
                    "Tokamax's v2 which need their tgmm sibling flattened in. "
                    "FORWARD ONLY: no tgmm_v2 ships beside it. Like the other "
                    "v2 kernels it needs no qwix. Validated bit-exactly against "
                    "baseline.grouped_matmul_loop with "
                    "preferred_element_type=float32. All FOUR v2 "
                    "implementations are genuinely different code, not vendored "
                    "copies: pairwise 23-57% of shared definitions are "
                    "AST-identical, and this one is the most distant from the "
                    "rest (23-28%). Unprofiled: no upstream benchmark defines a "
                    "native shape for it"
                ),
            },
            "tpu_inference_v2": {
                "file": "tpu_inference_v2_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "megablox/gmm_v2.py",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm_v2", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream, a straight copy with provenance. "
                    "FORWARD ONLY, like sglang-jax's. The closest of the four "
                    "v2 kernels to Tokamax's (44% of shared definitions "
                    "AST-identical) but still genuinely diverged. Adds a "
                    "fuse_act argument the other three lack, and takes a "
                    "per-block rhs_scale where sglang-jax takes a per-tensor "
                    "one. Needs no qwix. Validated bit-exactly against "
                    "baseline.grouped_matmul_loop with "
                    "preferred_element_type=float32. Unprofiled: no upstream "
                    "benchmark defines a native shape for it"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "gmm/megablox_gmm_kernel/gmm.py",
                "family": "grouped_matmul",
                "contract": "grouped_matmul_2d",
                "entry_points": ["gmm", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "m=32768,k=4096,n=1536,G=128,bf16",
                "profile": (
                    "profiles/native/grouped_matmul/gmm_sglang_jax/result.json"
                ),
            },
        },
    },
    "kernels/attention/ragged_paged_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "rpa_v3_cp": (
                "the context-parallel rpa_v3: same attention, but the KV "
                "sequence is sharded across devices, adding cp_rank, "
                "cp_group_size and q_pos_offsets. At cp_group_size=1 it "
                "degenerates to rpa_v3"
            ),
            "rpa_sglang_fused_4d": (
                "sglang-jax's ragged_paged_attention.py: 10 parameters "
                "(queries, keys, values, kv_cache_fused, kv_lens, "
                "page_indices, cu_q_lens, cu_kv_lens, distribution, "
                "custom_mask) over a 4-D fused KV cache with K and V "
                "interleaved on the head axis. Distinct from both rpa_v2 "
                "(6 parameters) and rpa_v3 (5-D packed cache)"
            ),
            "rpa_v2": (
                "f(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs); K "
                "and V interleaved in one paged cache, live sequence count in a "
                "one-element array, right-aligned causal mask"
            ),
            "rpa_v3": (
                "f(queries, keys, values, kv_cache, kv_lens, page_indices, "
                "cu_q_lens, distribution); K/V appended to a 5-D dtype-packed "
                "cache, flattened page table, distribution splits the batch "
                "into decode/prefill/mixed regions; not interchangeable with v2"
            ),
            "batched_rpa": (
                "the same eight required arguments as rpa_v3, but work is "
                "batched across sequences instead of looped per sequence, and a "
                "second Pallas kernel plans the schedule first. Carries a "
                "kv_layout selector: HEAD_ALONG_SUBLANE (upstream's default) "
                "uses rpa_v3's exact cache shape, SEQ_ALONG_LANE uses its own "
                "and requires page_size=128"
            ),
        },
        "implementations": {
            "tpu_inference_batched": {
                "file": "tpu_inference_batched_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/batched_rpa",
                "family": "ragged_paged_attention",
                "contract": "batched_rpa",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "the corpus's largest implementation by module count: nine "
                    "modules flattened in dependency order (utils, configs, "
                    "schedule, stitch_utils, flash_attention, bref_override, "
                    "tuned_params, kernel, wrapper), which reference each other "
                    "through the package and so need module-prefix stripping as "
                    "well as import removal. Two out-of-package references "
                    "substituted: init_logger -> stdlib logging, and "
                    "tpu_inference.envs -> a shim exposing "
                    "USE_BATCHED_RPA_SEQ_ON_LANE (default False as upstream). "
                    "Validated against tpu-inference v3's pure-JAX reference at "
                    "cosine 0.99998, page_size 16 and 128, and lowers to 4 "
                    "tpu_custom_calls -- its 2 launch points, each run once for "
                    "the decode region and once for the mixed region. "
                    "TWO KV LAYOUTS, and they are not interchangeable: "
                    "HEAD_ALONG_SUBLANE is upstream's default and its cache "
                    "shape is byte-for-byte rpa_v3's, which is what lets the v3 "
                    "reference validate this kernel with no conversion in "
                    "between; SEQ_ALONG_LANE has its own shape and REQUIRES "
                    "page_size=128 for tile alignment, rejecting anything else. "
                    "Comparing the two layouts needs kv_lens == q_len so the "
                    "kernel writes every KV token itself -- feeding "
                    "independently-random caches of the two shapes compares "
                    "different data and reports a spurious cosine 0.097. "
                    "RpaCase.PREFILL fails to compile (Mosaic operand-9 layout "
                    "verification) both flattened and unflattened, so it is an "
                    "upstream limitation, not a migration defect; upstream's own "
                    "wrapper only ever calls DECODE and MIXED. Auto-tuned block "
                    "sizes can overflow SMEM at shapes with no tuned entry "
                    "(tuned_params_mapping is empty upstream), so the tests pass "
                    "explicit BlockSizes. Unprofiled: upstream defines no "
                    "benchmark shape for this contract, as for the other rpa_v3 "
                    "kernels. WAS BLOCKED until 2026-08-04 by a flatten bug, not "
                    "an upstream one: the blind `re.sub(r\"\\bschedule\\.\", "
                    "\"\")` module-prefix strip also rewrote the `schedule` "
                    "PARAMETER of compute_metadata, turning "
                    "`schedule.s_idx[step, lane] = s_idx` into "
                    "`s_idx[step, lane] = s_idx`. That parses, imports and "
                    "passes every static check, then fails at run time with "
                    "'JAX arrays are immutable'. tools/flatten_gdn."
                    "strip_module_prefixes now does the strip over the AST with "
                    "a scope stack and skips shadowed names; "
                    "tests/test_flatten_tools.py pins it"
                ),
            },
            "tpu_inference_v3_cp": {
                "file": "tpu_inference_v3_cp_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/rpa_v3_cp/kernel.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v3_cp",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "the context-parallel v3 variant: it shards the KV sequence "
                    "across devices, so the contract carries cp_rank, "
                    "cp_group_size and q_pos_offsets the single-device v3 "
                    "kernel has none of. Only kernel.py plus the shared v3 "
                    "util.py needed flattening. NOTE the audit records the "
                    "launch point's enclosing function as run_rpa_kernel, which "
                    "is the inner function holding the pallas_call; the public "
                    "entry point is ragged_paged_attention. Validated against "
                    "the file's own ref_ragged_paged_attention at cosine "
                    "0.99997550 in the DEGENERATE cp_group_size=1 case -- this "
                    "host is a single v6e chip, so the actually-sharded path is "
                    "NOT validated and would need a multi-device host. "
                    "Not yet profiled"
                ),
            },
            "sglang_jax_v2": {
                "file": "sglang_jax_v2_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "ragged_paged_attention/ragged_paged_attention.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_sglang_fused_4d",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "a THIRD contract in this family, not rpa_v2 despite "
                    "the filename: the same 10 parameters as sglang's v3 file "
                    "plus a custom_mask no other migrated contract has, and a "
                    "4-D fused KV cache (pages, page_size, 2*num_kv_heads, "
                    "head_dim) -- K and V interleaved on the head axis, "
                    "unpacked -- where rpa_v3 uses the 5-D packed layout. That "
                    "4-D form was read from the kernel's own validation, which "
                    "destructures (_, page_size, cache_num_kv_heads_"
                    "interleaved, head_dim); the file's own "
                    "get_kv_cache_shape returns the 5-D shape and the kernel "
                    "REJECTS it. Upstream's only call site "
                    "(benchmark/kernels/flash_attention/) builds that same 5-D "
                    "shape and would hit the same assertion, so that benchmark "
                    "is stale against the kernel. Validated at kv_len == q_len "
                    "so no KV pre-exists and no layout conversion is needed: "
                    "cosine 0.99997848 against tpu-inference's pure-JAX v3 "
                    "reference. Cache reuse across layouts is not validated. "
                    "Not yet profiled"),
            },
            "tpu_inference_hd64": {
                "file": "tpu_inference_hd64_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "ragged_paged_attention/v3/kernel_hd64.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v3",
                "entry_points": ["ragged_paged_attention_hd64", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "a head-dim-64 specialisation of the v3 kernel with its "
                    "own tuned-size table; util.py, tuned_block_sizes_hd64.py "
                    "and kernel_hd64.py flattened, get_device_name extracted "
                    "from tpu_inference/utils.py and vLLM's init_logger "
                    "replaced with the stdlib logger. Its KV-cache layout "
                    "DIFFERS from the general v3 kernel's -- "
                    "(pages,page_size,1,2,128) against (pages,page_size,2,2,128) "
                    "at H=2/D=64 -- because v3 gives K and V a 128-lane each and "
                    "wastes half of both while hd64 packs K and V of one head "
                    "into one lane. Validated at kv_len == q_len so no KV "
                    "pre-exists in the cache and a zeroed cache is logically "
                    "identical in either layout: cosine 0.99998015 against "
                    "tpu-inference's own pure-JAX v3 reference, marginally "
                    "closer than the general v3 kernel's 0.99997669. Cache "
                    "reuse is ALSO validated, without a layout converter: each "
                    "kernel fills its OWN cache in a first call and reuses it in "
                    "a second, so both hold the same logical KV in their own "
                    "layouts by construction (hd64 0.99994159 against the v3 "
                    "kernel's 0.99993628 on the same two-phase workload). These "
                    "kernels alias their output onto q, so every call needs a "
                    "fresh copy. Also reports no tuned sizes for head_dim "
                    "64 at this shape and falls back to a heuristic, so timing "
                    "would need tuning settled first. Not yet profiled"),
            },
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/7p_Ragged_Paged_Attention/optimized.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v2",
                "entry_points": ["workload", "kernel", "ragged_paged_attention"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": (
                    "max_num_batched_tokens=4096,max_num_seqs=64,Hq=64,Hkv=8,"
                    "D=128,page_size=16,pages_per_seq=256,bf16"
                ),
                "profile": (
                    "profiles/native/ragged_paged_attention/rpa_jaxbench/result.json"
                ),
                "notes": (
                    "one compatibility fix: pltpu.ANY -> pl.MemorySpace.ANY, "
                    "which jax 0.10.2 removed"
                ),
            },
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "ragged_paged_attention/v2/kernel.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v2",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": (
                    "max_num_batched_tokens=4096,max_num_seqs=64,Hq=64,Hkv=8,"
                    "D=128,page_size=16,pages_per_seq=256,bf16"
                ),
                "profile": (
                    "profiles/native/ragged_paged_attention/"
                    "rpa_tpu_inference/result.json"
                ),
                "notes": (
                    "profiled with JAXBench's autotuned blocks so the "
                    "comparison isolates the kernel from the tuning"
                ),
            },
            "tpu_inference_v3": {
                "file": "tpu_inference_v3_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "ragged_paged_attention/v3/kernel.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v3",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "no upstream benchmark configuration defines a native shape "
                    "for the v3 contract, so this is validated for correctness "
                    "but deliberately not given a native-shape performance "
                    "number"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "ragged_paged_attention/ragged_paged_attention_v3.py",
                "family": "ragged_paged_attention",
                "contract": "rpa_v3",
                "entry_points": ["ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "same as tpu_inference_v3: correctness validated, no "
                    "upstream-defined native shape to profile at"
                ),
            },
        },
    },
    "kernels/memory/kv_cache_update": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "kv_cache_update": (
                "f(new_kv[T,H2,D], slices[3,S], kv_cache[P*page_size,H2,D], "
                "num_slices[1]); for each live column of slices, copy "
                "new_kv[new_kv_start:+len] into kv_cache[kv_cache_start:+len]. "
                "Zero FLOPs: judged on achieved HBM bandwidth, not MXU"
            ),
            "dsv4_compress_and_store": (
                "DeepSeek-V4's compressor: project hidden_states through "
                "wkv_wgate into a compressed KV state, scatter it into a packed "
                "uint8 cache, then at every compress_ratio boundary RMS-norm, "
                "RoPE, fp8-quantize and store one record. Unlike "
                "kv_cache_update this is NOT a pure copy -- it does real "
                "arithmetic, so it is not judged on bandwidth alone. Two "
                "implementations exist upstream and they do NOT share a cache "
                "layout: the Pallas path uses [pages, physical_page_size, 4, "
                "128] uint8, the pure-JAX path [pages, page_size//4, 4, 640]"
            ),
        },
        "implementations": {
            "tpu_inference_dsv4": {
                "file": "tpu_inference_dsv4_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/deepseek_v4",
                "family": "kv_cache_update",
                "contract": "dsv4_compress_and_store",
                "entry_points": [
                    "proj_and_save_state",
                    "compress_norm_rope_store",
                    "compressor_forward",
                    "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "six modules flattened in dependency order "
                    "(proj_and_save_state, then compress_and_store's config, "
                    "compute, buffered_ref, kernel, then compressor_v1 which "
                    "chains the two); 2 pallas_call sites. Three names renamed "
                    "on flatten: proj_and_save_state's Configs/Dimensions/"
                    "TileSizes collide with compress_and_store/config.py's, and "
                    "its inner kernel is literally named `kernel`, which the "
                    "corpus's own `kernel = <entry>` footer would shadow. "
                    "UNBLOCKED 2026-08-05 by MEASURING the packed layout rather "
                    "than deriving it -- the entry was carried UNVALIDATED for "
                    "a day because upstream publishes no host-side readback "
                    "(the slot-to-address math lives in on-device BufferedRef "
                    "classes) and its only pure-JAX twin, compressor.py, uses a "
                    "DIFFERENT cache layout. Two facts were read off the kernel "
                    "with one-token probes: (1) the SLOT_PACK axis is a "
                    "BYTE-PLANE axis -- writing all-ones sets exactly 4096 "
                    "bytes whose only values are 63 and 128, the two nonzero "
                    "bytes of a float32 1.0 -- so moving that axis last and "
                    "bitcasting recovers the state; (2) slot_mapping is in "
                    "PHYSICAL ROWS, not tokens: slot S occupies rows "
                    "[S, S+state_rows_per_token) of page S//physical_page_size, "
                    "so tokens must be spaced state_rows_per_token apart and "
                    "passing token indices silently overlaps them by all but "
                    "one row. Both readbacks are self-checking rather than "
                    "assumed: the state one round-trips small integers EXACTLY "
                    "(no wrong byte assignment does), and the record one is "
                    "decided by an independent oracle -- row-major over the "
                    "(4,128) row agrees BIT-EXACTLY with the pure-JAX twin's "
                    "record while the transposed reading is off by 1e34. "
                    "VALIDATED: proj_and_save_state over 128 slots to within "
                    "0.0255 against a bf16 ulp of 0.0402 (its matmul runs in "
                    "bf16 on the MXU, so an exact-fp32 expectation is what made "
                    "this look like a layout error at first); "
                    "compress_norm_rope_store bit-exact against "
                    "dsv4_reference.compress_norm_rope_store fed the same state. "
                    "TRAP still worth keeping: proj_and_save_state hard-codes "
                    "tile_n=128 and its pallas_call sets "
                    "disable_bounds_checks=True, so a num_tokens that is not a "
                    "multiple of 128 reads past the end of hidden_states and "
                    "HARD-FAULTS THE CHIP rather than raising. compressor_v1.py, "
                    "the only upstream caller, has no caller of its own -- dead "
                    "code upstream, which is why no test ships with it. "
                    "Unprofiled: upstream defines no benchmark shape"
                ),
            },
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": (
                    "ragged_paged_attention/v2/ragged_kv_cache_update.py"
                ),
                "family": "kv_cache_update",
                "contract": "kv_cache_update",
                "entry_points": ["kv_cache_update", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/kv_cache_update/kv_tpu_inference/result.json"
                ),
                "notes": (
                    "no upstream benchmark defines a native shape for this "
                    "kernel, so the profile records a declared validation "
                    "shape with native_source_shape=false"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "update_kv_cache/update_kv_cache.py",
                "family": "kv_cache_update",
                "contract": "kv_cache_update",
                "entry_points": ["kv_cache_update", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/kv_cache_update/kv_sglang_jax/result.json"
                ),
                "notes": (
                    "the audit counts two launches in this file: the kernel "
                    "and its shard_map wrapper, both migrated together. "
                    "Declared validation shape, as above"
                ),
            },
        },
    },
    "kernels/quantization/quantized_matmul": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "quantized_matmul_per_channel": (
                "f(x[n_batch,n_in], w_q[n_out,n_in], w_scale[n_out]); one "
                "scale per output channel, optional symmetric per-token "
                "activation quantization; weight zero points unsupported"
            ),
            "quantized_matmul_blockwise": (
                "same, but sub-channel along the contraction axis: block_size "
                "is a scalar over n_in and w_scale is [n_in//block_size,1,n_out], "
                "so the contraction must be split and scaled per block"
            ),
        },
        "implementations": {
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                # Two kernel modules were flattened together, so SOURCE names
                # the package directory rather than a single file.
                "upstream_path": "quantized_matmul",
                "family": "quantized_matmul",
                "contract": "quantized_matmul_per_channel",
                "entry_points": [
                    "quantized_matmul_kernel",
                    "blockwise_quantized_matmul_kernel",
                    "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/quantized_matmul/qm_tpu_inference__w8a8/result.json"
                ),
                "notes": (
                    "kernel.py and blockwise_kernel.py share helpers and are "
                    "flattened into one file; the block-wise entry point is "
                    "renamed to avoid shadowing. No upstream benchmark defines "
                    "a native shape, so the profile records a declared "
                    "validation shape the upstream tuned table covers"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "quantized_matmul/quantized_matmul_kernels",
                "family": "quantized_matmul",
                "contract": "quantized_matmul_per_channel",
                "entry_points": [
                    "quantized_matmul_kernel",
                    "blockwise_quantized_matmul_kernel",
                    "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/quantized_matmul/qm_sglang_jax__w8a8/result.json"
                ),
                "notes": (
                    "same layout as tpu-inference; the inner matmul_kernel is "
                    "AST-identical between the two repositories, only the "
                    "public wrappers differ. Declared validation shape, as above"
                ),
            },
        },
    },
    "kernels/attention/splash_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "splash_attention_mha": (
                "block-sparse attention over q[Hq,S,D], k/v[Hkv,S,D] with a "
                "mask_lib MultiHeadMask compiled to block metadata ahead of the "
                "call; no batch dimension (per-device, callers vmap); Q is NOT "
                "pre-scaled, unlike the Tokamax splash_mha_hsd contract"
            ),
        },
        "implementations": {
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/2p_GQA_Attention/optimized.py",
                "family": "splash_attention",
                "contract": "splash_attention_mha",
                "entry_points": [
                    "make_splash_mha_single_device",
                    "make_splash_mqa_single_device",
                    "attention_reference",
                    "workload",
                ],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward", "backward_dq", "backward_dkv"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/splash_attention/splash_jaxbench/result.json"
                ),
                "notes": (
                    "the corpus's first migrated backward kernels: gradients are "
                    "validated by differentiating the custom_vjp. Self-contained "
                    "on the pinned deps (mask_lib resolves inside jax 0.10.2). "
                    "JAXBench's 4p_Sparse_Attention is the same kernel (28 of 31 "
                    "definitions AST-identical) and is audited but not "
                    "separately migrated. Declared validation shape"
                ),
            },
            "maxtext": {
                "file": "maxtext_optimized.py",
                "repository": "MaxText",
                "upstream_path": "attention/splash_attention_kernel.py",
                "family": "splash_attention",
                "contract": "splash_attention_mha",
                "entry_points": [
                    "make_splash_mha_single_device",
                    "make_splash_mqa_single_device",
                    "attention_reference",
                ],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward", "backward_dq", "backward_dkv"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/splash_attention/splash_maxtext/result.json"
                ),
                "notes": (
                    "shares only ~15% of its definitions with JAXBench's copy "
                    "yet agrees numerically to 8 digits. MaxText's second, "
                    "vendored tokamax_splash_attention copy is audited but not "
                    "migrated. Declared validation shape"
                ),
            },
        },
    },
    "kernels/attention/mla_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "mla_ragged_paged_attention": (
                "nine required args ending cu_q_lens, distribution; latent KV "
                "stream shared by all query heads, cache holds latent and "
                "rotary parts concatenated with kv_dim = align(lkv,128) + "
                "align(r,128); cache_kv is donated"
            ),
            "mla_ragged_paged_attention_cu_kv": (
                "same, plus a required cu_kv_lens before distribution -- the "
                "same divergence seen between rpa_v2 and rpa_v3"
            ),
            "mla_ragged_paged_attention_head_major": (
                "the same nine argument NAMES as mla_ragged_paged_attention, "
                "but ql_nope is [num_q_heads, num_tokens, lkv_dim] and the "
                "output comes back head-major too; q_pe stays token-major. "
                "Nothing rejects a token-major call when num_tokens == "
                "num_q_heads -- it returns a wrong answer at cosine 0.534"
            ),
            "mla_sparse_topk": (
                "one fused q [num_tokens, num_q_heads, head_dim] instead of a "
                "ql_nope/q_pe split, no new_kv (the cache is written by a "
                "separate compressor), a uint8 DSV4 FP8-packed cache, "
                "attention_sinks, a carry-in (swa_accumution, swa_l, swa_m), "
                "and exactly one of topk_indices (CSA) or kv_lens_to_attend "
                "(HCA)"
            ),
            "mla_sliding_window": (
                "new_kv separate from cache_kv, a uint8 DSV4 FP8-packed cache "
                "the kernel itself writes, and four return values "
                "(out, cache, L, m) -- L and m are the carry-in the "
                "mla_sparse_topk kernel consumes"
            ),
        },
        "implementations": {
            "tpu_inference_dsv4": {
                "file": "tpu_inference_dsv4_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/deepseek_v4/mla.py",
                "contract": "mla_sparse_topk",
                "entry_points": ["mla_ragged_paged_attention", "kernel"],
                "family": "mla_attention",
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream, a straight copy with provenance. "
                    "A DIFFERENT contract from mla_ragged_paged_attention: one "
                    "fused q rather than a ql_nope/q_pe split, no new_kv, a "
                    "uint8 DSV4 FP8-packed cache, attention sinks and a "
                    "carry-in. Validated in BOTH modes (CSA topk_indices and "
                    "HCA kv_lens_to_attend) against baseline.ref_dsv4_sparse, "
                    "ported from upstream's own tests/kernels/deepseek_v4/"
                    "mla_test.py. Upstream's settings pin the output near the "
                    "carry-in value 5000/200 = 25 whatever the attention does, "
                    "so the corpus test also runs a neutral carry-in with no "
                    "sinks, where outputs land at absmean ~10 and the "
                    "comparison actually constrains the kernel. Deliberately "
                    "unprofiled: this kernel is the second half of a two-pass "
                    "attention and a standalone timing without the sliding-"
                    "window pass in front of it would not measure how it runs"
                ),
            },
            "tpu_inference_dsv4_swa": {
                "file": "tpu_inference_dsv4_swa_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/deepseek_v4/mla_swa.py",
                "contract": "mla_sliding_window",
                "entry_points": [
                    "mla_sliding_window_ragged_paged_attention", "kernel",
                ],
                "family": "mla_attention",
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream; a distinct contract, not a tuning "
                    "variant -- new_kv separate from cache_kv, a uint8 DSV4 "
                    "FP8-packed cache the kernel writes, and four returns "
                    "(out, cache, L, m). Validated over upstream's own three-"
                    "step schedule (prefill, mixed, full decode) against "
                    "baseline.ref_dsv4_sliding_window, ported from "
                    "tests/kernels/deepseek_v4/mla_swa_test.py, comparing all "
                    "four returns. CAUTION recorded by a test: at upstream's "
                    "own settings the OUTPUT comparison is vacuous -- sinks "
                    "drawn from [200,500] against logits reaching ~135 make "
                    "every output ~1e-26 on both sides, so two arrays of zeros "
                    "agree at any tolerance and only L, m and the cache carry "
                    "the check. The corpus test also runs sink=0.0, where "
                    "outputs land at absmean 0.52. FOLLOW-UP 2026-08-05: "
                    "tools/assertion_strength.py confirmed that degeneracy "
                    "mechanically -- three assert_allclose calls here were "
                    "satisfiable by an all-zero output -- so the test now "
                    "asserts which regime it is in rather than comparing two "
                    "arrays of zeros, and m/L are checked bitwise and at 1e-5 "
                    "since both come back near-exact. Deliberately unprofiled: "
                    "this is the first half of a two-pass attention, and "
                    "upstream defines no benchmark shape for it"
                ),
            },
            "tokamax": {
                "file": "tokamax_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/experimental/mla/pallas_mosaic_tpu_kernel.py"
                ),
                "family": "mla_attention",
                "contract": "mla_ragged_paged_attention",
                "entry_points": ["mla_ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": "profiles/native/mla_attention/mla_tokamax/result.json",
                "notes": (
                    "self-contained upstream, a straight copy. Validated "
                    "cross-repository against tpu-inference's pure-JAX "
                    "reference, which is correctness-only (not jittable), so "
                    "this family has no pure-JAX speed denominator. Declared "
                    "validation shape at DeepSeek-V3 MLA dims"
                ),
            },
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "mla/v1/kernel.py",
                "family": "mla_attention",
                "contract": "mla_ragged_paged_attention",
                "entry_points": [
                    "mla_ragged_paged_attention",
                    "ref_mla_ragged_paged_attention",
                    "kernel",
                ],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/mla_attention/mla_tpu_inference/result.json"
                ),
                "notes": (
                    "supplies the family's pure-JAX reference. Four helpers "
                    "from the ragged-paged-attention v3 util module are "
                    "inlined. Declared validation shape, as above"
                ),
            },
            "tpu_inference_v2": {
                "file": "tpu_inference_v2_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "mla/v2/kernel.py",
                "family": "mla_attention",
                "contract": "mla_ragged_paged_attention_head_major",
                "entry_points": ["mla_ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "four repo-local imports resolved: pack_new_kv inlined from "
                    "mla/v2/kv_utils.py (the other seven names that module "
                    "exports are byte-identical to definitions kernel.py "
                    "already has, so they were not copied), xpose_pipeline and "
                    "its host-side helper inlined from mla/v2/transpose.py, the "
                    "MLA_XPOSE_N_TILE_SIZE env lookup inlined with the same "
                    "name and default of 160, and the vLLM logger replaced by "
                    "the stdlib one. sympy.divisors, which transpose.py uses "
                    "only for host-side integer tile selection, was replaced by "
                    "a trial-division equivalent because sympy is not in the "
                    "pinned dependency set; a test checks it against the "
                    "brute-force definition for n in 1..2000. This is NOT the "
                    "qwix situation -- qwix decides numerics, divisors is a "
                    "definition. The file therefore holds 2 pallas_calls: the "
                    "MLA kernel, and the inlined xpose_pipeline, which is a "
                    "layout_transpose launch point and is NOT counted here. "
                    "WARNING: same nine argument names as v1 but ql_nope is "
                    "head-major and the output is head-major; see the contract "
                    "note. Validated against v1's pure-JAX reference with a "
                    "bf16 cache (cosine 0.99999142) and an fp8_e4m3fn cache "
                    "(0.99621725) -- upstream's own test covers only fp8 and "
                    "says v2 'only supports FP8 KV cache', but bf16 works and "
                    "agrees more closely. Registered for profiling as "
                    "mla-tpu-inference-v2, with the head-major transpose "
                    "applied to the REFERENCE output rather than wrapped around "
                    "the kernel, so the timed function stays the kernel alone. "
                    "No result.json yet: profiling any MLA implementation, "
                    "including the three that already have one recorded, "
                    "currently fails on this host at trace extraction -- 10 "
                    "traced iterations yield 3 jit device events and "
                    "profile_kernel refuses the incomplete trace. "
                    "flash_attention and ragged_paged_attention profile fine on "
                    "the same host, so it is specific to this family; MLA is "
                    "also the only family that sets donated_argnum"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "mla/v2/kernel.py",
                "family": "mla_attention",
                "contract": "mla_ragged_paged_attention_cu_kv",
                "entry_points": ["mla_ragged_paged_attention", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/mla_attention/mla_sglang_jax/result.json"
                ),
                "notes": (
                    "takes ten required args (adds cu_kv_lens), so a separate "
                    "contract. Its tuned block-size table is imported lazily "
                    "inside the lookup and is flattened in. Declared "
                    "validation shape, as above"
                ),
            },
        },
    },
    "kernels/state_space/gated_linear_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "kda_chunk_fwd": (
                "Kimi Delta Attention chunked forward: linear-attention "
                "recurrence with per-channel gates g, a beta delta term, and "
                "an optional in-kernel L2 norm of q/k. Reference is sglang-jax's "
                "own naive_recurrent_kda"
            ),
            "simple_gla_fwd": (
                "lighter gated linear attention: one scalar decay g_gamma per "
                "head, no delta term. References are sglang-jax's own "
                "naive_gla_prefill / naive_gla_decode"
            ),
        },
        "implementations": {
            "sglang_jax_kda": {
                "file": "sglang_jax_kda_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "kda/kda.py",
                "family": "gated_linear_attention",
                "contract": "kda_chunk_fwd",
                "entry_points": [
                    "chunk_kda_fwd", "chunk_local_cumsum_vector",
                    "kda_fwd_intra", "chunk_gated_delta_rule_fwd_h",
                    "chunk_kda_fwd_o_gk", "kernel",
                ],
                "migrated_launch_points": 4,
                "audited_launch_points": 4,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream, so no flattening was needed. "
                    "Four launch points, one per stage of the chunked "
                    "algorithm. Validated against upstream's own "
                    "naive_recurrent_kda at cosine 1.00000000 (rms-relative "
                    "9.6e-05) once the recurrence is made contracting: with a "
                    "weak decay the delta rule accumulates and outputs reach "
                    "~1e14, where absolute comparison is meaningless. "
                    "Not yet profiled"
                ),
            },
            "sglang_jax_simple_gla": {
                "file": "sglang_jax_simple_gla_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "simple_gla",
                "family": "gated_linear_attention",
                "contract": "simple_gla_fwd",
                "entry_points": [
                    "simple_gla_fwd", "chunk_simple_gla_fwd_varlen",
                    "decode_simple_gla_fused", "kernel",
                ],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "simple_gla.py + simple_gla_fused.py flattened into one "
                    "file. Two launch points in the chunked prefill path, one "
                    "in the fused decode path. The calling convention is not "
                    "guessable from the signature and was taken from "
                    "sglang-jax's own lightning_backend call sites: only "
                    "cu_seqlens_dev is passed (cu_seqlens_cpu must be None and "
                    "the non-varlen path raises), head dims must be multiples "
                    "of 128, batch must be 1, g_gamma is a LOG decay so it must "
                    "be negative, and decode_simple_gla_fused donates "
                    "recurrent_buffer. Prefill cos 0.99999720 / state "
                    "0.99999994; decode cos 0.99999672 / state 1.00000000 "
                    "against upstream's naive_gla_prefill and naive_gla_decode. "
                    "Not yet profiled"
                ),
            },
        },
    },
    "kernels/state_space/gated_delta_net": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "unit_lower_triangular_inverse": (
                "given a unit lower triangular A (ones on the diagonal), "
                "return A^-1. A helper for the chunked scan rather than a "
                "model op; the only contract in this family checkable against "
                "an independent oracle (jnp.linalg.inv)"
            ),
            "fused_conv1d_gated_delta_rule": (
                "depthwise causal conv1d fused with the gated delta rule, "
                "carrying TWO caches -- a conv state and a recurrent state -- "
                "where ragged_gated_delta_rule carries one. A different "
                "contract, not a tuning variant of it. Reference is Tokamax's "
                "own run_jax_gdn_attention_local_ref"
            ),
            "ragged_gated_delta_rule": (
                "gated delta rule (linear attention) over ragged sequences, "
                "carrying a per-head [d_k, d_v] recurrent state; returns "
                "(updated_recurrent_state, output). NOTE the SiLU precondition: "
                "the reference applies jax.nn.silu(mixed_qkv) itself, the "
                "tpu-inference kernels expect it already applied. Identical "
                "signatures, different preconditions -- see baseline.PRE_SILU_INPUT"
            ),
        },
        "implementations": {
            "tpu_inference_v1": {
                "file": "tpu_inference_v1_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "gdn/v1",
                "family": "gated_delta_net",
                "contract": "ragged_gated_delta_rule",
                "entry_points": [
                    "ragged_gated_delta_rule", "fused_decoding_gdn",
                    "fused_recurrent_gdn", "calculate_chunk_indices", "kernel",
                ],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "the corpus's first state_space family. Four upstream "
                    "modules flattened into one file; "
                    "get_default_block_sizes is defined in both the decode and "
                    "the recurrent kernel with different bodies and was renamed "
                    "per module. Three launch points: chunk metadata, recurrent "
                    "prefill, decode. Validated against upstream's own pure-JAX "
                    "reference at cosine 0.9999976 once the SiLU precondition is "
                    "respected; fed raw input it silently drops to 0.72. Not yet "
                    "profiled"
                ),
            },
            "tpu_inference_triangle_solver": {
                "file": "tpu_inference_triangle_solver_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "gdn/triangle_solver.py",
                "family": "gated_delta_net",
                "contract": "unit_lower_triangular_inverse",
                "entry_points": [
                    "newton_schulz_inverse_pallas",
                    "decompose_triangular_matrix_inverse_pallas", "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "two algorithms for one contract: Newton-Schulz iteration "
                    "and blockwise decomposition, both inverting a unit lower "
                    "triangular matrix. The only contract in this family with "
                    "an oracle independent of upstream, so it is checked "
                    "against jnp.linalg.inv (cos 1.00000000, maxabs ~5e-9) as "
                    "well as by the residual |A^-1 A - I|. VMEM bounds n: both "
                    "materialise the full [batch, n, n], so newton_schulz OOMs "
                    "at n>=128 and the blockwise kernel reaches n=128 but OOMs "
                    "at n=256. Validated envelope is pinned in the tests. "
                    "Not yet profiled"
                ),
            },
            "tpu_inference_v2": {
                "file": "tpu_inference_v2_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "gdn/v2",
                "family": "gated_delta_net",
                "contract": "ragged_gated_delta_rule",
                "entry_points": [
                    "ragged_gated_delta_rule_decode_only", "recurrent_scan",
                    "fused_decoding_gdn", "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "same contract as v1 but two independent entry points "
                    "instead of one dispatcher: a decode-only kernel and a "
                    "chunked recurrent scan. invert_triangular_matrix is "
                    "defined in two modules and was renamed per module. "
                    "SiLU convention differs from v1: recurrent_scan applies "
                    "SiLU itself (pass raw), decode_only takes an explicit "
                    "apply_silu flag. use_qk_norm_in_gdn=True is required, not "
                    "optional -- at False the kernel returns cosine ~7e-4. "
                    "Not yet profiled"
                ),
            },
            "tokamax_v3": {
                "file": "tokamax_v3_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule"
                ),
                "family": "gated_delta_net",
                "contract": "fused_conv1d_gated_delta_rule",
                "entry_points": ["fused_conv1d_gdn", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "seven upstream modules flattened; no renaming needed. One "
                    "pl.pallas_call source site, which lowers to 2 "
                    "tpu_custom_calls. Diverged vendored pair with "
                    "tpu-inference gdn/v3 (~42% AST-identical) yet the two "
                    "agree bit-for-bit on every output. Not yet profiled"
                ),
            },
            "tpu_inference_v3": {
                "file": "tpu_inference_v3_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "gdn/v3",
                "family": "gated_delta_net",
                "contract": "fused_conv1d_gated_delta_rule",
                "entry_points": ["fused_conv1d_gdn", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "the other half of the vendored pair; same seven modules "
                    "and same 24 definition names as Tokamax's, ~42% "
                    "AST-identical. Not yet profiled"
                ),
            },
        },
    },
    "kernels/memory/sparsecore_ragged_gather": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "ragged_gather": (
                "out[i] = x[indices[i]] for i in [start, end); output is padded "
                "to the SparseCore block size and column tile, so only the live "
                "rows and the first x.shape[-1] columns carry data"
            ),
            "ragged_gather_reduce": (
                "MoE combine: gather, scale rows by topk_weights, zero rows "
                "excluded by valid_rows_mask, then sum consecutive groups of "
                "reduce_group_size rows"
            ),
            "sc_gather_reduce": (
                "MaxText's third gather-reduce shape: positional op/idx, "
                "keyword-only reduce_group_size, NO valid_rows_mask, and "
                "explicit row_chunk_size/col_chunk_size tiling. bf16 only"
            ),
            "dense_gather_reduce": (
                "tpu-inference's dense combine: no valid_rows_mask, and "
                "topk_weights is 2-D [tokens, reduce_group_size] rather than "
                "one weight per gathered row"
            ),
        },
        "implementations": {
            "tpu_inference_gather_v2": {
                "file": "tpu_inference_gather_v2_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "sparse_core/ragged_gather_v2.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather",
                "entry_points": ["ragged_gather_v2", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "tpu-inference's gather for the same contract as Tokamax's, and "
                    "BIT-EXACT against x[indices] like them -- the fourth of the "
                    "four bit-exact gathers in this family. Not self-contained: it "
                    "calls core_map_helper.kernel, a thin pl.core_map wrapper "
                    "living beside it, which is inlined and RENAMED to "
                    "core_map_kernel because its upstream name would collide with "
                    "the corpus's own kernel = <entry> export. The rename is done "
                    "over the AST so the word kernel in prose is untouched. "
                    "Unprofiled: the family's overlap harness covers the gather "
                    "contract already via the Tokamax kernels"
                ),
            },
            "tpu_inference_dense_gather_reduce": {
                "file": "tpu_inference_dense_gather_reduce_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "sparse_core/dense_gather_reduce.py",
                "family": "sparsecore_ragged_gather",
                "contract": "dense_gather_reduce",
                "entry_points": ["dense_gather_reduce", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "a DIFFERENT contract from ragged_gather_reduce, not a tuning "
                    "variant: there is no valid_rows_mask, and topk_weights is 2-D "
                    "[tokens, reduce_group_size] rather than one weight per "
                    "gathered row. Self-contained. It checks is_compatible "
                    "itself and falls back to plain JAX when the shapes do not "
                    "suit the SparseCore path. CORRECTED 2026-08-05: the "
                    "original note recorded this as BIT-EXACT at num_rows=512, "
                    "hidden=512, out_rows=256, fp32 -- but is_compatible is "
                    "FALSE there, so the call took _jax_fallback and the "
                    "measurement compared plain JAX against the JAX reference, "
                    "which agrees trivially. The gate is idx.size %% "
                    "(row_chunk_size * num_cores * num_subcores) == 0, a "
                    "multiple of 16384 at the defaults. Re-validated where the "
                    "kernel actually runs (1 tpu_custom_call): out_rows=16384, "
                    "hidden=256, fp32 agrees to 4.8e-07 absolute and "
                    "hidden=512, bf16 to 0.0156 -- close, not bit-exact, like "
                    "the other gather-reduces that accumulate in fp32 and write "
                    "bf16. A test pins both the fallback below the threshold "
                    "and the launch above it"
                ),
            },
            "tpu_inference_gather_reduce_v2": {
                "file": "tpu_inference_gather_reduce_v2_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "sparse_core/ragged_gather_reduce_v2.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather_reduce",
                "entry_points": ["ragged_gather_reduce", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "the MoE combine. Shares 14 top-level definitions with "
                    "MaxText's v2 file of which 12 are AST-identical (85%): a "
                    "vendored pair that HAS diverged, so both are migrated and the "
                    "measurement recorded, the same call made for gdn v3 and the "
                    "opposite of the verbatim streamindex pair. Same "
                    "core_map_helper.kernel inlined and renamed as for the gather. "
                    "CORRECTED 2026-08-05: the 0.0156 figure below was first "
                    "measured at num_rows=512, hidden=512, out_rows=256, fp32 -- a "
                    "shape at which this kernel takes its XLA fallback and no Pallas "
                    "launch happens. It keeps small problems on the TensorCore "
                    "whenever size(x) * itemsize * 2 is under 60%% of VMEM, the same "
                    "fallback Tokamax's gather-reduce has and which this family "
                    "already documented; hidden must also be >= 2048 or its own "
                    "num_row_partitions <= num_simd_lanes assertion fires. "
                    "Re-validated with 1 tpu_custom_call at num_rows=16384, "
                    "hidden=2048, out_rows=4096, bf16, and a test pins the fallback "
                    "boundary. "
                    "Agrees with the reference to 0.0156 absolute, one bf16 "
                    "ulp at the output magnitude -- these kernels gather in fp32 "
                    "and write bf16, so this contract is close-not-exact where the "
                    "plain gathers are bit-exact"
                ),
            },
            "maxtext_gather_reduce_v2": {
                "file": "maxtext_gather_reduce_v2_optimized.py",
                "repository": "MaxText",
                "upstream_path": "ragged/ragged_gather_reduce_v2.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather_reduce",
                "entry_points": ["ragged_gather_reduce", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "MaxText's half of the diverged v2 pair above. Self-contained. "
                    "PRECONDITION found by measurement, not stated upstream: it "
                    "derives num_row_partitions from a cost model and asserts it is "
                    "<= num_simd_lanes (8 here), which a narrow hidden_size trips "
                    "-- hidden=512 raises num_row_partitions=32 must be <= "
                    "num_simd_lanes=8. hidden >= 2048 works. Validated at hidden "
                    "2048 and 4096, bf16, agreeing to 0.0156 absolute"
                ),
            },
            "maxtext_gather_reduce": {
                "file": "maxtext_gather_reduce_optimized.py",
                "repository": "MaxText",
                "upstream_path": "ragged/ragged_gather_reduce.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather_reduce",
                "entry_points": ["ragged_gather_reduce", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "MaxText's v1 combine, kept because it is a genuinely different "
                    "implementation from its own v2 (6 shared definitions, 1 "
                    "identical). It HARD-CODES num_column_partitions = 8 and "
                    "derives the row partitioning from that, which gives it a SHAPE "
                    "FLOOR the docstring does not mention: validated at "
                    "indices.shape[0] >= 1024, where it agrees with the reference "
                    "to 0.0155 absolute, but at 256 and 512 it returns a WRONG "
                    "ANSWER -- relative error 1.0, disagreeing with its own "
                    "enforce_fallback JAX path -- and at some shapes it HALTS THE "
                    "SPARSECORE outright, killing the process. Its own JAX fallback "
                    "is exact at every size. The corpus test stays above the floor "
                    "and asserts the hard-coded partitioning from the source rather "
                    "than tripping it"
                ),
            },
            "maxtext_sc_gather_reduce": {
                "file": "maxtext_sc_gather_reduce_optimized.py",
                "repository": "MaxText",
                "upstream_path": "gather_reduce_pallas.py",
                "family": "sparsecore_ragged_gather",
                "contract": "sc_gather_reduce",
                "entry_points": ["sc_gather_reduce", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "a third gather-reduce contract: positional op/idx, "
                    "keyword-only reduce_group_size, no valid_rows_mask, explicit "
                    "tiling. THREE preconditions, none of them documented and two "
                    "of them upstream bugs in effect. (1) bf16 ONLY -- the guard is "
                    "`if op.dtype != jnp.bfloat16` while its own error message says "
                    "'must be f32 or bf16'. (2) M must be divisible by "
                    "row_chunk_size * num_cores * num_subcores = 16384 at the "
                    "defaults. (3) reduce_group_size <= 4 here: the output "
                    "BlockSpec row dim is (num_lanes // group) // packing, which "
                    "with 8 SparseCore lanes and bf16 packing 2 is ZERO at group=8 "
                    "-- and group=8 is the only value upstream's own "
                    "gather_reduce_sc_test.py uses, so that test cannot pass on "
                    "this v6e. Also col_chunk_size defaults to 3584, which "
                    "overflows this device's 256 KiB SparseCore VMEM; 512 fits. "
                    "Validated at hidden=3584, out_rows=16384, group=4, "
                    "col_chunk_size=512"
                ),
            },
            "tokamax": {
                "file": "tokamax_optimized.py",
                "repository": "tokamax",
                "upstream_path": "tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather",
                "entry_points": ["ragged_gather_pallas", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/sparsecore_ragged_gather/scg_tokamax/result.json"
                ),
                "notes": (
                    "the corpus's first SparseCore kernels: they run on the "
                    "v6e's SparseCores via plsc.VectorSubcoreMesh, not the "
                    "TensorCore. Bit-exact against upstream's own "
                    "no-SparseCore fallback (x[indices]). Declared validation "
                    "shape; no upstream benchmark defines one"
                ),
            },
            "tokamax_v2": {
                "file": "tokamax_v2_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/ragged_gather/pallas_mosaic_v2_tpu_kernel.py"
                ),
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather",
                "entry_points": ["ragged_gather_pallas", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/sparsecore_ragged_gather/scg_tokamax_v2/result.json"
                ),
                "notes": (
                    "the v2 kernel for the same contract; bit-exact, but "
                    "measurably slower than v1 at the profiled shape. "
                    "Declared validation shape"
                ),
            },
            "tokamax_gather_reduce": {
                "file": "tokamax_gather_reduce_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/ragged_gather_reduce/pallas_mosaic_tpu_kernel.py"
                ),
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather_reduce",
                "entry_points": ["ragged_gather_reduce_pallas", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/sparsecore_ragged_gather/"
                    "scg_tokamax_gather_reduce/result.json"
                ),
                "notes": (
                    "the MoE combine step. Unlike the plain gathers this "
                    "kernel silently declines its own Pallas path when "
                    "size(x)*itemsize*2 < 0.6*vmem_capacity (38.4 MiB of fp32 "
                    "x on a v6e) and returns an XLA fallback, so it is "
                    "profiled at num_rows=8192, hidden=2048 rather than the "
                    "gather shape. Declared validation shape"
                ),
            },
            "maxtext": {
                "file": "maxtext_optimized.py",
                "repository": "MaxText",
                "upstream_path": "ragged/ragged_gather.py",
                "family": "sparsecore_ragged_gather",
                "contract": "ragged_gather",
                "entry_points": ["ragged_gather", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/sparsecore_ragged_gather/scg_maxtext/result.json"
                ),
                "notes": (
                    "adds an optional weights argument the Tokamax kernels do "
                    "not have, giving weights[:, None] * x[indices]; both paths "
                    "validated. Declared validation shape"
                ),
            },
        },
    },
    "kernels/sampling/topk_routing": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "router_topk": (
                "f(router_logits[batch,num_experts], topk) -> "
                "(weights[batch,topk], ids[batch,topk]); plain top-k"
            ),
            "router_biased_topk": (
                "same plus correction_bias[num_experts]; ranks on "
                "logits+bias but returns the PRE-bias weights of the winners"
            ),
            "router_grouped_topk": (
                "DeepSeek-style hierarchy: experts split into "
                "num_expert_group groups scored by their two best biased "
                "logits, best topk_group groups kept, topk chosen within them"
            ),
            "streamindex_topk": (
                "DeepSeek-V4's lightning indexer, and the one contract here "
                "that does not route: f(q[T,H,D], indexer_weights[T,H], "
                "cache_kv uint8[pages,page_size//4,4,width], seq_lens, "
                "page_indices, cu_q_lens, distribution, k, compression_ratio) "
                "-> i32[T,k] kv positions in COMPRESSED space. Score is "
                "sum_h relu(q_h . k_s) * w_h, ReLU BEFORE the head sum. "
                "seq_lens is UNCOMPRESSED and the kernel divides by "
                "compression_ratio itself; k must be a multiple of 128"
            ),
        },
        "implementations": {
            "tpu_inference_dsv4": {
                "file": "tpu_inference_dsv4_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "experimental/deepseek_v4/streamindex_topk.py",
                "family": "topk_routing",
                "contract": "streamindex_topk",
                "entry_points": ["streamindex_topk", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "self-contained upstream -- no repo-local imports at all, "
                    "so a straight copy with provenance. A RETRIEVAL kernel, "
                    "not a router: it returns kv positions rather than expert "
                    "weights, so it shares this directory but no contract with "
                    "the three router kernels. Validated on TPU against "
                    "baseline.streamindex_topk_ref, ported from upstream's own "
                    "tests/kernels/deepseek_v4/test_streamindex_topk.py, over "
                    "prefill, decode, mixed and GQA (8 q heads / 1 kv head) "
                    "cases. Correctness is EXACT set equality of the retrieved "
                    "indices -- a nearly-right index is a wrong index, as for "
                    "the router kernels next door. Two traps pinned by tests: "
                    "k must be a multiple of 128 (an assert in the kernel), and "
                    "seq_lens is in UNCOMPRESSED units -- the kernel divides by "
                    "compression_ratio itself, so passing an already-divided "
                    "length does not raise, it silently retrieves from a "
                    "prefix. Unprofiled: upstream defines no benchmark shape "
                    "for this contract"
                ),
            },
            "sglang_jax_dsa": {
                "file": "sglang_jax_dsa_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "dsa/streamindex_topk.py",
                "family": "topk_routing",
                "contract": "streamindex_topk",
                "entry_points": ["streamindex_topk", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "sglang-jax's copy of the same kernel, and the copy is "
                    "VERBATIM: comparing the two after ast.unparse and dropping "
                    "docstrings, 10 of 10 shared definitions match except "
                    "_streamindex_topk_kernel, which differs only in spelling "
                    "one comparison the other way round (`0 <= old_seq_idx` vs "
                    "`old_seq_idx >= 0`). They also agree BITWISE on output. "
                    "Kept as a separate migration rather than excluded as a "
                    "duplicate, following the grouped_matmul precedent where "
                    "sglang-jax's kernel is likewise the same code as "
                    "tpu-inference's (identical HLO) and both are counted -- "
                    "these are two upstream lineages that can diverge, and the "
                    "corpus records the measurement rather than picking one. "
                    "Contrast the splash-attention call, where MaxText's copy "
                    "lives in a directory literally named tokamax_splash_"
                    "attention and IS excluded. A test pins the equivalence, so "
                    "a future divergence shows up as a failure. Same "
                    "reference, same exact-match criterion, same two traps"
                ),
            },
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "biased_topk",
                "family": "topk_routing",
                "contract": "router_biased_topk",
                "entry_points": ["topk_pallas", "biased_topk_pallas", "kernel"],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/topk_routing/topk_sglang_biased/result.json"
                ),
                "notes": (
                    "kept separate from grouped_topk rather than merged: the "
                    "two upstream packages define three colliding names and "
                    "concatenating would silently shadow them. Correctness is "
                    "exact (ids bitwise, weights 0.0), not cosine. Declared "
                    "validation shape; no upstream benchmark defines one"
                ),
            },
            "sglang_jax_grouped": {
                "file": "sglang_jax_grouped_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "grouped_topk",
                "family": "topk_routing",
                "contract": "router_grouped_topk",
                "entry_points": ["grouped_topk_pallas", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": (
                    "profiles/native/topk_routing/topk_sglang_grouped/result.json"
                ),
                "notes": (
                    "returns batch-major like the other two; the kernel "
                    "computes transposed internally and transposes back. "
                    "Declared validation shape, as above"
                ),
            },
        },
    },
    "kernels/attention/ragged_mqa_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "ragged_attention": (
                "decode attention where each sequence attends only over its "
                "own prefix, given by `lengths`; whole blocks past that prefix "
                "are skipped rather than masked. Returns (out, logits_max, "
                "denominator) so callers can combine partial results"
            ),
        },
        "implementations": {
            "maxtext": {
                "file": "maxtext_optimized.py",
                "repository": "MaxText",
                "upstream_path": "attention/ragged_attention.py",
                "family": "ragged_mqa_attention",
                "contract": "ragged_attention",
                "entry_points": ["ragged_mqa", "ragged_mha", "ragged_gqa", "kernel"],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "PREPARED, NOT MIGRATED: validated on CPU but not yet on a "
                    "TPU. One repo-local name resolved (DEFAULT_MASK_VALUE, a "
                    "float constant, inlined with its defining expression), "
                    "and upstream's three reference_* functions moved from the "
                    "kernel file to baseline.py. One pallas_call serves three "
                    "entry points -- ragged_mqa/mha/gqa differ in how they "
                    "reshape and vmap around it -- so this is one launch "
                    "point, pinned by a test. All three match upstream's own "
                    "references to ~1e-7 under interpret mode on CPU, which is "
                    "upstream's own practice: MaxText ships a "
                    "RaggedAttentionCpuTest beside its tpu_only tests. That is "
                    "not a Mosaic lowering, so it does not count as migrated. "
                    "Careful: the kernel and the references take DIFFERENT "
                    "layouts -- ragged_gqa wants k/v as [B,S,KV,D] while "
                    "reference_gqa wants [B,KV,S,D] and a squeezed q -- and "
                    "mha/gqa return an unnormalised output the caller must "
                    "divide by the returned denominator"
                ),
            },
        },
    },
    "kernels/matmul/structured_sparse_matmul": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "structured_sparse_matmul": (
                "N:M structured sparse matmul: the kernel never sees a dense "
                "matrix, only the compressed (nonzeros, metadata) pair that "
                "Sparsifier produces, and must equal the dense product of the "
                "same matrices with pruned entries set to default_value"
            ),
        },
        "implementations": {
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "structured_sparse_matmul/v1/spmm.py",
                "family": "structured_sparse_matmul",
                "contract": "structured_sparse_matmul",
                "entry_points": ["structured_spmm", "_structured_spmm",
                                 "Sparsifier", "gen_sparse_mask", "kernel"],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "PREPARED, NOT MIGRATED: validated on CPU but not yet on a "
                    "TPU. The most self-contained file in the corpus -- zero "
                    "repo-local imports, carried verbatim. All eight sparsity "
                    "arrangements (either operand sparse, sparsity along the "
                    "contracting or free dimension, rhs transposed or not) "
                    "match jnp.dot on the densified operands EXACTLY under "
                    "interpret mode on CPU at 2:4 bf16. jnp.dot is upstream's "
                    "own choice of reference in spmm_v1_test.py and is a plain "
                    "identity. Sparsifier and gen_sparse_mask stay in the "
                    "kernel file rather than the baseline: they are part of "
                    "how the kernel is CALLED, not what it is checked against. "
                    "A test guards the premise that the compression actually "
                    "drops half the elements, since a no-op Sparsifier would "
                    "leave every comparison passing while checking nothing"
                ),
            },
        },
    },
    "kernels/convolution/causal_conv1d": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "ragged_causal_conv1d": (
                "depthwise causal convolution over many packed sequences of "
                "different lengths; each token sees its own sequence's history "
                "plus the kernel_size-1 carried in conv_state, and never its "
                "neighbour's"
            ),
        },
        "implementations": {
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "causal_conv1d/causal_conv1d.py",
                "family": "causal_conv1d",
                "contract": "ragged_causal_conv1d",
                "entry_points": ["ragged_causal_conv1d", "kernel"],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "PREPARED, NOT MIGRATED, and NOT checkable on CPU: the "
                    "kernel uses eight explicit DMAs and a semaphore, which "
                    "the Pallas interpreter does not emulate, so unlike the "
                    "other prepared families this one has had no correctness "
                    "check at all yet. One repo-local import resolved: "
                    "strided_ldst inlined whole (its two functions only make "
                    "sense read together). The reference is upstream's "
                    "reference_causal_conv1d, which lives in its TEST file "
                    "rather than beside the kernel and is EAGER-ONLY -- it "
                    "calls int() on distribution and query_start_loc entries, "
                    "so it does not survive jax.jit. Upstream's tolerance is "
                    "rtol = atol = 1e-2"
                ),
            },
        },
    },
    "kernels/collectives/collective_matmul": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "all_gather_matmul": (
                "a ring all-gather fused into a matmul: each device multiplies "
                "the shard it holds while the next is still in flight, so the "
                "communication hides under MXU work instead of preceding it"
            ),
            "hierarchical_reduce_scatter": (
                "reduce-scatter by recursive halving on SparseCore rather than "
                "the TensorCore, two-stage pipelined to overlap Die-to-Die and "
                "Chip-to-Chip ICI with local adds"
            ),
        },
        "implementations": {
            "tpu_inference_all_gather_matmul": {
                "file": "tpu_inference_all_gather_matmul_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "collectives/all_gather_matmul.py",
                "family": "collective_matmul",
                "contract": "all_gather_matmul",
                "entry_points": ["all_gather_matmul", "_all_gather_matmul_call",
                                 "kernel"],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "PREPARED; NOT VALIDATABLE ON THIS CORPUS'S HARDWARE. "
                    "Requires EXACTLY 8 DEVICES -- upstream's own test opens "
                    "`if jax.device_count() != 8: skipTest`. The kernel indexes "
                    "a ring with lax.axis_index and exchanges shards with its "
                    "left and right neighbours over 14 send/recv semaphores; on "
                    "the v6e-1 this corpus used, that ring has no neighbours. "
                    "This is a hardware requirement, not unfinished work: no "
                    "amount of single-device TPU time would close it. Two "
                    "modules inlined (util.py, the tuned block-size table); "
                    "their module qualifiers were dropped with the scope-aware "
                    "pass rather than a blind regex"
                ),
            },
            "tpu_inference_hierarchical_reduce_scatter": {
                "file": "tpu_inference_hierarchical_reduce_scatter_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "collectives/hierrs_sc/wrapper.py",
                "family": "collective_matmul",
                "contract": "hierarchical_reduce_scatter",
                "entry_points": ["hierarchical_reduce_scatter_local", "kernel"],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "PREPARED; NOT VALIDATABLE ON THIS CORPUS'S HARDWARE. "
                    "Requires a MULTI-CHIP topology: it runs on SparseCore and "
                    "pipelines Die-to-Die against Chip-to-Chip ICI, with "
                    "devices ordered by physical topology coordinates. Five "
                    "modules flattened in dependency order (config, topology, "
                    "dma_pipeline, kernel, wrapper). NOTE for anyone editing "
                    "the flatten: `config.` in these files is attribute access "
                    "on a Config INSTANCE, not a module qualifier -- stripping "
                    "it as a prefix would silently corrupt 76 reads"
                ),
            },
        },
    },
    "kernels/attention/paged_attention": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "paged_attention_decode": (
                "decode attention over a paged KV cache: one query token per "
                "sequence, keys and values scattered across physical pages "
                "named by page_indices, grouped-query by default. Q arrives "
                "**pre-scaled** -- the kernel has no sm_scale parameter"
            ),
        },
        "implementations": {
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/6p_Paged_Attention/optimized.py",
                "family": "paged_attention",
                "contract": "paged_attention_decode",
                "entry_points": ["paged_attention", "workload", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": ("B=64,H_q=64,H_kv=8,D=128,page=16,"
                                 "pages_per_seq=256,bf16"),
                "profile": None,
                "unprofiled_reason": (
                    "not yet profiled -- an open task rather than a decision: "
                    "JAXBench declares a native shape and the corpus's runner "
                    "covers this contract"
                ),
                "notes": (
                    "no repo-local imports; the one non-jax import resolves "
                    "inside the pinned jax. One rename follows that jax: the "
                    "six pltpu.ANY memory spaces become pl.ANY, the current "
                    "spelling of the same value. Validated at the native shape "
                    "against JAXBench's own baseline.py from the same "
                    "directory, carried as jaxbench_reference.py, at "
                    "JAXBench's own atol=1e-2/rtol=2e-2. TWO CONVENTIONS had "
                    "to be reconciled and both are pinned by tests, since "
                    "neither appears in a signature: the kernel takes "
                    "PRE-SCALED Q -- raw Q disagrees with the reference by "
                    "2.9 against a peak of 0.14, and pre-scaled by 0.0015 -- "
                    "and the reference's KV pages are laid out (total_pages, "
                    "page_size, num_kv_heads, head_dim) against the kernel's "
                    "(num_kv_heads, total_pages, page_size, head_dim), an axis "
                    "permutation. Note JAXBench's own workload does not "
                    "pre-scale, so its two files would disagree if anything "
                    "compared them; upstream only benchmarks them"
                ),
            },
        },
    },
    "kernels/sampling/speculative_decoding": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "verify_tree_greedy": (
                "greedy EAGLE tree verification: walk the draft tree from the "
                "root accepting while the target argmax matches the drafted "
                "token, falling back to the next sibling on a mismatch; "
                "returns the accepted path, its length and the predictions"
            ),
            "build_eagle_tree_structure": (
                "turns the draft model's parent_list/selected_index into a "
                "causal tree_mask, per-token positions, and the retrive_index "
                "/ retrive_next_token / retrive_next_sibling first-child, "
                "next-sibling encoding the verify kernels consume"
            ),
            "tree_speculative_sampling_target_only": (
                "the stochastic counterpart to greedy verification: accept a "
                "draft token when a uniform sample falls under the target "
                "probability, with per-token and accumulated thresholds"
            ),
        },
        "implementations": {
            "sglang_jax_verify_tree_greedy": {
                "file": "sglang_jax_verify_tree_greedy_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "speculative/verify_tree_greedy_kernel.py",
                "family": "speculative_decoding",
                "contract": "verify_tree_greedy",
                "entry_points": ["verify_tree_greedy_pallas_call",
                                 "verify_tree_greedy", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "one repo-local import resolved (cdiv, three lines, "
                    "inlined). Validated against upstream's own "
                    "test_verify_tree_greedy: a two-request tree fixed by hand "
                    "with exact expected integers, carried into baseline.py as "
                    "data with the upstream test quoted by extraction. This is "
                    "index arithmetic, so the check is exact equality, not a "
                    "tolerance; a further test asserts the oracle is not "
                    "satisfied by an empty walk. ONE NARROWING, from running "
                    "it: `predicts` is an output-only buffer -- the argument "
                    "gives only shape and dtype, out_shape allocates a fresh "
                    "array, and the kernel writes only the accepted path -- so "
                    "the remaining slots hold whatever that SMEM allocation "
                    "contained. Upstream's vector expects 0 there, which holds "
                    "alone and fails once other tests share the process. The "
                    "corpus asserts accept_index and accept_token_num in full "
                    "and predicts at the written indices, and a second test "
                    "pins that split so a kernel that started zeroing its "
                    "whole output would restore the wider assertion. Not yet "
                    "profiled: no native shape declared"
                ),
            },
            "sglang_jax_build_tree": {
                "file": "sglang_jax_build_tree_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "speculative/build_eagle_tree_structure_kernel.py",
                "family": "speculative_decoding",
                "contract": "build_eagle_tree_structure",
                "entry_points": ["build_eagle_tree_structure_pallas_call",
                                 "build_eagle_tree_structure", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "carried verbatim; the file imports only jax. UPSTREAM "
                    "SHIPS NO CORRECTNESS TEST for this kernel -- only a "
                    "performance benchmark -- so there is no oracle to carry "
                    "and the corpus does not invent one: re-deriving sglang's "
                    "own retrive_next_token/retrive_next_sibling layout would "
                    "be the kind of guess that produces false failures. "
                    "Validated two ways that need no such guess: the traversal "
                    "invariants, that walking first-child/next-sibling from "
                    "the root reaches every one of the draft_token_num nodes "
                    "exactly once and terminates; and by COMPOSITION, feeding "
                    "its three routing outputs into verify_tree_greedy, which "
                    "does have an upstream oracle. GAP: the packed tree_mask "
                    "byte layout is not checked, since nothing upstream "
                    "defines it independently. The wrapper is eager-only -- it "
                    "sizes its output buffers from input values -- so the "
                    "launch count is taken with the arrays closed over. Not "
                    "yet profiled: no native shape declared"
                ),
            },
            "sglang_jax_tree_sampling": {
                "file": "sglang_jax_tree_sampling_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": (
                    "speculative/tree_speculative_sampling_target_only_kernel.py"
                ),
                "family": "speculative_decoding",
                "contract": "tree_speculative_sampling_target_only",
                "entry_points": [
                    "tree_speculative_sampling_target_only_pallas_call",
                    "kernel",
                ],
                "migrated_launch_points": 0,
                "audited_launch_points": 1,
                "migrated_roles": [],
                "unmigrated_roles": ["forward"],
                "standalone": True,
                "correctness": "UNVALIDATED",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "THE KERNEL DOES NOT RUN, and that is upstream's, not the "
                    "flattening's. sglang-jax's own test for it is dead code: "
                    "the body of test_tree_speculative_sampling_target_only "
                    "begins with `return`, above the comment 'this kernel "
                    "still have some problems'. Fed the (unreachable) vectors "
                    "left behind it, the kernel raises TypeError: scan body "
                    "function carry input and carry output must have the same "
                    "pytree structure -- and upstream's UNFLATTENED file "
                    "raises the identical error, checked by loading it "
                    "directly with a synthetic module for its one repo-local "
                    "import. Carried so the corpus records the state of the "
                    "art rather than only its working parts, but counted as 0 "
                    "migrated launch points. tests/test_speculative_tpu.py "
                    "pins the failure, so an upstream fix shows up as a test "
                    "that starts passing rather than as silence"
                ),
            },
        },
    },
    "kernels/moe/fused_moe": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "fused_ep_moe_split_weights": (
                "expert-parallel MoE taking w1/w2/w3 as three separate "
                "weights and pre-computed topk_weights/topk_ids; the kernel "
                "never sees gating_output"
            ),
            "fused_ep_moe_split_weights_v2": (
                "same operand layout as v1, different kernel: SwiGLU clamp "
                "limit, block-wise fp8, and a compaction pass that skips "
                "empty experts; its block config carries 4 fields, not 9"
            ),
            "fused_ep_moe_fused_w1": (
                "expert-parallel MoE taking w1 with gate and up fused on "
                "axis 1, shape (E, 2, H, I), plus raw gating_output, doing "
                "its own top-k and scoring inside the kernel"
            ),
        },
        "implementations": {
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "fused_moe/v1/kernel.py",
                "family": "fused_moe",
                "contract": "fused_ep_moe_split_weights",
                "entry_points": ["fused_ep_moe", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "carried verbatim: the upstream file imports only jax and "
                    "the standard library. Validated against the ref_moe "
                    "upstream ships in the same file, on the gen_moe_inputs "
                    "from its own test, at its own atol=rtol=2e-1. The kernel "
                    "takes routing pre-computed, which upstream's test gets "
                    "from a flax TopK layer that is not in the pinned "
                    "dependency set; the corpus derives it with lax.top_k and "
                    "a test pins that this reproduces ref_moe's own internal "
                    "routing exactly, rather than leaving it a claim. "
                    "Validated at ep_size 1: routing, fused activation and "
                    "blocking are exercised, the cross-device dispatch is "
                    "not. Not yet profiled: upstream declares no native shape"
                ),
            },
            "sglang_jax_v2": {
                "file": "sglang_jax_v2_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "fused_moe/v2/kernel.py",
                "family": "fused_moe",
                "contract": "fused_ep_moe_split_weights_v2",
                "entry_points": ["fused_ep_moe_v2", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "carried verbatim, same as v1. v2 is its own "
                    "parameterisation rather than a variation on v1's: a "
                    "4-field block config instead of 9, a generator returning "
                    "8 arrays instead of 11, a ref_moe taking pre-computed "
                    "routing instead of raw gating, and a smaller default "
                    "shape. Its ref_moe is also EAGER-ONLY -- it forces a "
                    "traced scalar concrete, so it does not survive jax.jit; "
                    "that is upstream's property, is pinned by a test, and is "
                    "why its Pallas-freeness is checked structurally over the "
                    "reference module rather than by lowering. Validated at "
                    "ep_size 1. Not yet profiled: no native shape declared"
                ),
            },
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "fused_moe/v1/kernel.py",
                "family": "fused_moe",
                "contract": "fused_ep_moe_fused_w1",
                "entry_points": ["fused_ep_moe", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "one repo-local import resolved: the tuned block-size "
                    "table and its untuned fallback were inlined, and the "
                    "vLLM logger became the stdlib one. NOT the same kernel "
                    "as sglang-jax's v1 beside it, despite both being called "
                    "fused_ep_moe: of their 7 shared top-level names only "
                    "align_to, broadcast_minor and swigluoai are AST-"
                    "identical, while _fused_ep_moe_kernel, fused_ep_moe and "
                    "ref_moe all differ -- and the signatures differ too, so "
                    "they carry separate contracts. A test measures that "
                    "rather than asserting it. Validated against its own "
                    "ref_moe on its own gen_moe_inputs at atol=rtol=2e-1, at "
                    "ep_size 1. Not yet profiled: no native shape declared"
                ),
            },
        },
    },
    "kernels/memory/layout_transpose": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "layout_transpose": (
                "data movement only: `xpose_full` and `xpose_pipeline` return "
                "`jnp.transpose(x, axes)` exactly, and `pin_vmem_custom_call` "
                "returns its input unchanged -- its purpose is the side "
                "effect of leaving the buffer resident in VMEM"
            ),
        },
        "implementations": {
            "tpu_inference": {
                "file": "tpu_inference_optimized.py",
                "repository": "tpu-inference",
                "upstream_path": "mla/v2/transpose.py",
                "family": "layout_transpose",
                "contract": "layout_transpose",
                "entry_points": ["xpose_full", "xpose_pipeline",
                                 "pin_vmem_custom_call", "kernel"],
                "migrated_launch_points": 3,
                "audited_launch_points": 3,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "three launch points in one file, each separately shown "
                    "to reach Pallas. Two repo-local imports were inlined "
                    "(get_dtype_packing and its helper) and sympy.divisors "
                    "replaced by the trial-division equivalent shared with "
                    "flatten_mla.py. Validated bitwise -- these relocate "
                    "elements rather than compute on them -- against "
                    "jnp.transpose at upstream's own shapes, axes and fp8 "
                    "dtype from tests/kernels/transpose_test.py, including "
                    "its 128x2048x256 case that will not fit VMEM whole and "
                    "its bf16 384x128x512 case needing vmem_limit_bytes above "
                    "the 32 MiB scoped default. The host-side tile chooser is "
                    "covered by upstream's own tables, including the shapes "
                    "for which it must raise rather than mistile. "
                    "DELIBERATE DUPLICATION: mla_attention's "
                    "tpu_inference_v2_optimized.py inlines xpose_pipeline and "
                    "prev_closest_valid_divisor as a dependency and does not "
                    "count them; this file is the counted migration, and a "
                    "test compares the two copies by AST so they cannot "
                    "drift. Not yet profiled: upstream declares no native "
                    "shape"
                ),
            },
        },
    },
    "kernels/loss/cross_entropy": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "linear_softmax_cross_entropy": (
                "fused projection + log-softmax + cross entropy over "
                "x[B,H] @ w[H,V] against int32 labels[B], returning (loss, "
                "lse) without ever materialising the [B,V] logits; the "
                "backward consumes that lse rather than recomputing it"
            ),
        },
        "implementations": {
            "tokamax": {
                "file": "tokamax_optimized.py",
                "repository": "tokamax",
                "upstream_path": (
                    "tokamax/_src/ops/linear_softmax_cross_entropy_loss/"
                    "pallas_mosaic_tpu_kernel.py"
                ),
                "family": "cross_entropy",
                "contract": "linear_softmax_cross_entropy",
                "entry_points": [
                    "linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu",
                    "linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu",
                    "kernel",
                ],
                "migrated_launch_points": 2,
                "audited_launch_points": 2,
                "migrated_roles": ["forward", "backward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "no repo-local imports, but three substitutions were "
                    "needed and each is asserted against upstream's exact "
                    "text: jaxtyping annotations (evaluated at definition "
                    "time, so they cannot simply be deleted) become inert "
                    "stand-ins; pydantic's validated dataclass becomes the "
                    "stdlib one plus a __post_init__ reproducing its ge and "
                    "multiple_of constraints; and two spellings follow the "
                    "pinned jax 0.10.2 -- create_tensorcore_mesh, and naming "
                    "pltpu.HBM on the backward's out_type so emit_pipeline "
                    "takes the passthrough its own out_specs ask for rather "
                    "than trying to allocate a whole-array HBM buffer. "
                    "Validated against upstream's own reference.py, carried "
                    "here as tokamax_reference.py, at upstream's own 1e-4 "
                    "for all three reductions and for a ragged vocabulary. "
                    "Both launch points are separately shown to reach Pallas. "
                    "Not yet profiled: upstream declares no native shape, "
                    "only test parameterisations"
                ),
            },
        },
    },
    "kernels/matmul/dense_matmul": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "dense_matmul_2d": (
                "x[M,K] @ y[K,N] in bf16 with a float32 VMEM accumulator "
                "carried across the K axis"
            ),
        },
        "implementations": {
            "jaxbench": {
                "file": "jaxbench_optimized.py",
                "repository": "JAXBench",
                "upstream_path": "benchmark/8p_GEMM/optimized.py",
                "family": "dense_matmul",
                "contract": "dense_matmul_2d",
                "entry_points": ["matmul", "workload", "kernel"],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": "M=8192,K=8192,N=28672,bf16",
                "profile": None,
                "unprofiled_reason": (
                    "not yet profiled -- unlike the PallasBench kernels this "
                    "is an open task rather than a decision: JAXBench "
                    "declares a native shape and the corpus's runner covers "
                    "this contract"
                ),
                "notes": (
                    "carried verbatim: the upstream file imports only jax, "
                    "functools and the public Pallas namespaces. The kernel "
                    "body is upstream JAX's own, from "
                    "jax.experimental.pallas.ops.tpu.matmul, wrapped by "
                    "JAXBench as a workload. `kernel` is bound to `workload`, "
                    "not `matmul`: block_shape is keyword-only with no "
                    "default, so `matmul` alone is not callable. Validated "
                    "against JAXBench's own baseline.py from the same "
                    "directory -- carried here as jaxbench_reference.py -- at "
                    "JAXBench's own atol=1e-3/rtol=1e-2, which are loose "
                    "because both sides are bf16 on the MXU"
                ),
            },
        },
    },
    "kernels/moe/gated_mlp": {
        "baseline": "baseline.py",
        "semantic_contracts": {
            "fused_gated_mlp": (
                "down(up(x) * silu(gate(x))) in one pipeline, with gate and "
                "up weights interleaved per b_inter block in a single w_gu "
                "matrix, and a psum over the tensor-parallel mesh axis"
            ),
        },
        "implementations": {
            "sglang_jax": {
                "file": "sglang_jax_optimized.py",
                "repository": "sglang-jax",
                "upstream_path": "fused_mlp.py",
                "family": "gated_mlp",
                "contract": "fused_gated_mlp",
                "entry_points": [
                    "apply_fused_mlp_with_padding",
                    "apply_fused_mlp_sharded",
                    "local_fused_mlp",
                    "kernel",
                ],
                "migrated_launch_points": 1,
                "audited_launch_points": 1,
                "migrated_roles": ["forward"],
                "unmigrated_roles": [],
                "standalone": True,
                "correctness": "PASS",
                "native_shape": None,
                "profile": None,
                "notes": (
                    "carried verbatim apart from one repointed import: "
                    "upstream calls shard_map with check_rep, which in the "
                    "pinned jax 0.10.2 is the legacy API at "
                    "jax.experimental.shard_map, so the import is redirected "
                    "rather than the call translated to check_vma -- the two "
                    "flags do not mean the same thing. Validated against "
                    "upstream's own non-fused fallback in models/glm5_moe.py, "
                    "carried as sglang_jax_reference.py together with the "
                    "block-interleaved w_gu packing that file's "
                    "post_load_weights performs; both upstream regions are "
                    "quoted there by extraction rather than transcription. "
                    "The packing is load-bearing and a test proves it, by "
                    "showing the obvious concat([wg, wu]) gives a different "
                    "answer. Validated on a one-device mesh, where the "
                    "kernel's closing psum is the identity: the fusion, the "
                    "pipelining and the packed layout are exercised, the "
                    "cross-shard reduction is not. Not yet profiled: "
                    "upstream declares no native shape, taking b_seq and "
                    "b_inter from the model config"
                ),
            },
        },
    },
}


def _pallasbench_directories() -> dict[str, dict]:
    """The 19 PallasBench directories, derived from `pallasbench_tasks.json`.

    Every other entry above is written by hand, because every other entry
    describes a kernel with its own history: which launch points a file holds,
    which of them the corpus reached, what had to be inlined, what the
    validation shape had to be and why.  PallasBench is the one source where
    that prose would be 43 restatements of the same three facts -- one
    `pallas_call`, self-contained upstream, validated against upstream's own
    `jax_<task>`.  Deriving these keeps `pallasbench_tasks.json` the single
    place a PallasBench fact is written down, and keeps the ledger from drifting
    from the files the flatten script actually produced.

    The per-task semantic contract is upstream's own one-line description from
    `pallasbench/provenance.py`, which also records where PallasBench distilled
    each task from -- `L1/relu` from the JAX Pallas quickstart, `L3/gated_mlp`
    from MaxText, and so on.  Four tasks carry an upstream `change`; that is
    reproduced verbatim rather than summarised, since it is a modification to
    the kernel the corpus is carrying.
    """
    tasks = json.loads((ROOT / "tools" / "pallasbench_tasks.json").read_text())
    directories: dict[str, dict] = {}
    for task, spec in sorted(tasks.items()):
        # flash_attention predates this sweep and lives in its own directory
        # entry above; embedding_lookup does not lower on TPU at all.
        if spec["family"] == "flash_attention" or spec.get("excluded"):
            continue
        provenance = spec["provenance"]
        directory = f"kernels/{spec['category']}/{spec['family']}"
        entry = directories.setdefault(
            directory, {"baseline": "baseline.py", "semantic_contracts": {},
                        "implementations": {}},
        )
        entry["semantic_contracts"][task] = provenance["description"]

        notes = [
            f"level {spec['level']}; PallasBench distilled it from "
            f"{provenance['source']} ({provenance['reference']})",
            "one pallas_call, self-contained upstream apart from the "
            "provenance import; validated against upstream's own "
            f"jax_{task} from pallasbench/baselines/jax_baseline.py, on "
            "upstream's own generate_inputs",
        ]
        if spec.get("validation"):
            notes.append(
                f"validated at {spec['validation']['shapes']} rather than the "
                f"native {spec['input_shapes']}, because "
                f"{spec['validation']['reason']}"
            )
        if provenance.get("change"):
            notes.append(
                f"upstream records one change to this kernel: "
                f"{provenance['change']}"
            )
        entry["implementations"][f"pallasbench_{task}"] = {
            "file": f"pallasbench_{task}_optimized.py",
            "repository": "PallasBench",
            "upstream_path": f"pallasbench/kernels/{spec['file']}",
            "family": spec["family"],
            "contract": task,
            "entry_points": [spec["entry"], "kernel", "pallas_kernel"],
            "migrated_launch_points": 1,
            "audited_launch_points": 1,
            "migrated_roles": ["forward"],
            "unmigrated_roles": [],
            "standalone": True,
            "correctness": "PASS",
            "native_shape": ",".join(
                "x".join(str(d) for d in shape) for shape in spec["input_shapes"]
            ) + ",f32",
            "profile": None,
            "unprofiled_reason": (
                "PallasBench ships its own benchmark harness and publishes "
                "numbers for these kernels; re-measuring them under this "
                "corpus's instruction-level tracing flags, which inflate "
                "timings by roughly 27%, would produce a second and worse set "
                "of numbers for the same kernel"
            ),
            "notes": ". ".join(notes),
        }
    return directories


def _merge_pallasbench() -> None:
    """Fold the derived PallasBench entries into `CORPUS_DIRECTORIES`.

    Three of those directories also hold a hand-written implementation from
    another repository -- Tokamax's cross-entropy, JAXBench's GEMM, sglang-jax's
    fused MLP -- because PallasBench is not the only source in those families.
    So this merges rather than assigns: a plain `update` would silently drop
    whichever side was written second, and the ledger would under-count without
    any test noticing.
    """
    for directory, derived in _pallasbench_directories().items():
        existing = CORPUS_DIRECTORIES.get(directory)
        if existing is None:
            CORPUS_DIRECTORIES[directory] = derived
            continue
        if existing["baseline"] != derived["baseline"]:
            raise ValueError(f"{directory}: two baselines disagree")
        for key in ("semantic_contracts", "implementations"):
            clashes = set(existing[key]) & set(derived[key])
            if clashes:
                raise ValueError(f"{directory}: duplicate {key}: {sorted(clashes)}")
            existing[key].update(derived[key])


_merge_pallasbench()


def match_family(repository: str, path: str) -> tuple[str, str, str]:
    best: tuple[int, str, str, str] | None = None
    for repo, prefix, category, family, evidence in FAMILY_RULES:
        if repo != repository or not path.startswith(prefix):
            continue
        if best is None or len(prefix) > best[0]:
            best = (len(prefix), category, family, evidence)
    if best is None:
        raise KeyError(f"no family rule for {repository}:{path}")
    return best[1], best[2], best[3]


def match_role(function: str) -> str:
    lowered = function.lower()
    for needle, role in ROLE_RULES:
        if needle in lowered:
            return role
    return "forward"


def build(audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    points = []
    for record in audit["launch_points"]:
        category, family, evidence = match_family(
            record["repository"], record["path"]
        )
        points.append(
            {
                **record,
                "category": category,
                "family": family,
                "family_evidence": evidence,
                "role": match_role(record["enclosing_function"]),
                "tpu_compatible": record["backend"] in ("tpu", "portable"),
            }
        )

    families: dict[str, dict] = {}
    for point in points:
        entry = families.setdefault(
            point["family"],
            {
                "category": point["category"],
                "repositories": {},
                "audited_launch_points": 0,
                "audited_tpu_launch_points": 0,
                "roles": set(),
                "migration_status": "not_started",
            },
        )
        entry["audited_launch_points"] += 1
        entry["audited_tpu_launch_points"] += int(point["tpu_compatible"])
        entry["roles"].add(point["role"])
        repo_entry = entry["repositories"].setdefault(
            point["repository"], {"launch_points": 0, "tpu_launch_points": 0, "files": []}
        )
        repo_entry["launch_points"] += 1
        repo_entry["tpu_launch_points"] += int(point["tpu_compatible"])
        if point["path"] not in repo_entry["files"]:
            repo_entry["files"].append(point["path"])

    # Attribute each corpus implementation to its own semantic family, and
    # verify every asserted file actually exists before it can be counted.
    corpus: dict[str, dict] = {}
    for directory_name, spec in CORPUS_DIRECTORIES.items():
        directory = ROOT / directory_name
        if not (directory / spec["baseline"]).is_file():
            raise FileNotFoundError(f"{directory_name}: missing {spec['baseline']}")
        for name, implementation in spec["implementations"].items():
            path = directory / implementation["file"]
            if not path.is_file():
                raise FileNotFoundError(f"{directory_name}: missing {path}")
            family = implementation["family"]
            if family not in families:
                raise KeyError(f"{directory_name}/{name}: unknown family {family}")
            entry = families[family]
            entry.setdefault("corpus_implementations", []).append(
                {
                    "directory": directory_name,
                    "name": name,
                    "baseline": spec["baseline"],
                    "semantic_contracts": spec["semantic_contracts"],
                    **implementation,
                }
            )
            entry["migrated_launch_points"] = (
                entry.get("migrated_launch_points", 0)
                + implementation["migrated_launch_points"]
            )
        corpus[directory_name] = {
            "baseline": spec["baseline"],
            "semantic_contracts": spec["semantic_contracts"],
            "implementations": sorted(spec["implementations"]),
        }

    for entry in families.values():
        entry["roles"] = sorted(entry["roles"])
        entry["source_repository_count"] = len(entry["repositories"])
        entry["tpu_source_repository_count"] = sum(
            1 for repo in entry["repositories"].values() if repo["tpu_launch_points"]
        )
        entry.setdefault("corpus_implementations", [])
        migrated_points = entry.get("migrated_launch_points", 0)
        entry["migrated_launch_points"] = migrated_points
        if migrated_points == 0:
            entry["migration_status"] = "not_started"
        elif migrated_points >= entry["audited_tpu_launch_points"]:
            entry["migration_status"] = "complete"
        else:
            entry["migration_status"] = "partial"

    tpu_total = sum(1 for point in points if point["tpu_compatible"])
    migrated_total = sum(
        entry["migrated_launch_points"] for entry in families.values()
    )
    return {
        "generated_from": {
            "audit": str(audit_path.name),
            "pinned_commits": audit["pinned_commits"],
            "parse_failures": audit["parse_failures"],
        },
        "counts": {
            "source_files": len({(p["repository"], p["path"]) for p in points}),
            "audited_launch_points": len(points),
            "audited_tpu_launch_points": tpu_total,
            "audited_gpu_launch_points": len(points) - tpu_total,
            "semantic_families": len(families),
            "migrated_launch_points": migrated_total,
            "families_with_migrations": sum(
                1 for entry in families.values() if entry["migrated_launch_points"]
            ),
            "corpus_directories": len(corpus),
            "note": (
                "audited_tpu_launch_points is an inventory of migration "
                "candidates, not migration progress; only "
                "migrated_launch_points has been made standalone and validated"
            ),
        },
        "per_repository": audit["totals"],
        "corpus_directories": corpus,
        "families": dict(sorted(families.items())),
        "launch_points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "inventory.json")
    args = parser.parse_args()
    inventory = build(args.audit)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory["counts"], indent=2))


if __name__ == "__main__":
    main()
