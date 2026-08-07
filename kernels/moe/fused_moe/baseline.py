"""JAX references for the fused MoE kernels in this directory.

Every function here is upstream's own.  All three implementations ship a
pure-JAX ``ref_moe`` in the same file as their kernel, and all three of
upstream's test suites compare against it; this module extracts each one, with
its two or three helpers, and renames them per source because all three are
called ``ref_moe``.

Carrying them rather than writing one matters more here than almost anywhere
else in this corpus.  "Mixture of experts" does not name a single function: the
routing can be plain top-k or grouped top-k, the logits can be softmaxed or
sigmoided or used raw, the top-k weights can be renormalised or not, the
activation can be silu / gelu / clamped SwiGLU, the weights can be sub-channel
or per-channel quantised, and a shared expert may or may not be added on top.
Upstream's three files disagree with each other on several of those.  A
corpus-authored reference would be picking one combination and calling it "the"
MoE.

Source:
  https://github.com/sgl-project/sglang-jax
    commit: a7353325e8c00d287294c2cd679a77173f1a4594
    path: python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py
  https://github.com/sgl-project/sglang-jax
    commit: a7353325e8c00d287294c2cd679a77173f1a4594
    path: python/sgl_jax/test/kernels/fused_moe_v1_test.py
  https://github.com/sgl-project/sglang-jax
    commit: a7353325e8c00d287294c2cd679a77173f1a4594
    path: python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py
  https://github.com/sgl-project/sglang-jax
    commit: a7353325e8c00d287294c2cd679a77173f1a4594
    path: python/sgl_jax/test/kernels/fused_moe_v2_test.py
  https://github.com/vllm-project/tpu-inference
    commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
    path: tpu_inference/kernels/fused_moe/v1/kernel.py
  https://github.com/vllm-project/tpu-inference
    commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
    path: tests/kernels/fused_moe_v1_test.py
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("fused_ep_moe_split_weights", "fused_ep_moe_split_weights_v2",
                  "fused_ep_moe_fused_w1"),
    "sources": [{'repository': 'https://github.com/sgl-project/sglang-jax', 'commit': 'a7353325e8c00d287294c2cd679a77173f1a4594', 'path': 'python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py'}, {'repository': 'https://github.com/sgl-project/sglang-jax', 'commit': 'a7353325e8c00d287294c2cd679a77173f1a4594', 'path': 'python/sgl_jax/test/kernels/fused_moe_v1_test.py'}, {'repository': 'https://github.com/sgl-project/sglang-jax', 'commit': 'a7353325e8c00d287294c2cd679a77173f1a4594', 'path': 'python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py'}, {'repository': 'https://github.com/sgl-project/sglang-jax', 'commit': 'a7353325e8c00d287294c2cd679a77173f1a4594', 'path': 'python/sgl_jax/test/kernels/fused_moe_v2_test.py'}, {'repository': 'https://github.com/vllm-project/tpu-inference', 'commit': '8b9c90928c94c7230d1bc891534a301510a6a30d', 'path': 'tpu_inference/kernels/fused_moe/v1/kernel.py'}, {'repository': 'https://github.com/vllm-project/tpu-inference', 'commit': '8b9c90928c94c7230d1bc891534a301510a6a30d', 'path': 'tests/kernels/fused_moe_v1_test.py'}],
}

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

# ---- extracted from sglang-jax (v1): python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py ----
# swigluoai, activation_fn, ref_moe, renamed with a `_sglang_jax` suffix.

def swigluoai_sglang_jax(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def activation_fn_sglang_jax(acc1, acc3, act_fn):
    if act_fn == "silu":
        return jax.nn.silu(acc1) * acc3
    elif act_fn == "gelu":
        return jax.nn.gelu(acc1) * acc3
    elif act_fn == "swigluoai":
        return swigluoai_sglang_jax(acc1, acc3)
    else:
        raise RuntimeError(f"Unsupported activation function: {act_fn}")


def ref_moe_sglang_jax(
    tokens: jax.Array,  # (num_tokens, hidden_size)
    w1: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
    w3: jax.Array,  # (num_experts, hidden_size, intermediate_size)
    gating_output: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    *,
    use_grouped_topk: bool = False,
    num_groups: int = 1,
    top_k_groups: int = 1,
    bias: jax.Array | None = None,
    renormalize_topk_logits: bool = False,
    routed_scaling_factor: float | None = None,
    act_fn: str = "silu",
    quant_block_k: int | None = None,
    w1_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // quant_block_k, scale_dim2, scale_dim3)
    w2_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, intermediate_size // quant_block_k, scale_dim2, scale_dim3)
    w3_scale: (
        jax.Array | None
    ) = None,  # F32(num_experts, hidden_size // quant_block_k, scale_dim2, scale_dim3)
    b1: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
    b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
    b3: jax.Array | None = None,  # F32(num_experts, 1, intermediate_size)
    w1_shared: jax.Array | None = None,  # (hidden_size, se_intermediate_size) [Gate]
    w2_shared: jax.Array | None = None,  # (se_intermediate_size, hidden_size) [Down]
    w3_shared: jax.Array | None = None,  # (hidden_size, se_intermediate_size) [Up]
    # NOTE: Shared-expert weights use per-column scaling (axis=0 quantization in
    # `quantize_tensor`), so these scales are (1, 1, out_features).
    w1_shared_scale: jax.Array | None = None,  # (1, 1, se_inter)
    w2_shared_scale: jax.Array | None = None,  # (1, 1, hidden_size)
    w3_shared_scale: jax.Array | None = None,  # (1, 1, se_inter)
):
    n_tokens = tokens.shape[0]  # num_tokens
    num_experts = gating_output.shape[-1]

    # Compute gating scores for all experts
    gating_logits_f32 = gating_output.astype(jnp.float32)

    routing_scores = (
        gating_logits_f32 + jnp.expand_dims(bias.astype(jnp.float32), 0)
        if bias is not None
        else gating_logits_f32
    )

    if use_grouped_topk:
        assert num_experts % num_groups == 0
        experts_per_group = num_experts // num_groups
        reshaped_routing_scores = routing_scores.reshape(n_tokens, num_groups, experts_per_group)

        if bias is not None:
            top2_vals, _ = lax.top_k(reshaped_routing_scores, 2)
            group_scores = jnp.sum(top2_vals, axis=-1)
        else:
            group_scores = jnp.max(reshaped_routing_scores, axis=-1)

        group_mask_accum = jnp.zeros((n_tokens, num_groups), dtype=jnp.bool_)
        temp_group_scores = group_scores
        group_iota = jax.lax.broadcasted_iota(jnp.int32, (n_tokens, num_groups), 1)

        for _ in range(top_k_groups):
            curr_max_group_idx = jnp.argmax(temp_group_scores, axis=1, keepdims=True)
            curr_mask = group_iota == curr_max_group_idx
            group_mask_accum = jnp.logical_or(group_mask_accum, curr_mask)
            temp_group_scores = jnp.where(curr_mask, -jnp.float32(jnp.inf), temp_group_scores)

        expert_mask = jnp.repeat(
            jnp.expand_dims(group_mask_accum, axis=2), experts_per_group, axis=2
        ).reshape(n_tokens, num_experts)

        routing_scores = jnp.where(expert_mask, routing_scores, -jnp.float32(jnp.inf))

    # Select top-k experts per token
    _, top_k_indices = lax.top_k(routing_scores, top_k)
    top_k_logits = jnp.take_along_axis(gating_logits_f32, top_k_indices, axis=-1)

    if renormalize_topk_logits:
        top_k_logits = top_k_logits / jnp.sum(top_k_logits, axis=-1, keepdims=True)

    if routed_scaling_factor is not None:
        top_k_logits *= routed_scaling_factor

    t_outputs = []
    hidden_size, intermediate_size = w1.shape[-2:]

    # Process each token individually
    for i in range(n_tokens):
        curr_token = jnp.expand_dims(tokens[i], axis=0)  # [1, hidden_size]
        assigned_expert_ids = top_k_indices[i]  # [top_k] - indices of selected experts for token i
        tok_expert_act = []

        # Process each selected expert for the current token
        for expert_id in assigned_expert_ids:
            # Get expert weights
            expert_w1 = w1[expert_id].astype(jnp.float32)
            expert_w3 = w3[expert_id].astype(jnp.float32)
            if w1_scale is not None:
                expert_w1 *= jnp.repeat(w1_scale[expert_id, :, 0], quant_block_k, axis=0)[
                    :hidden_size
                ]
            if w3_scale is not None:
                expert_w3 *= jnp.repeat(w3_scale[expert_id, :, 0], quant_block_k, axis=0)[
                    :hidden_size
                ]
            expert_weight_2 = w2[expert_id].astype(jnp.float32)  # [intermediate_size, hidden_size]
            if w2_scale is not None:
                expert_weight_2 *= jnp.repeat(w2_scale[expert_id, :, 0], quant_block_k, axis=0)[
                    :intermediate_size
                ]

            # First linear layer (gate/up projections).
            gmm1_w1_proj = curr_token @ expert_w1  # [1, intermediate_size]
            gmm1_w3_proj = curr_token @ expert_w3  # [1, intermediate_size]
            if b1 is not None:
                gmm1_w1_proj += b1[expert_id : expert_id + 1, 0]
            if b3 is not None:
                gmm1_w3_proj += b3[expert_id : expert_id + 1, 0]

            # Apply gated activation: activation(gate) * up
            act = activation_fn_sglang_jax(gmm1_w1_proj, gmm1_w3_proj, act_fn)

            # Second linear layer (down projection)
            gmm_2_out = act @ expert_weight_2  # [1, hidden_size]
            if b2 is not None:
                gmm_2_out += b2[expert_id : expert_id + 1, 0]
            tok_expert_act.append(gmm_2_out)

        # Combine outputs from all selected experts
        experts_act = jnp.concatenate(tok_expert_act, axis=0)  # [top_k, hidden_size]

        # Weighted sum using top-k gating weights
        top_k_weights = top_k_logits[i]  # [top_k]
        top_k_weights = jnp.expand_dims(top_k_weights, axis=1)  # [top_k, 1]
        weighted_output = jnp.sum(
            experts_act * top_k_weights, axis=0, keepdims=True
        )  # [1, hidden_size]

        t_outputs.append(weighted_output.astype(tokens.dtype))

    moe_output = jnp.concatenate(t_outputs, axis=0)  # [actual_num_tokens, hidden_size]

    if w1_shared is not None:
        se_w1_gate = w1_shared.astype(jnp.float32)
        se_w1_up = w3_shared.astype(jnp.float32)

        if w1_shared_scale is not None:
            assert w1_shared_scale.shape == (1, 1, w1_shared.shape[1]), w1_shared_scale.shape
            se_w1_gate *= w1_shared_scale[0, 0, :][None, :]

        if w3_shared_scale is not None:
            assert w3_shared_scale.shape == (1, 1, w3_shared.shape[1]), w3_shared_scale.shape
            se_w1_up *= w3_shared_scale[0, 0, :][None, :]

        gate_out = tokens.astype(jnp.float32) @ se_w1_gate
        up_out = tokens.astype(jnp.float32) @ se_w1_up

        act = activation_fn_sglang_jax(gate_out, up_out, act_fn)

        se_w2 = w2_shared.astype(jnp.float32)
        if w2_shared_scale is not None:
            assert w2_shared_scale.shape == (1, 1, w2_shared.shape[1]), w2_shared_scale.shape
            se_w2 *= w2_shared_scale[0, 0, :][None, :]

        se_output = act @ se_w2

        return moe_output + se_output.astype(moe_output.dtype)

    return moe_output


# ---- extracted from sglang-jax (v1): python/sgl_jax/test/kernels/fused_moe_v1_test.py ----
# gen_moe_inputs, renamed with a `_sglang_jax` suffix.

def gen_moe_inputs_sglang_jax(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
    has_shared_expert=False,
    se_intermediate_size=None,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 12)
    k0, k1, k2, k3, k4, k5, k6, k7, k8 = keys[:9]

    a = jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype) / 10

    w1 = (
        jax.random.normal(k1, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w2 = (
        jax.random.normal(k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w3 = (
        jax.random.normal(k3, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)

    if has_bias:
        b1 = (
            jax.random.normal(k4, (num_experts, 1, intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        b2 = (jax.random.normal(k5, (num_experts, 1, hidden_size), dtype=jnp.float32) / 10).astype(
            dtype
        )
        b3 = (
            jax.random.normal(k6, (num_experts, 1, intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
    else:
        b1 = b2 = b3 = None

    # Shared Expert Weights
    w1_shared = w2_shared = w3_shared = None
    if has_shared_expert:
        if se_intermediate_size is None:
            se_intermediate_size = intermediate_size

        k9, k10, k11 = keys[9:]
        w1_shared = (
            jax.random.normal(k9, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w2_shared = (
            jax.random.normal(k10, (se_intermediate_size, hidden_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w3_shared = (
            jax.random.normal(k11, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)

    # Construct gating logits with deterministic, strictly-ordered top-k per token.
    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)

    # Generate unique top-k indices per token (sample without replacement).
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)

    # Add a strictly decreasing boost so top-1 > top-2 > ... > top-k
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    gating_output = (gating_output + one_hot).astype(dtype)

    return a, w1, w2, w3, b1, b2, b3, gating_output, w1_shared, w2_shared, w3_shared


# ---- extracted from sglang-jax (v2): python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py ----
# swigluoai, activation_fn, ref_moe, renamed with a `_sglang_jax_v2` suffix.

def swigluoai_sglang_jax_v2(
    gate: jax.Array, up: jax.Array, *, alpha: float = 1.702, limit: float = 7.0
) -> jax.Array:
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def activation_fn_sglang_jax_v2(acc1, acc3, act_fn, swiglu_limit=None):
    if act_fn == "swigluoai":
        return swigluoai_sglang_jax_v2(acc1, acc3)
    if act_fn == "silu":
        act = jax.nn.silu(acc1)
    elif act_fn == "gelu":
        act = jax.nn.gelu(acc1)
    else:
        raise RuntimeError(f"Unsupported activation: {act_fn}")
    # Optional SwiGLU clamp (e.g. maxtext-trained models such as Ling3-Flash clamp
    # the gated activation on their late layers). Cap the post-activation gate
    # single-sided and the up branch double-sided before the multiply. None =
    # disabled (default), preserving prior behavior bit-for-bit.
    if swiglu_limit is not None:
        act = jnp.clip(act, max=swiglu_limit)
        acc3 = jnp.clip(acc3, min=-swiglu_limit, max=swiglu_limit)
    return act * acc3


def ref_moe_sglang_jax_v2(
    tokens,
    w1,
    w2,
    w3,
    topk_weights,
    topk_ids,
    top_k,
    *,
    act_fn="silu",
    swiglu_limit=None,
    shared_swiglu_limit=None,
    w1_shared=None,
    w2_shared=None,
    w3_shared=None,
    w1_shared_scale=None,
    w2_shared_scale=None,
    w3_shared_scale=None,
    quant_block_k=None,
    w1_scale=None,
    w2_scale=None,
    w3_scale=None,
):
    num_tokens = tokens.shape[0]
    hidden_size = tokens.shape[1]
    num_experts = w1.shape[0]

    tokens_f32 = tokens.astype(jnp.float32)
    output = jnp.zeros_like(tokens_f32)

    def _dequant(w, scale, qbk):
        if scale is None:
            return w.astype(jnp.float32)
        w_f32 = w.astype(jnp.float32)
        if qbk is None:
            return w_f32 * scale.squeeze(1)
        s = jnp.repeat(scale, qbk, axis=0).squeeze(1)
        return w_f32 * s

    for t_id in range(num_tokens):
        for k_id in range(top_k):
            e_id = int(topk_ids[t_id, k_id])
            if e_id < 0 or e_id >= num_experts:
                continue
            weight = float(topk_weights[t_id, k_id])
            x = tokens_f32[t_id : t_id + 1]
            gate = x @ _dequant(
                w1[e_id], w1_scale[e_id] if w1_scale is not None else None, quant_block_k
            )
            up = x @ _dequant(
                w3[e_id], w3_scale[e_id] if w3_scale is not None else None, quant_block_k
            )
            act = activation_fn_sglang_jax_v2(gate, up, act_fn, swiglu_limit)
            out = act @ _dequant(
                w2[e_id], w2_scale[e_id] if w2_scale is not None else None, quant_block_k
            )
            output = output.at[t_id].add(out[0] * weight)

    if w1_shared is not None:

        def _deq_se(w, sc):
            wf = w.astype(jnp.float32)
            return wf if sc is None else wf * jnp.asarray(sc).astype(jnp.float32)

        gate_se = tokens_f32 @ _deq_se(w1_shared, w1_shared_scale)
        up_se = tokens_f32 @ _deq_se(w3_shared, w3_shared_scale)
        act_se = activation_fn_sglang_jax_v2(gate_se, up_se, act_fn, shared_swiglu_limit)
        out_se = act_se @ _deq_se(w2_shared, w2_shared_scale)
        output = output + out_se

    return output.astype(tokens.dtype)


# ---- extracted from sglang-jax (v2): python/sgl_jax/test/kernels/fused_moe_v2_test.py ----
# gen_moe_inputs, renamed with a `_sglang_jax_v2` suffix.

def gen_moe_inputs_sglang_jax_v2(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_shared_expert=False,
    se_intermediate_size=None,
):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 12)
    k0, k1, k2, k3, k7, k8 = keys[0], keys[1], keys[2], keys[3], keys[7], keys[8]

    a = jax.random.normal(k0, (num_tokens, hidden_size), dtype=jnp.float32).astype(dtype) / 10
    w1 = (
        jax.random.normal(k1, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w2 = (
        jax.random.normal(k2, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32) / 10
    ).astype(dtype)
    w3 = (
        jax.random.normal(k3, (num_experts, hidden_size, intermediate_size), dtype=jnp.float32) / 10
    ).astype(dtype)

    w1_shared = w2_shared = w3_shared = None
    if has_shared_expert:
        if se_intermediate_size is None:
            se_intermediate_size = intermediate_size
        k9, k10, k11 = keys[9], keys[10], keys[11]
        w1_shared = (
            jax.random.normal(k9, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w2_shared = (
            jax.random.normal(k10, (se_intermediate_size, hidden_size), dtype=jnp.float32) / 10
        ).astype(dtype)
        w3_shared = (
            jax.random.normal(k11, (hidden_size, se_intermediate_size), dtype=jnp.float32) / 10
        ).astype(dtype)

    # Strictly-ordered, deterministic top-k per token (top-1 > top-2 > ...).
    gating_output = jax.random.normal(k7, (num_tokens, num_experts), dtype=jnp.float32)
    token_keys = jax.random.split(k8, num_tokens)
    top_k_indices = jax.vmap(lambda kk: jax.random.permutation(kk, num_experts)[:top_k])(
        token_keys
    ).astype(jnp.int32)
    boosts = (30.0 - jnp.arange(top_k, dtype=jnp.float32)).reshape(1, top_k)
    one_hot = jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32) * boosts[..., None],
        axis=1,
    )
    gating_output = (gating_output + one_hot).astype(dtype)
    return a, w1, w2, w3, gating_output, w1_shared, w2_shared, w3_shared


# ---- extracted from vLLM tpu-inference (v1): tpu_inference/kernels/fused_moe/v1/kernel.py ----
# apply_scoring_fn, swigluoai, apply_act_fn, ref_moe, renamed with a `_tpu_inference` suffix.

def apply_scoring_fn_tpu_inference(scoring_fn: str, x):
    match scoring_fn:
        case "softmax":
            return jax.nn.softmax(x, axis=-1)
        case "sigmoid":
            # TODO(catswe): use jax.nn.sigmoid once mosaic lowering bug with bf16 input is fixed
            return 1 / (1 + jnp.exp(-x))
        case _:
            raise NotImplementedError(
                f"Unsupported scoring function: {scoring_fn}")


def swigluoai_tpu_inference(gate: jax.Array,
              up: jax.Array,
              *,
              alpha: float = 1.702,
              limit: float = 7.0) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""
    gate = jnp.clip(gate, max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def apply_act_fn_tpu_inference(acc1, acc3, act_fn):
    if act_fn == "silu":
        return jax.nn.silu(acc1) * acc3
    elif act_fn == "gelu":
        return jax.nn.gelu(acc1) * acc3
    elif act_fn == "swigluoai":
        return swigluoai_tpu_inference(acc1, acc3)
    else:
        raise NotImplementedError(f"Unsupported activation function: {act_fn}")


def ref_moe_tpu_inference(
        tokens: jax.Array,  # (num_tokens, hidden_size)
        w1: jax.Array,  # (num_experts, 2, hidden_size, intermediate_size)
        w2: jax.Array,  # (num_experts, intermediate_size, hidden_size)
        gating_output: jax.Array,  # (num_tokens, num_experts)
        top_k: int,
        *,
        renormalize_topk_logits: bool = False,
        act_fn: str = "silu",
        scoring_fn: str = "softmax",
        subc_quant_w1_sz: int | None = None,
        subc_quant_w2_sz: int | None = None,
        w1_scale:
    (
        jax.Array | None
    ) = None,  # F32(num_experts, 2, hidden_size //subc_quant_w1_sz, 1, intermediate_size)
        w2_scale:
    (
        jax.Array | None
    ) = None,  # F32(num_experts, intermediate_size // subc_quant_w2_sz, 1, hidden_size)
        b1: jax.Array
    | None = None,  # F32(num_experts, 2, 1, intermediate_size)
        b2: jax.Array | None = None,  # F32(num_experts, 1, hidden_size)
):
    n_tokens = tokens.shape[0]  # num_tokens

    # Compute gating scores for all experts
    gating_logits = apply_scoring_fn_tpu_inference(scoring_fn,
                                     gating_output)  # [num_tokens, n_experts]

    # Select top-k experts per token
    top_k_logits, top_k_indices = lax.top_k(
        gating_logits, top_k)  # [num_tokens, top_k], [num_tokens, top_k]

    if renormalize_topk_logits:
        top_k_logits = top_k_logits / jnp.sum(
            top_k_logits, axis=-1, keepdims=True)

    t_outputs = []
    hidden_size, intermediate_size = w1.shape[-2:]

    # Process each token individually
    for i in range(n_tokens):
        curr_token = jnp.expand_dims(tokens[i], axis=0)  # [1, hidden_size]
        assigned_expert_ids = top_k_indices[
            i]  # [top_k] - indices of selected experts for token i
        tok_expert_act = []

        # Process each selected expert for the current token
        for expert_id in assigned_expert_ids:
            # Get expert weights
            expert_w1 = w1[expert_id, 0].astype(jnp.float32)
            expert_w3 = w1[expert_id, 1].astype(jnp.float32)
            if w1_scale is not None:
                assert subc_quant_w1_sz is not None
                expert_w1 *= jnp.repeat(w1_scale[expert_id, 0, :, 0],
                                        subc_quant_w1_sz,
                                        axis=0)[:hidden_size]
                expert_w3 *= jnp.repeat(w1_scale[expert_id, 1, :, 0],
                                        subc_quant_w1_sz,
                                        axis=0)[:hidden_size]
            expert_weight_1 = jnp.concat(
                [expert_w1, expert_w3],
                axis=-1)  # [hidden_size, 2 * intermediate_size]
            expert_weight_2 = w2[expert_id].astype(
                jnp.float32)  # [intermediate_size, hidden_size]
            if w2_scale is not None:
                assert subc_quant_w2_sz is not None
                expert_weight_2 *= jnp.repeat(w2_scale[expert_id, :, 0],
                                              subc_quant_w2_sz,
                                              axis=0)[:intermediate_size]

            # First linear layer with SwiGLU activation
            gmm_1_out = curr_token @ expert_weight_1  # [1, 2 * intermediate_size]

            # Split into gate and up projections for SwiGLU
            gmm1_w1_proj, gmm1_w3_proj = jnp.split(
                gmm_1_out, 2,
                axis=-1)  # [1, intermediate_size], [1, intermediate_size]
            if b1 is not None:
                gmm1_w1_proj += b1[expert_id:expert_id + 1, 0, 0]
                gmm1_w3_proj += b1[expert_id:expert_id + 1, 1, 0]

            # Apply gated activation: activation(gate) * up
            act = apply_act_fn_tpu_inference(gmm1_w1_proj, gmm1_w3_proj, act_fn)

            # Second linear layer (down projection)
            gmm_2_out = act @ expert_weight_2  # [1, hidden_size]
            if b2 is not None:
                gmm_2_out += b2[expert_id:expert_id + 1, 0]
            tok_expert_act.append(gmm_2_out)

        # Combine outputs from all selected experts
        experts_act = jnp.concatenate(tok_expert_act,
                                      axis=0)  # [top_k, hidden_size]

        # Weighted sum using top-k gating weights
        top_k_weights = top_k_logits[i]  # [top_k]
        top_k_weights = jnp.expand_dims(top_k_weights, axis=1)  # [top_k, 1]
        weighted_output = jnp.sum(experts_act * top_k_weights,
                                  axis=0,
                                  keepdims=True)  # [1, hidden_size]

        t_outputs.append(weighted_output.astype(tokens.dtype))

    return jnp.concatenate(t_outputs,
                           axis=0)


# ---- extracted from vLLM tpu-inference (v1): tests/kernels/fused_moe_v1_test.py ----
# gen_moe_inputs, renamed with a `_tpu_inference` suffix.

def gen_moe_inputs_tpu_inference(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
):
    key = jax.random.key(seed)
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

    a = (jax.random.normal(k0, (num_tokens, hidden_size),
                           dtype=jnp.bfloat16).astype(dtype) / 10)

    w1 = (jax.random.normal(
        k1,
        (num_experts, 2, hidden_size, intermediate_size),
        dtype=jnp.bfloat16,
    ) / 10).astype(dtype)
    w2 = (jax.random.normal(k2, (num_experts, intermediate_size, hidden_size),
                            dtype=jnp.bfloat16) / 10).astype(dtype)

    if has_bias:
        b1 = (jax.random.normal(k3, (num_experts, 2, 1, intermediate_size),
                                dtype=jnp.bfloat16) / 10).astype(dtype)
        b2 = (jax.random.normal(k4, (num_experts, 1, hidden_size),
                                dtype=jnp.bfloat16) / 10).astype(dtype)
    else:
        b1 = b2 = None

    gating_output = (
        jax.random.normal(k5, (num_tokens, num_experts), dtype=jnp.bfloat16) +
        jnp.arange(num_tokens * num_experts, dtype=jnp.bfloat16).reshape(
            num_tokens, num_experts) / 100)

    # To generate unique top-k!
    top_k_indices = jax.random.randint(k6, (num_tokens, top_k),
                                       minval=0,
                                       maxval=num_experts - 1,
                                       dtype=jnp.int32)

    one_hot = (jnp.sum(
        jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.bfloat16),
        axis=1,
    ) * 30)

    gating_output = (gating_output + one_hot).astype(dtype)

    return a, w1, w2, b1, b2, gating_output


#: The corpus protocol: one named reference per contract.
REFERENCES = {
    "fused_ep_moe_split_weights": ref_moe_sglang_jax,
    "fused_ep_moe_split_weights_v2": ref_moe_sglang_jax_v2,
    "fused_ep_moe_fused_w1": ref_moe_tpu_inference,
}
