"""Pure-JAX reference for the gated delta rule over ragged sequences.

This is **upstream's own reference**, not a re-derived one:

  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py
  transformation: copied verbatim as the corpus baseline with source metadata
    and a smoke runner added; the implementation is unchanged. The file imports
    nothing outside jax.

Upstream describes it as "mainly for unit test", which is exactly the role a
corpus baseline plays. It is used here rather than a hand-written reference
because the contract has enough moving parts -- ragged sequence boundaries, a
null state block for padded tokens, a `has_initial_state` mask that zeroes
carried state for fresh prefills -- that an independently written version would
more likely be wrong than the kernels it is meant to check.

Contract ``ragged_gated_delta_rule``::

    mixed_qkv          [num_tokens, 2 * n_kq * d_k + n_v * d_v]
    b, a               [num_tokens, n_v]
    recurrent_state    [num_blocks, n_v, d_k, d_v]   block 0 is the null block
    A_log, dt_bias     [n_v]
    query_start_loc    [num_seqs + 1]    start index per sequence, last = total
    state_indices      [max_reqs]        request index -> state block
    distribution       [3] int32         (decode_end, prefill_end, mixed_end)
    has_initial_state  [max_reqs] bool   False = start from zero state
    ->                 (updated_recurrent_state, output[num_tokens, n_v * d_v])

`n_kq`, `n_v`, `d_k`, `d_v` are keyword-only and static.

The rule itself is a gated rank-one state update per token: `mixed_qkv` is
passed through SiLU and split into query/key/value, query and key are L2
normalized, and the state evolves as a decay-plus-correction recurrence driven
by the per-token gates derived from `a`, `b`, `A_log` and `dt_bias`.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py",
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "ragged_gated_delta_rule",
}

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ragged gated delta rule Ref implementation mainly for unit test."""

import dataclasses
import enum
import functools
from typing import Optional, Tuple

import jax
from jax import lax
import jax.numpy as jnp


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """L2 normalize along last dimension.

    Sum-of-squares and rsqrt run in fp32 even when ``x`` is bf16, to
    match GPU FLA's ``l2norm_fwd``.

    Args:
        x: input to normalize
        eps: epsilon for numerical stability

    Returns:
        normalized x, in the same dtype as `x`.
    """
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def _recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single-step recurrent update for decode.

    Args:
      query: Tensor of shape `(B, H, T, d_k)`. Current implementation assumes
        `T=1`.
      key: Tensor of shape `(B, H, T, d_k)`. Current implementation assumes `T=1`.
      value: Tensor of shape `(B, H, T, d_v)`. Current implementation assumes
        `T=1`.
      g: Tensor of shape `(B, H, T)`. Current implementation assumes `T=1`.
      beta: Tensor of shape `(B, H, T)`. Current implementation assumes `T=1`.
      state: Optional initial recurrent state of shape `(B, H, d_k, d_v)`.

    Returns:
      A tuple containing:
      - output: The output tensor of shape `(B, H, T, d_v)`.
      - new_state: The updated recurrent state of shape `(B, H, d_k, d_v)`.
    """
    B, H, T, d_k = query.shape
    d_v = value.shape[-1]

    if state is None:
        state = jnp.zeros((B, H, d_k, d_v), dtype=query.dtype)

    q = query[:, :, 0]
    k = key[:, :, 0]
    v = value[:, :, 0]
    beta_val = beta[:, :, 0]
    g_val = g[:, :, 0]

    scale = d_k**-0.5
    q = q * scale

    # v_diff = v - e^g * (k @ state)
    k_state = jnp.einsum("bhd, bhdm -> bhm", k, state)
    v_diff = v - jnp.exp(g_val)[..., None] * k_state

    # v_new = beta * v_diff
    v_new = beta_val[..., None] * v_diff

    # out = e^g * (q @ state) + (q . k) * v_new
    q_state = jnp.einsum("bhd, bhdm -> bhm", q, state)
    q_k = jnp.sum(q * k, axis=-1, keepdims=True)

    out = jnp.exp(g_val)[..., None] * q_state + q_k * v_new

    # s_new = state * exp(g) + k outer v_new
    k_v_new = jnp.einsum("bhd, bhm -> bhdm", k, v_new)
    new_state = state * jnp.exp(g_val)[..., None, None] + k_v_new

    return out[:, :, None, :], new_state


def ragged_gated_delta_rule(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Applies the gated delta rule over ragged sequences and updates recurrent state.

    Args:
      mixed_qkv: Combined QKV tensor of shape `(num_tokens, 2 * n_kq * d_k + n_v *
        d_v)`.
      b: B tensor of shape `(num_tokens, n_v)`.
      a: A tensor of shape `(num_tokens, n_v)`.
      recurrent_state: Recurrent state of shape `(num_blocks, n_v, d_k, d_v)`.
        `num_blocks` is always equal or larger than `max_seqs + 1`. The first
        block is a null_block and only used for padded / invalid tokens.
      A_log: Log of A parameter of shape `(n_v,)`.
      dt_bias: Delta T bias of shape `(n_v,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_reqs,)` mapping request index to state
        index.
      distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
        mixed_end)`.
      has_initial_state: Boolean tensor of shape `(max_reqs,)`. ``True`` when
        the request's slot already holds a valid recurrent state (chunked-
        prefill continuation, prefix-cache hit, or running decode);
        ``False`` for brand-new prefills, which must start from zero
        regardless of the slot's contents. Mirrors GPU's
        `initial_state[~has_initial_state, ...] = 0` in
        `gdn_linear_attn._forward_core`.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Dimension of key.
      d_v: Dimension of value.

    Returns:
      A tuple containing:
      - updated_recurrent_state: The updated recurrent state of shape
      `(num_blocks,
        n_v, d_k, d_v)`.
      - output: The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    mixed_qkv = jax.nn.silu(mixed_qkv)
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k
    query = mixed_qkv[..., :key_dim]
    key = mixed_qkv[..., key_dim:key_dim * 2]
    value = mixed_qkv[..., key_dim * 2:]
    max_reqs = state_indices.shape[0]
    token_idx = jnp.arange(num_tokens)

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(valid_loc_mask, query_start_loc,
                                          last_valid_loc)

    req_indices = (jnp.sum(
        token_idx[:, None] >= effective_query_start_loc[None, :], axis=1) - 1)
    req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
    valid_mask = token_idx < last_valid_loc

    # Zero the carry's recurrent state for slots whose request has no prior
    # context. We do this once up front so the scan can keep its simple
    # token-by-token shape: each step reads `recurrent_state_all[state_index]`,
    # which now holds zeros for new prefills regardless of what stale data
    # the slot may have held from a previous request. Mirrors GPU's
    # `initial_state[~has_initial_state, ...] = 0`.
    gathered_states = recurrent_state[state_indices]
    masked_initial_states = jnp.where(
        has_initial_state[:, None, None, None],
        gathered_states,
        jnp.zeros_like(gathered_states),
    )
    recurrent_state = recurrent_state.at[state_indices].set(
        masked_initial_states)

    def scan_fn(carry, xs):
        recurrent_state_all = carry
        (
            curr_q,
            curr_k,
            curr_v,
            curr_b,
            curr_a,
            request_index,
            is_valid_token,
        ) = xs

        curr_q = curr_q[None, None, :]
        curr_k = curr_k[None, None, :]
        curr_v = curr_v[None, None, :]
        curr_b = curr_b[None, None, :]
        curr_a = curr_a[None, None, :]

        state_index = state_indices[request_index]
        recurrent_state = recurrent_state_all[state_index][None, ...]

        B, T = 1, 1
        query_reshaped = curr_q.reshape(B, T, n_kq, d_k)
        key_reshaped = curr_k.reshape(B, T, n_kq, d_k)
        value_reshaped = curr_v.reshape(B, T, n_v, d_v)

        # Cast b to fp32 before sigmoid to match GPU's
        # `fused_gdn_gating_kernel`
        # (`vllm/model_executor/layers/mamba/gdn_linear_attn.py`).
        beta = jax.nn.sigmoid(curr_b.astype(jnp.float32))
        g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            curr_a.astype(jnp.float32) + dt_bias.astype(jnp.float32))

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            query_reshaped = jnp.repeat(query_reshaped, repeat_factor, axis=2)
            key_reshaped = jnp.repeat(key_reshaped, repeat_factor, axis=2)

        query_reshaped = jnp.transpose(query_reshaped,
                                       (0, 2, 1, 3)).astype(jnp.float32)
        key_reshaped = jnp.transpose(key_reshaped,
                                     (0, 2, 1, 3)).astype(jnp.float32)
        value_reshaped = jnp.transpose(value_reshaped,
                                       (0, 2, 1, 3)).astype(jnp.float32)
        beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
        g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

        query_reshaped = _l2_normalize(query_reshaped)
        key_reshaped = _l2_normalize(key_reshaped)

        output, new_recurrent_state = _recurrent_gated_delta_rule_step(
            query_reshaped,
            key_reshaped,
            value_reshaped,
            g,
            beta,
            state=recurrent_state,
        )

        output = jnp.transpose(output, (0, 2, 1, 3)).astype(query.dtype)
        output = output.reshape(B, T, -1)

        recurrent_state_all = jnp.where(
            is_valid_token,
            recurrent_state_all.at[state_index].set(
                new_recurrent_state[0].astype(recurrent_state_all.dtype)),
            recurrent_state_all,
        )

        return recurrent_state_all, output[0, 0]

    carry_init = recurrent_state
    xs = (query, key, value, b, a, req_indices, valid_mask)

    new_recurrent_state, output = jax.lax.scan(scan_fn, carry_init, xs)
    return new_recurrent_state, output


# --- corpus additions ---------------------------------------------------
# Everything above this line is upstream's file verbatim, except the import
# block, which merges the imports of both references — the same wording the
# sibling gated_linear_attention/baseline.py uses for the same situation.

# ## The SiLU precondition -- identical signatures, different contracts
#
# `ragged_gated_delta_rule` above and tpu-inference's v1 kernel of the same
# name take **byte-identical argument lists**, and disagree on what `mixed_qkv`
# means:
#
#   this reference        applies `jax.nn.silu(mixed_qkv)` as its first line
#   v1 kernel wrapper     documents `mixed_qkv` as "post-conv/silu" and does not
#
# Fed the same raw tensor, the two agree to a cosine of only **0.72**. Fed
# according to their preconditions, they agree to **0.9999976**. Nothing in the
# type signature or the shapes catches the difference -- both run, both return
# plausible-looking arrays -- so it is recorded here and enforced by a test.
#
# The corpus therefore treats these as one semantic contract with an explicit
# pre-silu boundary: `PRE_SILU_INPUT` names which side of it each entry point
# sits on, and `to_post_silu` converts.

#: Whether an entry point expects `mixed_qkv` *before* SiLU is applied.
#
# This is not uniform even within one repository. All four entry points below
# come from tpu-inference and take the same leading arguments, and they make
# three different choices:
#
#   reference + v2 recurrent_scan   apply SiLU themselves  -> pass raw
#   v1 ragged_gated_delta_rule      does not               -> pass post-SiLU
#   v2 decode_only                  explicit `apply_silu`  -> caller decides
#
# Each was determined by measurement, not by reading: the wrong choice still
# runs and still returns the right shapes. v1 fed raw scores cosine 0.72; v2's
# recurrent_scan fed post-SiLU scores 0.983 -- close enough to look like a
# tolerance problem rather than a contract error, which is exactly the trap.
PRE_SILU_INPUT = {
    "baseline.ragged_gated_delta_rule": True,
    "tpu_inference_v1.ragged_gated_delta_rule": False,
    "tpu_inference_v2.recurrent_scan": True,
    # Honours its `apply_silu` argument; True below means "pass raw and let
    # apply_silu=True do it", which is the default.
    "tpu_inference_v2.ragged_gated_delta_rule_decode_only": True,
}

#: `recurrent_scan` requires this; at False it returns cosine ~7e-4, i.e. it is
#: computing something else entirely rather than a slightly different result.
V2_RECURRENT_SCAN_REQUIRES_QK_NORM = True


def to_post_silu(mixed_qkv):
    """Convert a pre-SiLU `mixed_qkv` to what the v1 kernels expect."""
    return jax.nn.silu(mixed_qkv)


kernel = ragged_gated_delta_rule
workload = ragged_gated_delta_rule


def create_inputs(
    *,
    num_tokens: int = 256,
    num_seqs: int = 4,
    n_kq: int = 8,
    n_v: int = 16,
    d_k: int = 128,
    d_v: int = 128,
    dtype=None,
    seed: int = 0,
):
    """Inputs for ``ragged_gated_delta_rule`` at one ragged batch.

    ``n_v`` must be a multiple of ``n_kq``: the rule repeats each key/query head
    ``n_v // n_kq`` times, which is the GQA-style sharing this kernel assumes.

    Sequence boundaries are evenly spaced, and every request is given a state
    block **at or above 1** because block 0 of ``recurrent_state`` is the null
    block reserved for padded and invalid tokens.
    """
    import jax.numpy as jnp

    if n_v % n_kq:
        raise ValueError(f"{n_v=} must be a multiple of {n_kq=}")
    if num_tokens % num_seqs:
        raise ValueError(f"{num_tokens=} must be divisible by {num_seqs=}")
    dtype = dtype or jnp.float32

    keys = jax.random.split(jax.random.key(seed), 5)
    qkv_width = 2 * n_kq * d_k + n_v * d_v
    per_seq = num_tokens // num_seqs
    # `distribution` is (decode_end, prefill_end, mixed_end); the reference
    # only reads [2], the count of valid sequences.
    distribution = jnp.array([0, num_seqs, num_seqs], jnp.int32)
    num_blocks = num_seqs + 1

    return dict(
        mixed_qkv=jax.random.normal(keys[0], (num_tokens, qkv_width), dtype),
        b=jax.random.normal(keys[1], (num_tokens, n_v), dtype),
        a=jax.random.normal(keys[2], (num_tokens, n_v), dtype),
        recurrent_state=jax.random.normal(
            keys[3], (num_blocks, n_v, d_k, d_v), jnp.float32
        ),
        A_log=jax.random.normal(keys[4], (n_v,), jnp.float32),
        dt_bias=jnp.zeros((n_v,), jnp.float32),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * per_seq,
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        distribution=distribution,
        has_initial_state=jnp.ones((num_seqs,), jnp.bool_),
        static=dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v),
    )


def call(built: dict):
    """Apply the reference to a ``create_inputs`` dict."""
    static = built["static"]
    return ragged_gated_delta_rule(
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
        built["has_initial_state"], **static,
    )


# =======================================================================
# Contract 2: fused_conv1d_gated_delta_rule
# =======================================================================
# Everything below is Tokamax's own pure-JAX reference, copied verbatim,
# except that its `_recurrent_gated_delta_rule_step` is renamed with a
# `_tokamax_` prefix. Both repositories name that helper identically, so
# concatenating them made the second definition shadow the first and the
# tpu-inference reference above silently ran Tokamax's copy. The two are
# alpha-equivalent at these pinned commits, so nothing computed differently
# — but flatten_gdn.py records that this vendored pair shares 21 top-level
# names with only about a third to a half AST-identical, so a re-pin could
# have substituted a diverged helper without a word. flatten_gdn.py already
# applies exactly this guard when it concatenates the v1 modules; baseline.py
# never got it, because no tool assembles this file end to end.
#
#   repository: https://github.com/openxla/tokamax
#   commit:     927e3f94e8ffe0430cf38bd1423112bb2f69ec66
#   path:       tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule/
#               reference.py
#
# The v3 kernels fuse a depthwise **causal conv1d** with the gated delta rule
# and carry *two* caches -- a conv state and a recurrent state -- where the v1
# kernels carry one. That is a different contract, not a tuning variant, so it
# gets its own named reference here rather than sharing the one above.
#
# Entry point for the fused contract: `run_jax_gdn_attention_local_ref`.
# Tokamax's `ragged_gated_delta_rule_ref` also appears below; it is that
# repository's copy of the GDN-only reference and is kept for provenance, not
# used as the corpus baseline for contract 1.

SOURCE_FUSED = {
    "repository": "https://github.com/openxla/tokamax",
    "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
    "path": (
        "tokamax/_src/ops/experimental/causal_conv1d_gated_delta_rule/"
        "reference.py"
    ),
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "fused_conv1d_gated_delta_rule",
}

def l2norm_chunked(
    x: jnp.ndarray, dim: int = -1, eps: float = 1e-6
) -> jnp.ndarray:
  """Normalizes x along the specified dimension using L2 norm."""
  x_f32 = x.astype(jnp.float32)
  inv_norm = lax.rsqrt((x_f32 * x_f32).sum(axis=dim, keepdims=True) + eps)
  return (x_f32 * inv_norm).astype(x.dtype)


def l2_normalize_ref(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """L2 normalize along last dimension."""
  x_f32 = x.astype(jnp.float32)
  norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
  return (x_f32 / norm).astype(x.dtype)


def _tokamax_recurrent_gated_delta_rule_step(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Single-step recurrent update for decode."""
  batch_size, num_heads, _, d_k = query.shape
  d_v = value.shape[-1]

  if state is None:
    state = jnp.zeros((batch_size, num_heads, d_k, d_v), dtype=query.dtype)

  q = query[:, :, 0]
  k = key[:, :, 0]
  v = value[:, :, 0]
  beta_val = beta[:, :, 0]
  g_val = g[:, :, 0]

  scale = d_k**-0.5
  q = q * scale

  # v_diff = v - e^g * (k @ state)
  k_state = jnp.einsum("bhd, bhdm -> bhm", k, state)
  v_diff = v - jnp.exp(g_val)[..., None] * k_state

  # v_new = beta * v_diff
  v_new = beta_val[..., None] * v_diff

  # out = e^g * (q @ state) + (q . k) * v_new
  q_state = jnp.einsum("bhd, bhdm -> bhm", q, state)
  q_k = jnp.sum(q * k, axis=-1, keepdims=True)

  out = jnp.exp(g_val)[..., None] * q_state + q_k * v_new

  # s_new = state * exp(g) + k outer v_new
  k_v_new = jnp.einsum("bhd, bhm -> bhdm", k, v_new)
  new_state = state * jnp.exp(g_val)[..., None, None] + k_v_new

  return out[:, :, None, :], new_state


def ragged_gated_delta_rule_ref(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
  """Applies the gated delta rule over ragged sequences and updates recurrent state."""
  mixed_qkv = jax.nn.silu(mixed_qkv)
  num_tokens = mixed_qkv.shape[0]
  key_dim = n_kq * d_k
  query = mixed_qkv[..., :key_dim]
  key = mixed_qkv[..., key_dim : key_dim * 2]
  value = mixed_qkv[..., key_dim * 2 :]
  max_reqs = state_indices.shape[0]
  token_idx = jnp.arange(num_tokens)

  num_valid_seqs = distribution[2]
  valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
  last_valid_loc = query_start_loc[num_valid_seqs]
  effective_query_start_loc = jnp.where(
      valid_loc_mask, query_start_loc, last_valid_loc
  )

  req_indices = (
      jnp.sum(token_idx[:, None] >= effective_query_start_loc[None, :], axis=1)
      - 1
  )
  req_indices = jnp.clip(req_indices, 0, max_reqs - 1)
  valid_mask = token_idx < last_valid_loc

  gathered_states = recurrent_state[state_indices]
  masked_initial_states = jnp.where(
      has_initial_state[:, None, None, None],
      gathered_states,
      jnp.zeros_like(gathered_states),
  )
  recurrent_state = recurrent_state.at[state_indices].set(masked_initial_states)

  def scan_fn(carry, xs):
    recurrent_state_all = carry
    (
        curr_q,
        curr_k,
        curr_v,
        curr_b,
        curr_a,
        request_index,
        is_valid_token,
    ) = xs

    curr_q = curr_q[None, None, :]
    curr_k = curr_k[None, None, :]
    curr_v = curr_v[None, None, :]
    curr_b = curr_b[None, None, :]
    curr_a = curr_a[None, None, :]

    state_index = state_indices[request_index]
    recurrent_state = recurrent_state_all[state_index][None, ...]

    batch_size, num_steps = 1, 1
    query_reshaped = curr_q.reshape(batch_size, num_steps, n_kq, d_k)
    key_reshaped = curr_k.reshape(batch_size, num_steps, n_kq, d_k)
    value_reshaped = curr_v.reshape(batch_size, num_steps, n_v, d_v)

    beta = jax.nn.sigmoid(curr_b.astype(jnp.float32))
    g = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        curr_a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )

    repeat_factor = n_v // n_kq
    if repeat_factor > 1:
      query_reshaped = jnp.repeat(query_reshaped, repeat_factor, axis=2)
      key_reshaped = jnp.repeat(key_reshaped, repeat_factor, axis=2)

    query_reshaped = jnp.transpose(query_reshaped, (0, 2, 1, 3)).astype(
        jnp.float32
    )
    key_reshaped = jnp.transpose(key_reshaped, (0, 2, 1, 3)).astype(jnp.float32)
    value_reshaped = jnp.transpose(value_reshaped, (0, 2, 1, 3)).astype(
        jnp.float32
    )
    beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)
    g = jnp.transpose(g, (0, 2, 1)).astype(jnp.float32)

    query_reshaped = l2_normalize_ref(query_reshaped)
    key_reshaped = l2_normalize_ref(key_reshaped)

    output, new_recurrent_state = _tokamax_recurrent_gated_delta_rule_step(
        query_reshaped,
        key_reshaped,
        value_reshaped,
        g,
        beta,
        state=recurrent_state,
    )

    output = jnp.transpose(output, (0, 2, 1, 3)).astype(query.dtype)
    output = output.reshape(batch_size, num_steps, -1)

    recurrent_state_all = jnp.where(
        is_valid_token,
        recurrent_state_all.at[state_index].set(
            new_recurrent_state[0].astype(recurrent_state_all.dtype)
        ),
        recurrent_state_all,
    )

    return recurrent_state_all, output[0, 0]

  carry_init = recurrent_state
  xs = (query, key, value, b, a, req_indices, valid_mask)

  new_recurrent_state, output = lax.scan(scan_fn, carry_init, xs)
  return new_recurrent_state, output


def _fix_query_start_loc(query_start_loc, num_valid_seqs):
  """Fixes query_start_loc to be non-decreasing for invalid sequences."""
  last_valid_loc = query_start_loc[num_valid_seqs]
  valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
  return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _get_boundary_indices(starts, lengths, kernel_size, num_valid_seqs):
  """Computes indices for boundary fixup."""
  valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
  starts = jnp.where(valid_mask, starts, 1)[:, None]
  lengths = lengths[:, None]
  k_range = jnp.arange(kernel_size - 1)[None, :]
  gather_indices = starts + jnp.minimum(k_range, lengths - 1)
  scatter_indices = jnp.where(
      (k_range < lengths) & valid_mask[:, None],
      starts + k_range,
      -1,
  )
  return gather_indices, scatter_indices


def _get_state_update_indices(query_start_loc, kernel_size, num_tokens):
  """Computes indices for updating the convolutional state."""
  lengths = query_start_loc[1:] - query_start_loc[:-1]

  k_range = jnp.arange(kernel_size - 1)

  safe_idx_x = (
      query_start_loc[1:, None] - jnp.arange(kernel_size - 1, 0, -1)[None, :]
  )
  safe_idx_x = jnp.clip(safe_idx_x, 0, num_tokens - 1)

  is_from_old_state = k_range[None, :] < (kernel_size - 1 - lengths)[:, None]

  idx_g = k_range[None, :] + lengths[:, None]
  idx_g = jnp.clip(idx_g, 0, kernel_size - 2)

  return safe_idx_x, is_from_old_state, idx_g


def _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias):
  """Depthwise 1D convolution using loops over kernel size."""
  num_tokens = x.shape[0]
  kernel_size = conv_weight.shape[-1]
  out = None

  padded_x = jnp.pad(x, ((kernel_size - 1, 0), (0, 0)))

  for k in range(kernel_size):
    x_slice = padded_x[k : k + num_tokens, :].astype(jnp.float32)
    weight_slice = conv_weight[:, 0, k].astype(jnp.float32)
    if out is None:
      if conv_bias is None:
        out = x_slice * weight_slice
      else:
        out = x_slice * weight_slice + conv_bias[jnp.newaxis, :]
    else:
      out += x_slice * weight_slice

  assert out is not None
  return out.astype(x.dtype)


def ragged_conv1d_mixed_prefill(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
  """Applies 1D convolution, optimized for prefill."""
  num_tokens = x.shape[0]
  max_blocks = state_indices.shape[0]
  num_valid_seqs = distribution[2]

  out = _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias)

  query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
  starts = query_start_loc[:-1]
  lengths = query_start_loc[1:] - query_start_loc[:-1]
  gather_indices, scatter_indices = _get_boundary_indices(
      starts, lengths, kernel_size, num_valid_seqs
  )
  x_first = x[gather_indices]

  gathered_state = conv_state[state_indices]

  gathered_state = jnp.where(
      has_initial_state[:, None, None],
      gathered_state,
      jnp.zeros_like(gathered_state),
  )

  combined_tokens = jnp.concatenate([gathered_state, x_first], axis=1)

  b_out = lax.conv_general_dilated(
      combined_tokens,
      conv_weight,
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "OIW", "NWC"),
      feature_group_count=x.shape[-1],
      precision=lax.Precision.HIGHEST,
  ).reshape(-1, x.shape[-1])
  if conv_bias is not None:
    b_out += conv_bias[jnp.newaxis, :]

  out = out.at[scatter_indices.flatten()].set(
      b_out.astype(out.dtype), mode="drop", wrap_negative_indices=False
  )
  total_valid_tokens = query_start_loc[num_valid_seqs]
  valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
  out = jnp.where(valid_token_mask[:, jnp.newaxis], out, 0.0)

  true_valid_seq_mask = jnp.arange(max_blocks) < num_valid_seqs
  safe_idx_x, is_from_old_state, idx_g = _get_state_update_indices(
      query_start_loc, kernel_size, num_tokens
  )

  x_tokens = x[safe_idx_x]
  r_grid = jnp.arange(max_blocks)[:, None]
  state_tokens = gathered_state[r_grid, idx_g]

  new_state_extracted = jnp.where(
      is_from_old_state[..., None], state_tokens, x_tokens
  )

  updated_conv_state = conv_state.at[state_indices].set(
      jnp.where(
          true_valid_seq_mask[:, None, None],
          new_state_extracted,
          conv_state[state_indices],
      )
  )

  return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_decode_only(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
  """Apply conv1d for decode-only case."""
  num_tokens = x.shape[0]

  token_idx = jnp.arange(num_tokens)
  req_state_indices = state_indices[token_idx]
  gathered_state = conv_state[req_state_indices]

  lhs = jnp.concatenate([gathered_state, x[:, jnp.newaxis, :]], axis=1)

  out = jnp.einsum(
      "nkd,dk->nd",
      lhs,
      conv_weight[:, 0, :],
      precision=lax.Precision.HIGHEST,
  )

  if conv_bias is not None:
    out = out + conv_bias

  num_valid_seqs = distribution[2]

  new_state_extracted = jnp.concatenate(
      [gathered_state[:, 1:, :], x[:, jnp.newaxis, :]], axis=1
  )

  token_idx = jnp.arange(num_tokens)
  valid_mask = token_idx < num_valid_seqs
  states_to_set = jnp.where(
      valid_mask[:, jnp.newaxis, jnp.newaxis],
      new_state_extracted,
      gathered_state,
  )

  updated_conv_state = conv_state.at[req_state_indices].set(states_to_set)

  out = jnp.where(valid_mask[:, jnp.newaxis], out, 0.0)

  return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_jax(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Applies 1D convolution over ragged sequences and updates state."""
  is_decode_only = distribution[0] == distribution[2]

  def decode_only_branch(_):
    return ragged_conv1d_decode_only(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

  def mixed_prefill_branch(_):
    return ragged_conv1d_mixed_prefill(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

  return jax.lax.cond(
      is_decode_only, decode_only_branch, mixed_prefill_branch, operand=None
  )


class RaggedConv1dImpl(enum.Enum):
  JAX = "ragged_conv1d_jax"


class RaggedGatedDeltaRuleImpl(enum.Enum):
  REF = "ragged_gated_delta_rule_ref"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
  ragged_conv1d_impl: RaggedConv1dImpl = RaggedConv1dImpl.JAX
  ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
      RaggedGatedDeltaRuleImpl.REF
  )


def run_jax_gdn_attention_local_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Runs the local JAX GDN attention mechanism with combined QKV tensors."""
  max_reqs = seq_lens.shape[0]
  query_lens = query_start_loc[1 : max_reqs + 1] - query_start_loc[:max_reqs]
  has_initial_state = (seq_lens - query_lens) > 0

  conv_impl = ragged_conv1d_jax

  out_mixed_qkv, new_conv_state = conv_impl(
      qkv,
      conv_state,
      conv_weight,
      conv_bias,
      query_start_loc,
      state_indices,
      distribution,
      has_initial_state,
      kernel_size=kernel_size,
  )

  ragged_gdn_impl = functools.partial(
      ragged_gated_delta_rule_ref,
      has_initial_state=has_initial_state,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
  )
  new_recurrent_state, output = ragged_gdn_impl(
      out_mixed_qkv,
      b,
      a,
      recurrent_state,
      a_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
  )

  return (new_conv_state, new_recurrent_state), output

def create_inputs_fused(
    *,
    num_tokens: int = 256,
    num_seqs: int = 4,
    n_kq: int = 8,
    n_v: int = 16,
    d_k: int = 128,
    d_v: int = 128,
    kernel_size: int = 4,
    seed: int = 0,
):
    """Inputs for ``fused_conv1d_gated_delta_rule`` (the v3 contract).

    Adds the conv1d half that contract 1 has no notion of: a `conv_state` cache
    holding the previous ``kernel_size - 1`` positions per slot, plus the
    depthwise weight and bias. As with the recurrent state, slot 0 is the null
    block for padded and invalid tokens.
    """
    if n_v % n_kq:
        raise ValueError(f"{n_v=} must be a multiple of {n_kq=}")
    if num_tokens % num_seqs:
        raise ValueError(f"{num_tokens=} must be divisible by {num_seqs=}")

    keys = jax.random.split(jax.random.key(seed), 8)
    dim_size = 2 * n_kq * d_k + n_v * d_v
    per_seq = num_tokens // num_seqs
    num_blocks = num_seqs + 1

    return dict(
        qkv=jax.random.normal(keys[0], (num_tokens, dim_size), jnp.float32),
        b=jax.random.normal(keys[1], (num_tokens, n_v), jnp.float32),
        a=jax.random.normal(keys[2], (num_tokens, n_v), jnp.float32),
        conv_state=jax.random.normal(
            keys[3], (num_blocks, kernel_size - 1, dim_size), jnp.float32
        ),
        recurrent_state=jax.random.normal(
            keys[4], (num_blocks, n_v, d_k, d_v), jnp.float32
        ),
        # Depthwise: one length-`kernel_size` filter per channel. The
        # docstring in upstream's wrapper says `[kernel_size - 1, dim_size]`,
        # but its own assertion demands `(dim, 1, kernel_size)` -- the
        # assertion is what runs, so that is what is built here.
        conv_weight=jax.random.normal(
            keys[5], (dim_size, 1, kernel_size), jnp.float32
        ) * 0.1,
        conv_bias=jnp.zeros((dim_size,), jnp.float32),
        a_log=jax.random.normal(keys[6], (n_v,), jnp.float32),
        dt_bias=jnp.zeros((n_v,), jnp.float32),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * per_seq,
        state_indices=jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        distribution=jnp.array([0, num_seqs, num_seqs], jnp.int32),
        seq_lens=jnp.full((num_seqs,), per_seq, jnp.int32),
        static=dict(
            n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, kernel_size=kernel_size
        ),
    )


FUSED_ARG_ORDER = (
    "qkv", "b", "a", "conv_state", "recurrent_state", "conv_weight",
    "conv_bias", "a_log", "dt_bias", "query_start_loc", "state_indices",
    "distribution", "seq_lens",
)


def fused_args(built: dict) -> tuple:
    """Positional arguments shared by the fused reference and the v3 kernels."""
    return tuple(built[name] for name in FUSED_ARG_ORDER)


def main() -> None:
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--n-kq", type=int, default=8)
    parser.add_argument("--n-v", type=int, default=16)
    parser.add_argument("--d-k", type=int, default=128)
    parser.add_argument("--d-v", type=int, default=128)
    args = parser.parse_args()

    built = create_inputs(
        num_tokens=args.num_tokens, num_seqs=args.num_seqs,
        n_kq=args.n_kq, n_v=args.n_v, d_k=args.d_k, d_v=args.d_v,
    )
    static = built["static"]
    compiled = jax.jit(ragged_gated_delta_rule, static_argnames=tuple(static))
    start = time.perf_counter()
    state, output = compiled(
        built["mixed_qkv"], built["b"], built["a"], built["recurrent_state"],
        built["A_log"], built["dt_bias"], built["query_start_loc"],
        built["state_indices"], built["distribution"],
        built["has_initial_state"], **static,
    )
    jax.block_until_ready((state, output))
    print(json.dumps({
        "implementation": "baseline",
        "contract": "ragged_gated_delta_rule",
        "state_shape": list(state.shape),
        "output_shape": list(output.shape),
        "dtype": str(output.dtype),
        "compile_and_run_ms": (time.perf_counter() - start) * 1e3,
    }))


if __name__ == "__main__":
    main()
