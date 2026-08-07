"""Pure-JAX references for the gated linear attention kernels here.

Both are **upstream's own**, copied verbatim from sglang-jax rather than
re-derived -- the same choice made in gated_delta_net, and for the same reason:
a recurrence with ragged sequence boundaries and a carried state is easy to
write subtly wrong, and a wrong reference produces false failures rather than
catching real ones.

Two contracts live in this directory, and they are not variants of each other:

kda_chunk_fwd -- Kimi Delta Attention. Per-channel gates g, a beta
delta term, and an optional L2 normalisation of q/k inside the kernel::

    naive_recurrent_kda(q, k, v, g, beta, scale, initial_state,
                        output_final_state)

simple_gla_fwd -- a lighter recurrence with a single scalar decay per head
(g_gamma) and no delta term::

    naive_gla_prefill(q, k, v, g_gamma, h0, cu_seqlens, scale)
    naive_gla_decode(q, k, v, g_gamma, h0, scale)

Simple GLA splits prefill and decode into two references because upstream
splits the kernels the same way -- the fused decode path lives in its own
module.

Provenance
----------
  repository: https://github.com/sgl-project/sglang-jax
  commit:     a7353325e8c00d287294c2cd679a77173f1a4594
  paths:      python/sgl_jax/srt/kernels/kda/naive.py
              python/sgl_jax/srt/kernels/simple_gla/native.py
  transformation: both files copied verbatim; only their imports were merged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

SOURCE_KDA = {
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/kernels/kda/naive.py",
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "kda_chunk_fwd",
}

SOURCE_SIMPLE_GLA = {
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/kernels/simple_gla/native.py",
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "simple_gla_fwd",
}


# =====================================================================
# Contract 1: kda_chunk_fwd  (from kda/naive.py)
# =====================================================================
def acc_dtype(input_dtype) -> jnp.dtype:
    """Accumulator dtype: fp64 for fp64 inputs, fp32 otherwise."""
    return jnp.float64 if input_dtype == jnp.float64 else jnp.float32


def naive_recurrent_kda(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """
    Core recurrence (per timestep):
        S' = S_{t-1} * exp(g_t)                            decay
        residual = v_t - k_t^T @ S'                        prediction error
        S_t = S' + beta_t * k_t ⊗ residual                 delta update
        o_t = (q_t * scale)^T @ S_t                        output

    Dtype behavior (matching FLA):
      - All inputs cast to fp32 for computation
      - Hidden state S is fp32 accumulator
      - Output o computed in fp32, cast back to original dtype
      - Final state S stays in fp32
      - fp64 mode: all computation in fp64, no precision cast

    Args:
        q:               [B, T, H, K] — Queries
        k:               [B, T, H, K] — Keys
        v:               [B, T, H, V] — Values
        g:               [B, T, H, K] — Per-element gate in log-space (e.g., -exp(A)*softplus(g))
        beta:            [B, T, H]    — Learning rate / step size for delta rule
        scale:           Scalar query scale. Defaults to K ** -0.5.
        initial_state:   [B, H, K, V] — Initial hidden state. Optional.
        output_final_state: Whether to return the final hidden state.

    Returns:
        o:           [B, T, H, V] — Output (original input dtype)
        final_state: [B, H, K, V] in fp32 (or fp64), or None
    """
    orig_dtype, acc_dt = v.dtype, acc_dtype(q.dtype)

    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert (
        v.ndim == 4 and v.shape[:3] == q.shape[:3]
    ), f"v shape {v.shape} incompatible with q shape {q.shape}"
    assert g.ndim == 4 and g.shape == q.shape, f"g shape {g.shape} != q shape {q.shape}"
    assert beta.ndim == 3 and beta.shape == q.shape[:3], f"beta shape {beta.shape} != {q.shape[:3]}"

    B, T, H, K, V = *q.shape, v.shape[-1]

    if initial_state is not None:
        assert initial_state.shape == (
            B,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != ({B}, {H}, {K}, {V})"

    if scale is None:
        scale = K**-0.5

    # [B, T, H, K] -> [B, H, T, K], cast to acc_dt
    q, k, v, g = (jnp.transpose(x, (0, 2, 1, 3)).astype(acc_dt) for x in (q, k, v, g))
    # q: [B, H, T, K]   k: [B, H, T, K]   v: [B, H, T, V]   g: [B, H, T, K]

    # [B, T, H] -> [B, H, T]
    beta = jnp.transpose(beta, (0, 2, 1)).astype(acc_dt)  # [B, H, T]

    q = q * scale  # [B, H, T, K]

    S = jnp.zeros((B, H, K, V), dtype=acc_dt)  # [B, H, K, V] hidden state
    if initial_state is not None:
        S += initial_state.astype(acc_dt)  # [B, H, K, V]
    o = jnp.zeros((B, H, T, V), dtype=acc_dt)  # [B, H, T, V] output buffer

    for i in range(T):
        q_i = q[:, :, i]  # [B, H, K]
        k_i = k[:, :, i]  # [B, H, K]
        v_i = v[:, :, i]  # [B, H, V]
        g_i = g[:, :, i]  # [B, H, K]
        b_i = beta[:, :, i]  # [B, H]

        # 1. Decay the state
        # exp(g_i): [B, H, K] -> [B, H, K, 1] via broadcast
        S = S * jnp.exp(g_i)[..., None]  # [B, H, K, V]

        # 2. Delta rule update
        # k_i[..., None]: [B, H, K, 1],  k_i[..., None] * S: [B, H, K, V]
        v_predicted = (k_i[..., None] * S).sum(-2)  # [B, H, V]
        residual = v_i - v_predicted  # [B, H, V]

        # b_i[..., None] * k_i: [B, H, K],  einsum -> [B, H, K, V]
        S = S + jnp.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, residual)  # [B, H, K, V]

        # 3. Compute output: einsum [B,H,K] x [B,H,K,V] -> [B, H, V]
        o = o.at[:, :, i].set(jnp.einsum("bhk,bhkv->bhv", q_i, S))  # [B, H, V]

    final_state = S if output_final_state else None  # [B, H, K, V] or None
    # [B, H, T, V] -> [B, T, H, V], cast back to orig_dtype
    return jnp.transpose(o, (0, 2, 1, 3)).astype(orig_dtype), final_state


# =====================================================================
# Contract 2: simple_gla_fwd  (from simple_gla/native.py)
# =====================================================================
def naive_gla_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    h0: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Naive GLA decode using jnp.einsum.

    Args:
        q: Query tensor [B, 1, H, K]
        k: Key tensor [B, 1, H, K]
        v: Value tensor [B, 1, H, K]
        g_gamma: Gate decay per head [H], negative values (e.g., ALiBi slopes)
        h0: Initial state [B, H, K, K]
        scale: Optional output scaling factor

    Returns:
        output: [B, 1, H, K]
        h1: Updated state [B, H, K, K]
    """
    B, T, H, K = q.shape
    assert T == 1, f"Decode expects T=1, got {T}"

    q_t = q[:, 0].astype(jnp.float32)
    k_t = k[:, 0].astype(jnp.float32)
    v_t = v[:, 0].astype(jnp.float32)
    g_gamma = g_gamma.astype(jnp.float32)
    h0 = h0.astype(jnp.float32)

    if scale is None:
        scale = K**-0.5

    decay = jnp.exp(g_gamma)[None, :, None, None]
    kv = jnp.einsum("bhk,bhv->bhkv", k_t, v_t)
    h1 = decay * h0 + kv
    o = jnp.einsum("bhk,bhkv->bhv", q_t, h1)
    o = o * scale

    output = o[:, None, :, :]

    return output, h1


def naive_gla_prefill(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g_gamma: jax.Array,
    h0: jax.Array,
    cu_seqlens: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Naive GLA prefill using per-request scan + jnp.einsum.

    Args:
        q: Query tensor [1, T_total, H, K] (varlen packed)
        k: Key tensor [1, T_total, H, K] (varlen packed)
        v: Value tensor [1, T_total, H, K] (varlen packed)
        g_gamma: Gate decay per head [H], negative values
        h0: Initial state per request [B, H, K, K]
        cu_seqlens: Cumulative sequence lengths [B+1], e.g., [0, 128, 384] for 2 requests
        scale: Optional output scaling factor

    Returns:
        output: [1, T_total, H, K]
        h_final: Final state per request [B, H, K, K]
    """
    assert q.shape[0] == 1, f"Prefill expects batch=1 (varlen), got {q.shape[0]}"

    q = q[0].astype(jnp.float32)
    k = k[0].astype(jnp.float32)
    v = v[0].astype(jnp.float32)
    g_gamma = g_gamma.astype(jnp.float32)
    h0 = h0.astype(jnp.float32)

    T = q.shape[0]
    _, K = q.shape[1], q.shape[2]

    if scale is None:
        scale = K**-0.5

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    reset_mask = token_idx == cu_seqlens[:-1][seq_ids]
    decay = jnp.exp(g_gamma)

    def scan_fn(carry, inputs):
        h_prev, final_states = carry
        seq_id, do_reset, q_t, k_t, v_t = inputs
        h = jnp.where(do_reset, h0[seq_id], h_prev)
        kv = jnp.einsum("hk,hv->hkv", k_t, v_t)
        h = decay[:, None, None] * h + kv
        o_t = jnp.einsum("hk,hkv->hv", q_t, h)
        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_t

    init_carry = (
        jnp.zeros_like(h0[0]),
        h0,
    )
    (_, h_final), output = jax.lax.scan(
        scan_fn,
        init_carry,
        (seq_ids, reset_mask, q, k, v),
    )

    output = output * scale

    return output[None, :, :, :], h_final


__all__ = ["naive_gla_decode", "naive_gla_prefill"]
