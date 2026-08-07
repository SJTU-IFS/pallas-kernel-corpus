"""Standalone sglang-jax Simple GLA kernels.

Source:
  repository: https://github.com/sgl-project/sglang-jax
  commit: a7353325e8c00d287294c2cd679a77173f1a4594
  path: python/sgl_jax/srt/kernels/simple_gla/
  files: ('simple_gla.py', 'simple_gla_fused.py')
  transformation: the module files above flattened into one file in dependency order, with repo-local imports dropped

Entry point: ``simple_gla_fwd`` (also exported as ``kernel``).

Contract ``simple_gla_fwd`` -- **3 Pallas launch points**.
Simple GLA: one scalar decay `g_gamma` per head rather than KDA's
per-channel gates. Three launch points -- two in the chunked
prefill path and one in the fused decode path, which upstream
keeps in a separate module that imports the first.

Validated against upstream's own pure-JAX reference, which is copied into
``baseline.py`` from the same directory.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/kernels/simple_gla",
    "files": ('simple_gla.py', 'simple_gla_fused.py'),
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "simple_gla_fwd",
    "launch_points": 3,
}

from functools import singledispatch
import enum
import functools
import inspect as _inspect
import math
import os

import jax
from jax.experimental.pallas import dslice
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

# --- from simple_gla.py -------------------------------------------
# =============================================================================
# Utilities (from tops/utils.py and tops/ops/utils.py)
# =============================================================================


def assert_shape_or_none(
    x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if x is None:
        return
    if isinstance(x, list | tuple):
        has_names = isinstance(name, list | tuple) and len(name) == len(x)
        for i, tensor in enumerate(x):
            if tensor is not None:
                curr_name = name[i] if has_names else f"{name}_{i}"
                assert (
                    tensor.shape == expected_shape
                ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape(
    x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
    if isinstance(x, list | tuple):
        has_names = isinstance(name, list | tuple) and len(name) == len(x)
        for i, tensor in enumerate(x):
            curr_name = name[i] if has_names else f"{name}_{i}"
            assert (
                tensor.shape == expected_shape
            ), f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
    else:
        assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def exp(x):
    return jnp.exp(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


# =============================================================================
# Fused recurrent (from tops/ops/simple_gla/fused_recurrent.py)
# Pure JAX implementation using jax.lax.scan, decode-friendly.
# =============================================================================


def _scan_segment(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over one dense segment."""
    if reverse:
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        if g is not None:
            g = jnp.flip(g, axis=1)

    B, _T, H, K = q.shape
    V = v.shape[-1]
    h0 = initial_state if initial_state is not None else jnp.zeros((B, H, K, V), dtype=q.dtype)

    q_t = jnp.swapaxes(q, 0, 1)
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)
    g_t = jnp.swapaxes(g, 0, 1) if g is not None else jnp.zeros((q_t.shape[0], B, H), dtype=q.dtype)
    use_g = g is not None

    def step(h, xs):
        q_i, k_i, v_i, g_i = xs
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma[None, :]
        else:
            decay = jnp.broadcast_to(g_gamma[None, :], (B, H))

        h = h * jnp.exp(decay)[:, :, None, None]
        h = h + k_i[:, :, :, None] * v_i[:, :, None, :]
        o_i = jnp.sum(h * (q_i[:, :, :, None] * scale), axis=2)
        return h, o_i

    h_final, o_t = jax.lax.scan(step, h0, (q_t, k_t, v_t, g_t))
    o = jnp.swapaxes(o_t, 0, 1)

    if reverse:
        o = jnp.flip(o, axis=1)

    return o, h_final


def _scan_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None,
    g_gamma: jax.Array | None,
    scale: float,
    initial_state: jax.Array | None,
    reverse: bool,
    cu_seqlens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run recurrent Simple GLA over packed varlen data with one JAX scan."""
    _B, T, H, K = q.shape
    V = v.shape[-1]
    N = cu_seqlens.shape[0] - 1

    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    seq_starts = cu_seqlens[:-1]
    seq_ends = cu_seqlens[1:]
    token_starts = seq_starts[seq_ids]
    token_ends = seq_ends[seq_ids]

    if reverse:
        scan_order = token_ends - 1 - (token_idx - token_starts)
        reset_mask = token_idx == (token_ends - 1)
    else:
        scan_order = token_idx
        reset_mask = token_idx == token_starts

    scan_seq_ids = seq_ids[scan_order]
    q_s = q[0, scan_order]
    k_s = k[0, scan_order]
    v_s = v[0, scan_order]
    g_s = g[0, scan_order] if g is not None else jnp.zeros((T, H), dtype=q.dtype)

    h0_all = initial_state if initial_state is not None else jnp.zeros((N, H, K, V), dtype=q.dtype)
    use_g = g is not None

    def step(carry, xs):
        h_prev, final_states = carry
        seq_id, do_reset, q_i, k_i, v_i, g_i = xs

        h = jnp.where(do_reset, h0_all[seq_id], h_prev)
        if use_g:
            decay = g_i
            if g_gamma is not None:
                decay = decay + g_gamma
        else:
            decay = g_gamma

        h = h * jnp.exp(decay)[:, None, None]
        h = h + k_i[:, :, None] * v_i[:, None, :]
        o_i = jnp.sum(h * (q_i[:, :, None] * scale), axis=1)

        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_i

    init_carry = (
        jnp.zeros((H, K, V), dtype=q.dtype),
        h0_all,
    )
    (h_last, final_states), o_scan = jax.lax.scan(
        step,
        init_carry,
        (scan_seq_ids, reset_mask[scan_order], q_s, k_s, v_s, g_s),
    )
    del h_last

    inv_order = jnp.argsort(scan_order)
    o = o_scan[inv_order][None, ...]
    return o, final_states


def fused_recurrent_simple_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: np.ndarray | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Simple GLA fused recurrent forward for decode-friendly execution.

    Args:
        q: [B, T, H, K] queries.
        k: [B, T, H, K] keys.
        v: [B, T, H, V] values.
        g: [B, T, H] optional per-token log gate.
        g_gamma: [H] optional per-head constant log decay.
        scale: Optional query scaling factor. Defaults to K ** -0.5.
        initial_state: [N, H, K, V] optional recurrent state, where N=B for dense
            mode and N=len(cu_seqlens)-1 for varlen mode.
        output_final_state: Whether to return the final recurrent state.
        reverse: Whether to process each sequence in reverse time order.
        cu_seqlens: [N+1] cumulative sequence lengths for packed varlen inputs.

    Returns:
        Tuple of output [B, T, H, V] in q.dtype and optional final state
        [N, H, K, V] in the input dtype.
    """
    assert q.ndim == 4, f"q must be 4D [B,T,H,K], got {q.ndim}D"
    assert v.ndim == 4, f"v must be 4D [B,T,H,V], got {v.ndim}D"

    B, T, H, K = q.shape
    V = v.shape[-1]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    assert k.shape == q.shape, f"k shape {k.shape} != q shape {q.shape}"
    assert v.shape[:3] == q.shape[:3], f"v shape {v.shape} incompatible with q"
    assert g is not None or g_gamma is not None, "At least one of g or g_gamma must be provided"
    if g is not None:
        assert g.ndim == 3 and g.shape == (B, T, H), f"g shape {g.shape} != {(B, T, H)}"
    if g_gamma is not None:
        assert (
            g_gamma.ndim == 1 and g_gamma.shape[0] == H
        ), f"g_gamma shape {g_gamma.shape} != ({H},)"
    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
    if initial_state is not None:
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"

    if scale is None:
        scale = K**-0.5
    scale = float(scale)

    q_f = q
    k_f = k
    v_f = v
    g_f = g
    g_gamma_f = g_gamma
    h0_f = initial_state

    if cu_seqlens is None:
        o, ht = _scan_segment(
            q_f,
            k_f,
            v_f,
            g=g_f,
            g_gamma=g_gamma_f,
            scale=scale,
            initial_state=h0_f,
            reverse=reverse,
        )
        return o, (ht if output_final_state else None)

    cu_f = jnp.asarray(cu_seqlens, dtype=jnp.int32)
    o, ht = _scan_varlen(
        q_f,
        k_f,
        v_f,
        g=g_f,
        g_gamma=g_gamma_f,
        scale=scale,
        initial_state=h0_f,
        reverse=reverse,
        cu_seqlens=cu_f,
    )
    return o, (ht if output_final_state else None)


# =============================================================================
# Chunk forward h — varlen path (from tops/ops/common/chunk_h.py)
# Pallas TPU kernel for computing hidden states with variable-length sequences.
# =============================================================================


def _build_chunk_map(cu_seqlens, T_sum, BT):
    NT = T_sum // BT
    chunk_ids = lax.iota(jnp.int32, NT)
    chunk_pos = chunk_ids * BT
    N = cu_seqlens.shape[-1] - 1
    seq_idx = jnp.searchsorted(cu_seqlens[1:], chunk_pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    return seq_idx


def _chunk_fwd_h_kernel_varlen(
    k_ref,  # [1, BT, BK]
    v_ref,  # [1, BT, BV]
    h0_ref,  # [N, 1, BK, BV]
    gk_ref,  # [1, BT, BK]
    g_gamma_ref,  # [H,]
    cu_seqlens_ref,  # [num_seq+1]
    chunk_to_seq,  # [T_sum/BT]
    seq_real_lens_ref,  # [N] real (non-padded) seq length, or None
    h_ref,  # [NS, 1, BK, BV]
    ht_ref,  # [N, 1, BK , BV]
    scratch_ref,  # [BK, BV]
    *,
    BT,
    BS,
):
    BT, BK = k_ref.shape[1], k_ref.shape[2]
    BV = v_ref.shape[2]

    NTS = BS // BT
    b_h_start = jnp.zeros((BK, BV), dtype=jnp.float32)

    i_h, _i_k, _i_v, i_t = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)

    if g_gamma_ref is not None:
        b_g = g_gamma_ref[i_h].astype(jnp.float32) * (jnp.arange(0, BT) + 1)
    t0 = i_t * BT

    seq_idx = chunk_to_seq[i_t]

    bos = cu_seqlens_ref[seq_idx]
    eos = cu_seqlens_ref[seq_idx + 1]

    @pl.when(bos != eos)
    def _():
        # reset h state
        @pl.when(t0 == bos)
        def reset_state():
            if h0_ref is not None:
                scratch_ref[...] = h0_ref[seq_idx, 0].astype(scratch_ref.dtype)
            else:
                scratch_ref[...] = b_h_start

        # store intermediate state
        @pl.when(i_t % NTS == 0)
        def store_fn():
            s_i = i_t // NTS
            h_ref[s_i, 0] = scratch_ref[...].astype(h_ref.dtype)
            return None

        k_tile = k_ref[(0, slice(None), slice(None))]  # [BT,BK]
        v_tile = v_ref[(0, slice(None), slice(None))]  # [BT,BV]

        if g_gamma_ref is not None:
            # Use real (non-padded) length when seq_real_lens is provided so b_g_last
            # only covers real tokens — sequences padded to chunk_size internally
            # would otherwise accumulate decay over zero-padding tail.
            if seq_real_lens_ref is not None:
                real_eos = bos + seq_real_lens_ref[seq_idx]
                effective_remaining = jnp.maximum(real_eos - t0, 0)
            else:
                effective_remaining = eos - t0
            # tpu not support scalar bf16 mul
            L_chunk = jnp.minimum(BT, effective_remaining)
            b_g_last = (g_gamma_ref[i_h].astype(jnp.float32) * L_chunk).astype(g_gamma_ref.dtype)
            scratch_ref[...] *= exp(b_g_last)
            # Mask exponent to avoid NaN (0 * inf) in padding positions
            v_decay_exp = b_g_last - b_g
            v_decay_exp = jnp.where(jnp.arange(BT) < L_chunk, v_decay_exp, -1e9)
            v_tile = (v_tile * exp(v_decay_exp)[:, None]).astype(v_tile.dtype)

        if gk_ref is not None:
            gk_tile = gk_ref[(0, slice(None), slice(None))]  # BT * BK
            g_last = gk_tile[-1, :]
            decay = exp(g_last)
            scratch_ref[...] = scratch_ref[...] * decay[:, None]  # [BK, BV] * [BK,1]
            k_tile = (k_tile * exp(g_last[None, :] - gk_tile)).astype(k_tile.dtype)

        # state update
        scratch_ref[...] = scratch_ref[...] + jax.lax.dot(
            k_tile.astype(jnp.float32).T,
            v_tile.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

        @pl.when(t0 + BT >= eos)
        def write_final():
            if ht_ref is not None:
                ht_ref[seq_idx, 0] = scratch_ref[...].astype(jnp.float32)


@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "chunk_size",
        "split_size",
        "states_in_fp32",
    ],
)
def chunk_fwd_h_kernel_varlen(
    k: jax.Array,  # [B,T,H,K]
    v: jax.Array,  # [B,T,H,V]
    g: jax.Array | None = None,  # [B,T,H]
    g_gamma: jax.Array | None = None,  # (H,)
    gk: jax.Array | None = None,  # [B,T,H,K]
    gv: jax.Array | None = None,  # [B,T,H,V]
    h0: jax.Array | None = None,  # [N,H,K,V]
    output_final_state: bool = False,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 128,
    split_size: int | None = None,
    states_in_fp32: bool = False,
    seq_real_lens: jax.Array | None = None,  # [N]
):
    interpret = get_interpret()
    assert g is None, "g should be None."
    assert gv is None, "gv should be None."
    BK = 128
    BV = 128
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert K % 128 == 0, "K % 128 must equal to 0."
    assert V % 128 == 0, "V % 128 must equal to 0."
    assert T % chunk_size == 0, "T mod chunk_size must equal to 0."

    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"

    T_sum = B * T
    chunk_to_seq = _build_chunk_map(cu_seqlens=cu_seqlens_dev, T_sum=T_sum, BT=BT)

    N, NS = (
        len(cu_seqlens_dev) - 1,
        T_sum // BS,
    )

    k = jnp.reshape(k, (T_sum, H, K))
    v = jnp.reshape(v, (T_sum, H, V))

    k = jnp.transpose(k, (1, 0, 2))  # (H,B*T,K)
    v = jnp.transpose(v, (1, 0, 2))  # (H,B*T,V)
    if gk is not None:
        gk = jnp.reshape(gk, (T_sum, H, K))
        gk = jnp.transpose(gk, (1, 0, 2))  # (H,B*T,K)

    grid = (H, pl.cdiv(K, BK), pl.cdiv(V, BV), T_sum // BT)

    def k_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def gk_index_map(head_index, k_index, _, t_index):
        return head_index, t_index, k_index

    def v_index_map(head_index, _, v_index, t_index):
        return head_index, t_index, v_index

    def h0_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def ht_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    def h_index_map(head_index, k_index, v_index, t_index):
        return 0, head_index, k_index, v_index

    out_shape = [
        jax.ShapeDtypeStruct(
            shape=(NS, H, K, V), dtype=k.dtype if not states_in_fp32 else jnp.float32
        )
    ]
    out_specs = [pl.BlockSpec((NS, 1, BK, BV), h_index_map)]
    if output_final_state:
        out_shape.append(jax.ShapeDtypeStruct(shape=(N, H, K, V), dtype=jnp.float32))
        out_specs.append(pl.BlockSpec((N, 1, BK, BV), ht_index_map))
    else:
        out_shape.append(None)
        out_specs.append(None)

    in_specs = [
        pl.BlockSpec((1, BT, BK), k_index_map),
        pl.BlockSpec((1, BT, BV), v_index_map),
    ]
    if h0 is not None:
        in_specs.append(pl.BlockSpec((N, 1, BK, BV), h0_index_map))
    else:
        in_specs.append(None)
    if gk is not None:
        in_specs.append(pl.BlockSpec((1, BT, BK), gk_index_map))
    else:
        in_specs.append(None)

    if g_gamma is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)

    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    if seq_real_lens is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    else:
        in_specs.append(None)
    scratch = pltpu.VMEM((BK, BV), jnp.float32)
    scratch_shapes = [scratch]
    kernel = functools.partial(
        _chunk_fwd_h_kernel_varlen,
        BT=BT,
        BS=BS,
    )
    h, ht = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=128 * 1024 * 1024,
        ),
    )(k, v, h0, gk, g_gamma, cu_seqlens_dev, chunk_to_seq, seq_real_lens)
    if output_final_state:
        return h, ht
    return h, None


# =============================================================================
# Chunk forward o (from tops/ops/common/chunk_o.py)
# Pallas TPU kernel for computing chunk output.
# =============================================================================


def _chunk_fwd_o_kernel(
    q_ref,
    k_ref,
    v_ref,
    h_ref,
    g_ref,
    g_gamma_ref,
    scale_ref,
    o_ref,
    *,
    BT: int,
):
    """Pallas kernel for chunk_fwd_o.

    Grid: (H, total_NT, num_v_tiles)
    Refs (after block spec indexing):
      q_ref/k_ref: (1, 1, BT, K)
      v_ref: (1, 1, BT, BV)
      h_ref: (1, 1, K, BV)
      g_ref: (1, 1, BT, 128) or None  (broadcast to 4D for TPU alignment)
      g_gamma_ref: [H] via SMEM or ANY
      scale_ref: (1,) via SMEM or ANY
      o_ref: (1, 1, BT, BV)
    """
    b_q = q_ref[0, 0]  # (BT, K)
    b_k = k_ref[0, 0]  # (BT, K)
    b_v = v_ref[0, 0]  # (BT, BV)
    b_h = h_ref[0, 0]  # (K, BV)

    b_o = jnp.dot(
        b_q,
        b_h,
        preferred_element_type=jnp.float32,
    )
    b_A = jnp.dot(
        b_q,
        b_k.T,
        preferred_element_type=jnp.float32,
    )

    if g_ref is not None:
        b_g = g_ref[0, 0, :, 0].astype(jnp.float32)  # (BT,)
        b_o = b_o * exp(b_g)[:, None]
        g_diff = b_g[:, None] - b_g[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_diff = jnp.where(fwd_mask, g_diff, 0.0)
        b_A = b_A * exp(safe_g_diff)

    if g_gamma_ref is not None:
        head_idx = pl.program_id(0)
        b_gamma = g_gamma_ref[head_idx].astype(jnp.float32)
        b_g_gamma = b_gamma * (jnp.arange(BT) + 1).astype(jnp.float32)
        b_o = b_o * exp(b_g_gamma)[:, None]
        g_gamma_diff = b_g_gamma[:, None] - b_g_gamma[None, :]
        fwd_mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
        safe_g_gamma_diff = jnp.where(fwd_mask, g_gamma_diff, 0.0)
        b_A = b_A * exp(safe_g_gamma_diff)

    mask = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    b_A = jnp.where(mask, b_A, 0.0)
    scale = scale_ref[0].astype(jnp.float32)

    # Keep b_A in fp32 for precision; upcast b_v instead.
    b_o = (
        b_o * scale
        + jnp.dot(
            b_A,
            b_v.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        * scale
    )
    o_ref[0, 0] = b_o.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("chunk_size",),
)
def _chunk_fwd_o_pl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float,
    chunk_size: int = 64,
) -> jax.Array:
    """Pallas launcher for chunk_fwd_o on the uniform-length path."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    total_NT = B * NT

    def _reshape_bt(x, D):
        return x.reshape(B, NT, BT, H, D).transpose(3, 0, 1, 2, 4).reshape(H, total_NT, BT, D)

    _q = _reshape_bt(q, K)  # (H, total_NT, BT, K)
    _k = _reshape_bt(k, K)  # (H, total_NT, BT, K)
    _v = _reshape_bt(v, V)  # (H, total_NT, BT, V)
    _h = h.reshape(B, NT, H, K, V).transpose(2, 0, 1, 3, 4).reshape(H, total_NT, K, V)
    _g = None
    if g is not None:
        _g = g.reshape(B, NT, BT, H).transpose(3, 0, 1, 2).reshape(H, total_NT, BT)
        _g = jnp.broadcast_to(_g[:, :, :, None], (H, total_NT, BT, 128))  # 4D for TPU alignment

    BV = 128 if V % 128 == 0 else V
    num_v_tiles = V // BV

    if num_v_tiles > 1:
        # Split V into tiles and merge with H: (H, ..., V) -> (H*num_v_tiles, ..., BV)
        _v = (
            _v.reshape(H, total_NT, BT, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, BT, BV)
        )
        _h = (
            _h.reshape(H, total_NT, K, num_v_tiles, BV)
            .transpose(0, 3, 1, 2, 4)
            .reshape(H * num_v_tiles, total_NT, K, BV)
        )
        # g_gamma: repeat each head value for its V-tiles
        if g_gamma is not None:
            g_gamma = jnp.repeat(g_gamma, num_v_tiles)  # (H * num_v_tiles,)

    H_VT = H * num_v_tiles
    grid = (H_VT, total_NT)

    # q/k/g index by head = hv_idx // num_v_tiles; v/h index by hv_idx directly
    spec_qk = pl.BlockSpec(
        (1, 1, BT, K), index_map=lambda hv_idx, nt_idx: (hv_idx // num_v_tiles, nt_idx, 0, 0)
    )
    spec_v = pl.BlockSpec((1, 1, BT, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0))
    spec_h = pl.BlockSpec((1, 1, K, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0))
    interpret = get_interpret()
    spec_g = (
        None
        if _g is None
        else pl.BlockSpec(
            (1, 1, BT, 128), index_map=lambda hv_idx, nt_idx: (hv_idx // num_v_tiles, nt_idx, 0, 0)
        )
    )
    spec_gamma = (
        None if g_gamma is None else pl.BlockSpec(memory_space=pl.ANY if interpret else pltpu.SMEM)
    )
    spec_scale = pl.BlockSpec(memory_space=pl.ANY if interpret else pltpu.SMEM)

    o = pl.pallas_call(
        functools.partial(_chunk_fwd_o_kernel, BT=BT),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=[spec_qk, spec_qk, spec_v, spec_h, spec_g, spec_gamma, spec_scale],
            out_specs=pl.BlockSpec(
                (1, 1, BT, BV), index_map=lambda hv_idx, nt_idx: (hv_idx, nt_idx, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((H_VT, total_NT, BT, BV), v.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
        ),
        interpret=interpret,
    )(_q, _k, _v, _h, _g, g_gamma, jnp.asarray(scale, dtype=jnp.float32).reshape(1))

    if num_v_tiles > 1:
        o = (
            o.reshape(H, num_v_tiles, total_NT, BT, BV)
            .transpose(0, 2, 3, 1, 4)
            .reshape(H, total_NT, BT, V)
        )

    o = o.reshape(H, B, NT, BT, V).transpose(1, 2, 3, 0, 4)
    return o.reshape(B, T, H, V)


def chunk_fwd_o(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    h: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunk forward output computation."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size

    if scale is None:
        scale = K**-0.5

    assert_shape(q, (B, T, H, K))
    assert_shape(k, (B, T, H, K))
    assert_shape(v, (B, T, H, V))
    assert_shape_or_none(g, (B, T, H))
    assert_shape_or_none(g_gamma, (H,))
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size={C}"
    assert (cu_seqlens_cpu is None) or (
        cu_seqlens_cpu % chunk_size == 0
    ).all(), "All sequence lengths must be divisible by chunk_size"
    if cu_seqlens_cpu is not None or cu_seqlens_dev is not None:
        assert B == 1, f"Packed varlen chunk_fwd_o expects B=1, got B={B}"
    assert scale is not None

    return _chunk_fwd_o_pl(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        chunk_size=chunk_size,
    )


# =============================================================================
# Chunk forward varlen + simple_gla_fwd entry point
# (from tops/ops/simple_gla/chunk.py and tops/ops/simple_gla/__init__.py)
# =============================================================================


def _build_align_gather_idx(cu_seqlens, aligned_cu, T_aligned):
    """For each position in the aligned layout, return (orig_pos, is_valid).
    Padding positions are mapped to 0 with is_valid=False; caller masks them."""
    N = cu_seqlens.shape[0] - 1
    real_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pos = jnp.arange(T_aligned, dtype=jnp.int32)
    seq_idx = jnp.searchsorted(aligned_cu[1:], pos, side="right")
    seq_idx = jnp.clip(seq_idx, 0, N - 1)
    offset_in_seq = pos - aligned_cu[seq_idx]
    orig_pos = cu_seqlens[seq_idx] + offset_in_seq
    is_valid = offset_in_seq < real_lens[seq_idx]
    gather_idx = jnp.where(is_valid, orig_pos, jnp.int32(0))
    return gather_idx, is_valid


def _compute_t_aligned(T_orig, N, chunk_size):
    """Static upper bound for the per-seq-aligned packed length."""
    BT = chunk_size
    T_max = T_orig + N * (BT - 1)
    return ((T_max + BT - 1) // BT) * BT


def _align_varlen_inputs(q, k, v, cu_seqlens_dev, chunk_size, T_aligned):
    """Pad each sequence to a multiple of chunk_size and rebuild cu_seqlens."""
    BT = chunk_size
    N = cu_seqlens_dev.shape[0] - 1
    real_lens = cu_seqlens_dev[1:] - cu_seqlens_dev[:-1]
    aligned_lens = ((real_lens + BT - 1) // BT) * BT
    aligned_cu = jnp.zeros(N + 1, dtype=jnp.int32)
    aligned_cu = aligned_cu.at[1:].set(jnp.cumsum(aligned_lens))
    gather_idx, is_valid = _build_align_gather_idx(cu_seqlens_dev, aligned_cu, T_aligned)

    def _gather_and_mask(x):
        gathered = x[0, gather_idx]
        gathered = jnp.where(is_valid[:, None, None], gathered, 0)
        return gathered[None]

    q_a = _gather_and_mask(q)
    k_a = _gather_and_mask(k)
    v_a = _gather_and_mask(v)
    return q_a, k_a, v_a, aligned_cu, real_lens


def _unalign_output(o_aligned, cu_seqlens_orig, aligned_cu, T_orig):
    """Scatter the aligned-layout output back to the original packed layout."""
    T_aligned = o_aligned.shape[1]
    gather_idx, is_valid = _build_align_gather_idx(cu_seqlens_orig, aligned_cu, T_aligned)
    o_out = jnp.zeros(
        (1, T_orig, o_aligned.shape[2], o_aligned.shape[3]),
        dtype=o_aligned.dtype,
    )
    o_out = o_out.at[0, gather_idx].add(jnp.where(is_valid[:, None, None], o_aligned[0], 0))
    return o_out


@functools.partial(
    jax.jit,
    static_argnames=["scale", "use_ht", "chunk_size"],
)
def chunk_simple_gla_fwd_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
    B, T_orig, H, K, V = *q.shape, v.shape[-1]
    N = cu_seqlens_dev.shape[0] - 1 if cu_seqlens_dev is not None else B

    assert_shape(q, (B, T_orig, H, K))
    assert_shape(k, (B, T_orig, H, K))
    assert_shape(v, (B, T_orig, H, V))
    assert_shape_or_none(g, (B, T_orig, H))
    assert_shape_or_none(g_gamma, (H,))
    assert_shape_or_none(h0, (N, H, K, V))
    assert cu_seqlens_cpu is None, "cu_seqlens_cpu must be None."
    assert cu_seqlens_dev is not None, "cu_seqlens_dev must not be None."
    assert (K % 128 == 0) and (V % 128 == 0)
    assert B == 1, "B must be 1."

    T_aligned = _compute_t_aligned(T_orig, N, chunk_size)
    q_a, k_a, v_a, aligned_cu, real_seq_lens = _align_varlen_inputs(
        q,
        k,
        v,
        cu_seqlens_dev,
        chunk_size,
        T_aligned,
    )

    h, ht = chunk_fwd_h_kernel_varlen(
        k=k_a,
        v=v_a,
        g=g,
        g_gamma=g_gamma,
        gk=None,
        gv=None,
        h0=h0,
        output_final_state=use_ht,
        states_in_fp32=False,
        cu_seqlens_dev=aligned_cu,
        chunk_size=chunk_size,
        seq_real_lens=real_seq_lens,
    )
    # Pallas output buffers are NOT zero-initialized on TPU. Zero-length
    # sequences are skipped by @pl.when(bos != eos), leaving their ht
    # entries undefined. Replace with h0 (or zeros) so downstream scatter
    # doesn't write garbage into the recurrent state pool.
    if use_ht and ht is not None:
        zero_len_mask = (real_seq_lens == 0)[:, None, None, None]
        if h0 is not None:
            ht = jnp.where(zero_len_mask, h0, ht)
        else:
            ht = jnp.where(zero_len_mask, 0.0, ht)
    o = chunk_fwd_o(
        q=q_a,
        k=k_a,
        v=v_a,
        g=g,
        g_gamma=g_gamma,
        h=h,
        scale=scale,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_dev=aligned_cu,
        chunk_size=chunk_size,
    )

    o = _unalign_output(o, cu_seqlens_dev, aligned_cu, T_orig)
    return o, ht


class SimpleGLAKernelMode(enum.Enum):
    """Simple GLA kernel implementation mode."""

    CHUNK = "chunk"
    FUSED_CHUNK = "fused_chunk"


def simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens_cpu: jax.Array | None = None,
    cu_seqlens_dev: jax.Array | None = None,
    chunk_size: int = 64,
    mode: SimpleGLAKernelMode = SimpleGLAKernelMode.FUSED_CHUNK,
):
    if cu_seqlens_dev is not None:
        fn = chunk_simple_gla_fwd_varlen
    else:
        raise NotImplementedError(
            f"Non-varlen simple_gla_fwd (mode={mode}) is not vendored. "
            "Only the varlen path (cu_seqlens_dev != None) is supported."
        )
    return fn(
        q,
        k,
        v,
        g=g,
        g_gamma=g_gamma,
        scale=scale,
        h0=h0,
        use_ht=use_ht,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_dev=cu_seqlens_dev,
        chunk_size=chunk_size,
    )


# --- from simple_gla_fused.py -------------------------------------
_COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS = (
    "disable_semaphore_checks" in _inspect.signature(pltpu.CompilerParams).parameters
)


def _semaphore_kwargs(disable_semaphore_checks: bool) -> dict:
    """Forward `disable_semaphore_checks` only if the running jaxlib supports it."""
    if _COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS:
        return {"disable_semaphore_checks": disable_semaphore_checks}
    return {}


def _decode_simple_gla_kernel(
    # BlockSpec inputs (full-tensor blocks; grid is (1, 1) outside)
    q_ref,  # (N, H, BK)
    k_ref,  # (N, H, BK)
    v_ref,  # (N, H, BV)
    # SMEM inputs
    g_gamma_ref,  # [H]
    recurrent_indices_ref,  # [N]
    has_initial_state_ref,  # [N]
    # ANY-memory inputs (HBM-resident, kernel manages DMA)
    recurrent_buffer_ref,  # [total_slots*H, K, V] — pre-flattened
    # Outputs
    o_ref,  # (N, H, BV)
    updated_recurrent_buffer_ref,  # [total_slots*H, K, V] — aliased
    # Scratch
    h_in_buf,  # VMEM (2, H, BK, BV) buffer.dtype
    h_out_buf,  # VMEM (2, H, BK, BV) buffer.dtype
    sem_gather,  # SemaphoreType.DMA((2,))
    sem_scatter,  # SemaphoreType.DMA((2,))
    *,
    BK: int,
    BV: int,
    scale: float,
    N: int,
    H: int,
):
    """Single Pallas program per (k_block, v_block); ALL tokens + ALL heads inside.

    Pipeline (token-level async double-buffer):
        prologue:
            start gather(token 0) → h_in_buf[0]   via sem_gather[0]
            if N >= 2: start gather(token 1) → h_in_buf[1] via sem_gather[1]

        for n in range(N):
            bank = n % 2
            wait gather(token n) on sem_gather[bank]
            if n >= 2: wait scatter(token n-2) on sem_scatter[bank]
            for each head h in 0..H-1:
                materialise h_in_buf[bank, h] as fp32 scratch (masked by
                    has_init AND pool_idx != 0)
                h_new = h_old * exp(g_gamma[h]) + outer(k, v)
                o_ref[n, h, :] = sum(q * h_new, axis=K) * scale
                stage h_to_scatter (masked to 0 when pool_idx == 0) into
                    h_out_buf[bank, h]
            start scatter(token n) → buf  via sem_scatter[bank]
            if n+2 < N: start gather(token n+2) → h_in_buf[bank]

        drain:
            wait scatter(N-1) on sem_scatter[(N-1)%2]
            if N >= 2: wait scatter(N-2) on sem_scatter[(N-2)%2]

    Notes:
      * ``recurrent_buffer_ref`` is pre-flattened to (total_slots*H, K, V)
        by the launcher (workaround for a JAX 0.8.x interpret-mode
        RefReshaper bug). Slot indexing uses ``pl.ds(pool_idx * H, H)`` to
        pull all H consecutive heads for one token in a single DMA.
      * The ``h_to_scatter`` mask protects the dummy slot 0 invariant:
        when pool_idx[n] == 0, we still scatter (keeping sem balance),
        but the scattered value is zeros, so slot 0 stays zero-valued.
    """
    i_k = pl.program_id(0)
    i_v = pl.program_id(1)

    def _buf_in_slice(n_token: int):
        pool_idx = recurrent_indices_ref[n_token]
        return recurrent_buffer_ref.at[
            pl.ds(pool_idx * H, H),
            pl.ds(i_k * BK, BK),
            pl.ds(i_v * BV, BV),
        ]

    def _buf_out_slice(n_token: int):
        pool_idx = recurrent_indices_ref[n_token]
        return updated_recurrent_buffer_ref.at[
            pl.ds(pool_idx * H, H),
            pl.ds(i_k * BK, BK),
            pl.ds(i_v * BV, BV),
        ]

    # ───── Prologue: kick off gathers for tokens 0 and 1 ─────
    pltpu.make_async_copy(_buf_in_slice(0), h_in_buf.at[0], sem_gather.at[0]).start()
    if N >= 2:
        pltpu.make_async_copy(_buf_in_slice(1), h_in_buf.at[1], sem_gather.at[1]).start()

    # Per-head decay scalars are loaded inside the per-head loop below.
    # Mosaic TPU's infer-vector-layout rejects 1D→3D reshapes
    # (``vector<H>`` → ``vector<H×1×1>``), so we can't broadcast a (H,)
    # decay vector against a (H, BK, BV) scratch. Per-head 2D tiles avoid
    # the issue; the DMA stays H-batched.

    # ───── Steady-state ping-pong loop over tokens ─────
    for n in range(N):
        bank = n % 2
        pool_idx = recurrent_indices_ref[n]
        has_init = has_initial_state_ref[n]
        # pool_idx == 0 means dummy slot ⇒ no prior state.
        # has_init == False means new sequence ⇒ no prior state.
        use_state = jnp.logical_and(has_init, pool_idx != 0)
        scatter_mask = pool_idx != 0

        # 1) Wait for this token's gather to land.
        pltpu.make_async_copy(_buf_in_slice(n), h_in_buf.at[bank], sem_gather.at[bank]).wait()

        # 2) Wait for prior scatter on this bank (token n-2) before
        #    reusing h_out_buf[bank] as the next scatter source.
        if n >= 2:
            pltpu.make_async_copy(
                h_out_buf.at[bank], _buf_out_slice(n - 2), sem_scatter.at[bank]
            ).wait()

        # 3) Per-head 2D compute.
        for h in range(H):
            decay_h = jnp.exp(g_gamma_ref[h].astype(jnp.float32))  # scalar

            gathered_h = h_in_buf.at[bank, h][...].astype(jnp.float32)  # (BK, BV)
            h_old_h = jnp.where(use_state, gathered_h, 0.0)

            q_h = q_ref[n, h, :].astype(jnp.float32)  # (BK,)
            k_h = k_ref[n, h, :].astype(jnp.float32)  # (BK,)
            v_h = v_ref[n, h, :].astype(jnp.float32)  # (BV,)

            h_new_h = h_old_h * decay_h + k_h[:, None] * v_h[None, :]  # (BK, BV)

            # Dot mirrors ``_scan_segment``'s default-precision sum.
            o_h = jnp.sum(q_h[:, None] * h_new_h, axis=0) * scale  # (BV,)
            o_ref[n, h, :] = o_h.astype(o_ref.dtype)

            # Mask to zeros when pool_idx == 0 so writing back to dummy
            # slot 0 leaves it unchanged.
            h_to_scatter_h = jnp.where(scatter_mask, h_new_h, 0.0)
            h_out_buf.at[bank, h][...] = h_to_scatter_h.astype(h_out_buf.dtype)

        # 4) Start scatter for this token (H heads at once — single DMA).
        pltpu.make_async_copy(h_out_buf.at[bank], _buf_out_slice(n), sem_scatter.at[bank]).start()

        # 5) Pre-issue gather for token (n + 2) using h_in_buf[bank]
        #    (consumed by step 2 above).
        if n + 2 < N:
            pltpu.make_async_copy(
                _buf_in_slice(n + 2), h_in_buf.at[bank], sem_gather.at[bank]
            ).start()

    # ───── Drain: wait for the two trailing scatters ─────
    last1_bank = (N - 1) % 2
    pltpu.make_async_copy(
        h_out_buf.at[last1_bank], _buf_out_slice(N - 1), sem_scatter.at[last1_bank]
    ).wait()
    if N >= 2:
        last2_bank = (N - 2) % 2
        pltpu.make_async_copy(
            h_out_buf.at[last2_bank], _buf_out_slice(N - 2), sem_scatter.at[last2_bank]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=["scale"],
    donate_argnames=["recurrent_buffer"],
)
def _launch_decode_simple_gla(
    q: jax.Array,  # [N, H, K]
    k: jax.Array,  # [N, H, K]
    v: jax.Array,  # [N, H, V]
    g_gamma: jax.Array,  # [H]
    recurrent_buffer: jax.Array,  # [total_slots, H, K, V]
    recurrent_indices: jax.Array,  # [N]
    has_initial_state: jax.Array,  # [N]
    *,
    scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Launch the DECODE fused Pallas kernel.

    Grid is (cdiv(K, BK), cdiv(V, BV)) — the token dim N is iterated
    inside the kernel. Block specs are full-tensor blocks; each program
    covers one (BK, BV) tile across ALL tokens and ALL heads.

    Notes:
      I-1: ``recurrent_buffer`` is pre-flattened to (total_slots*H, K, V)
           so the kernel can use ``pl.ds(pool_idx * H, H)`` to gather all
           H heads of one token in a single contiguous DMA.
      I-2: ``input_output_aliases={6: 1}`` — input position 6 is the flat
           recurrent buffer (after q, k, v, g_gamma, indices, has_init);
           output position 1 is the flat updated buffer.
      I-3: ``disable_semaphore_checks=True`` is required because the
           kernel uses raw ``make_async_copy`` for the async double-buffer
           pipeline. Pattern matches ``ragged_paged_attention_v3.py``.
      I-4: Only a single K tile is supported for DECODE. The output block
           does not reduce partial sums across K programs, so ``K > 128``
           must fail fast in the launcher.
    """
    interpret = get_interpret()

    N, H, K = q.shape
    V = v.shape[-1]
    BK = min(K, 128)
    BV = min(V, 128)
    assert K == BK, f"decode_simple_gla_fused only supports K <= 128; got K={K}"
    assert V % BV == 0

    # In-launcher pre-flatten — see I-1 above.
    buf_total_slots = recurrent_buffer.shape[0]
    buf_flat_shape = (buf_total_slots * H, K, V)
    recurrent_buffer_flat = jnp.reshape(recurrent_buffer, buf_flat_shape)

    # Grid: K and V blocking only. N is iterated inside the kernel so
    # the output block can be the full (N, H, BV) tensor.
    grid = (pl.cdiv(K, BK), pl.cdiv(V, BV))

    def q_index_map(k_i, _v_i):
        return 0, 0, k_i

    def k_index_map(k_i, _v_i):
        return 0, 0, k_i

    def v_index_map(_k_i, v_i):
        return 0, 0, v_i

    def o_index_map(_k_i, v_i):
        return 0, 0, v_i

    in_specs = [
        pl.BlockSpec((N, H, BK), q_index_map),
        pl.BlockSpec((N, H, BK), k_index_map),
        pl.BlockSpec((N, H, BV), v_index_map),
        pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
        pl.BlockSpec(memory_space=pltpu.SMEM),  # recurrent_indices
        pl.BlockSpec(memory_space=pltpu.SMEM),  # has_initial_state
        pl.BlockSpec(memory_space=pl.ANY),  # recurrent_buffer (HBM)
    ]
    out_specs = [
        pl.BlockSpec((N, H, BV), o_index_map),
        pl.BlockSpec(memory_space=pl.ANY),  # updated_recurrent_buffer
    ]
    out_shape = [
        jax.ShapeDtypeStruct((N, H, V), q.dtype),
        # IMPORTANT: input AND output pre-flattened to the same shape,
        # required by input_output_aliases.
        jax.ShapeDtypeStruct(buf_flat_shape, recurrent_buffer.dtype),
    ]

    scratch_shapes = [
        # Two banks for inbound gather staging (kept in buffer dtype).
        # Each bank holds (H, BK, BV) — one token's worth of all heads.
        pltpu.VMEM((2, H, BK, BV), recurrent_buffer.dtype),
        # Two banks for outbound scatter staging (kept in buffer dtype).
        pltpu.VMEM((2, H, BK, BV), recurrent_buffer.dtype),
        # 2-element DMA semaphore arrays for gather and scatter.
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]

    kernel = functools.partial(_decode_simple_gla_kernel, BK=BK, BV=BV, scale=scale, N=N, H=H)

    o, updated_buffer_flat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=128 * 1024 * 1024,
            **(_semaphore_kwargs(True)),  # see launcher I-3
        ),
        input_output_aliases={6: 1},  # see launcher I-2
    )(
        q,
        k,
        v,
        g_gamma,
        recurrent_indices,
        has_initial_state,
        recurrent_buffer_flat,
    )

    # Restore the original (total_slots, H, K, V) view for the caller.
    updated_buffer = jnp.reshape(updated_buffer_flat, recurrent_buffer.shape)
    return o, updated_buffer


def decode_simple_gla_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    recurrent_buffer: jax.Array,
    recurrent_indices: jax.Array,
    has_initial_state: jax.Array,
    *,
    g_gamma: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DECODE fused entry point. Returns (o, updated_recurrent_buffer)."""
    K = q.shape[-1]
    if scale is None:
        scale = K**-0.5
    return _launch_decode_simple_gla(
        q,
        k,
        v,
        g_gamma,
        recurrent_buffer,
        recurrent_indices,
        has_initial_state,
        scale=float(scale),
    )


__all__ = ["decode_simple_gla_fused"]


kernel = simple_gla_fwd
