"""Standalone vLLM tpu-inference gated delta net kernels (v1).

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/gdn/v1/
  transformation: four upstream modules flattened into this file in dependency
    order -- fused_gdn_kernel_common, fused_gdn_decode_kernel,
    fused_gdn_recurrent_kernel, fused_gdn_kernel_wrapper.  Repo-local imports
    were dropped and `get_default_block_sizes` was renamed per module (see
    below); no other change.

Entry point: ``ragged_gated_delta_rule`` (also exported as ``kernel``).  Its
signature is identical to the pure-JAX reference in ``baseline.py``, so the two
are directly comparable.

**Three Pallas launch points** live here:

- ``calculate_chunk_indices`` -- builds the chunk metadata the recurrent kernel
  scans over;
- ``fused_recurrent_gdn`` -- the chunked prefill path;
- ``fused_decoding_gdn`` -- the single-token decode path.

``ragged_gated_delta_rule`` dispatches between the last two using
``distribution``, so a mixed batch can touch both.

**Renamed on flattening.** Upstream defines ``get_default_block_sizes`` in the
decode kernel *and* in the recurrent kernel, with different bodies. Both are
module-local, so upstream has no conflict; concatenating them into one file
does. They are preserved here as ``get_default_block_sizes_decode`` and
``get_default_block_sizes_recurrent``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/gdn/v1",
    "files": (
        "fused_gdn_kernel_common.py",
        "fused_gdn_decode_kernel.py",
        "fused_gdn_recurrent_kernel.py",
        "fused_gdn_kernel_wrapper.py",
    ),
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_gated_delta_rule",
    "launch_points": 3,
    "renamed_on_flatten": {
        "get_default_block_sizes": (
            "get_default_block_sizes_decode",
            "get_default_block_sizes_recurrent",
        ),
    },
}

import dataclasses
import functools

import jax
from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# --- from fused_gdn_kernel_common.py ------------------------------
def validate_gdn_inputs(
    q,
    k,
    v,
    g,
    initial_state,
    state_indices,
    *,
    b=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
):
    """Validate shapes, dtypes, and TPU alignment for fused GDN kernels.

    Args:
        q: ``[T, H_qk, K]``.
        k: ``[T, H_qk, K]``.
        v: ``[T, H_v, V]``.
        g: ``[T, H_v, K]``.
        initial_state: ``[num_states, H_v, K, V]``.
        state_indices: ``[max_num_req]`` int32.
        b: ``[T, H_v, num_lanes]`` or ``None``.
        use_gate_in_kernel: Whether gate transformation is applied inside kernel.
        A_log: ``[H_v, num_lanes]`` float32 or ``None``.
        dt_bias: ``[H_v, num_lanes]`` float32 or ``None``.

    Returns:
        ``(T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing)``.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]
    V = v.shape[2]
    dtype = q.dtype
    num_states = initial_state.shape[0]
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 32 // (dtype.itemsize * 8)

    # Shape checks
    if k.shape != (T, H_qk, K):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
    if v.shape != (T, H_v, V):
        raise ValueError(f"v shape {v.shape} must be [{T}, {H_v}, {V}]")
    if g.shape != (T, H_v, K):
        raise ValueError(f"g shape {g.shape} must be [{T}, {H_v}, {K}]")
    if initial_state.shape[1:] != (H_v, K, V):
        raise ValueError(
            f"initial_state trailing dims {initial_state.shape[1:]} "
            f"must be ({H_v}, {K}, {V})")
    if b is not None and (b.ndim != 3 or b.shape[0] != T or b.shape[1] != H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}, ...]")

    # TPU alignment
    if K % num_lanes != 0 or V % num_lanes != 0:
        raise ValueError(f"K={K}, V={V} must be multiples of {num_lanes}")
    if H_qk % packing != 0:
        raise ValueError(
            f"H_qk={H_qk} must be a multiple of packing={packing}")
    if H_v % packing != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of packing={packing}")

    # Dtype checks
    if k.dtype != dtype or v.dtype != dtype:
        raise ValueError(f"q/k/v must share the same dtype, got q={dtype}, "
                         f"k={k.dtype}, v={v.dtype}")

    if state_indices.dtype != jnp.int32:
        raise ValueError(
            f"state_indices must be int32, got {state_indices.dtype}")

    # Gate-in-kernel checks
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        if dt_bias is not None and (dt_bias.ndim != 2
                                    or dt_bias.shape[0] != H_v):
            raise ValueError(
                f"dt_bias shape {dt_bias.shape} must be [{H_v}, ...]")
        if dt_bias is not None and dt_bias.dtype != jnp.float32:
            raise ValueError(f"dt_bias must be float32, got {dt_bias.dtype}")

    return T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing


# --- from fused_gdn_decode_kernel.py ------------------------------
def get_default_block_sizes_decode(
    T: int,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    state_dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
) -> int:
    """Choose bt to balance pipelining and VMEM utilization to minimize latency

    Accounts for state scratch ``(bt, H_v, K, V)`` of ``state_dtype``, optional
    a_log / dt_bias, and bt-proportional tiles that ``emit_pipeline``
    double-buffers (q, k, v, g, b, o).
    """
    ibits = dtype.itemsize * 8

    # Fixed (not bt-dependent), in bits
    num_lanes = pltpu.get_tpu_info().num_lanes
    fixed_bits = 0
    if use_gate_in_kernel:
        fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
    if has_dt_bias:
        fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

    # bt-proportional (in bits):
    #   state scratch: (2*bt, H_v, K, V) state_dtype (double buffer)
    #   pipeline tiles (×2 for emit_pipeline double buffering):
    #     q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
    #     g(bt,H_v,K) float32                     -> H_v·K·32
    #     v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
    #     b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
    sbits = state_dtype.itemsize * 8
    per_bt_bits = (
        # State scratch: 2 (double buffer) * bt * H_v * K * V * sbits bits
        2 * H_v * K * V * sbits +
        # Pipeline tiles (multiplied by 2 for double buffering in emit_pipeline)
        2 * (
            2 * H_qk * K * ibits  # q and k
            + H_v * K * 32  # g (float32)
            + 2 * H_v * V * ibits  # v and o
            + H_v * num_lanes * ibits  # b
        ))

    bt_max = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)

    # bt_max is not the optimal bt size because it limits the pipelining capability
    # The first step needs to load synchronously from HBM to start and last step needs
    # to write to HBM.
    bt_adjusted = min(pl.cdiv(T, 8), bt_max)

    # Round down to the nearest power of 2
    return 1 << (bt_adjusted.bit_length() - 1)


# ── Outer kernel ──────────────────────────────────────────────────────


def _decode_kernel_main(
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K] float32
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    distribution_ref,  # [2] int32 (SMEM)
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, bt, H_v, K, V] VMEM scratch
    h_load_sems,
    h_store_sems,
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    bt: int,
    apply_silu: bool,
):
    decode_end = distribution_ref[0]
    nb_t = pl.cdiv(decode_end, bt)
    repeat_factor = H_v // H_qk

    bounded_bt = pl.BoundedSlice(bt)

    def token_map(i):
        t_start = i * bt
        t_size = jnp.minimum(bt, decode_end - t_start)
        return (pl.ds(t_start, t_size), 0, 0)

    qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), token_map)
    g_spec = pl.BlockSpec((bounded_bt, H_v, K), token_map)
    v_spec = pl.BlockSpec((bounded_bt, H_v, V), token_map)
    if b_hbm is not None:
        b_last = b_hbm.shape[2]
        b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), token_map)
    else:
        b_spec = None

    if use_gate_in_kernel and a_log_hbm is not None:
        a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
    else:
        a_log_spec = None
    dt_bias_spec = (pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _:
                                 (0, 0)) if dt_bias_hbm is not None else None)

    # ── Prologue: start loading first bt-block's states ──
    for i_t in range(bt):

        @pl.when(i_t < decode_end)
        def _first_load():
            si = state_indices_ref[i_t]
            pltpu.make_async_copy(
                state_hbm.at[pl.ds(si, 1), :, :, :],
                h_bufs.at[0, pl.ds(i_t, 1), :, :, :],
                h_load_sems.at[0],
            ).start()

    # ── Inner kernel (runs per bt-block) ──
    def _inner_kernel(
        q_ref,  # [<=bt, H_qk, K]
        k_ref,  # [<=bt, H_qk, K]
        v_ref,  # [<=bt, H_v, V]
        g_ref,  # [<=bt, H_v, K]
        b_ref,  # [<=bt, H_v, num_lanes]
        a_log_ref,  # [H_v, num_lanes] or None
        dt_bias_ref,  # [H_v, num_lanes] or None
        o_ref,  # [<=bt, H_v, V]
        h_bufs_s,
        state_indices_s,  # [max_num_req] int32 (SMEM)
        h_load_sems_s,
        h_store_sems_s,
    ):
        block_id = pl.program_id(0)
        t_start = block_id * bt
        block_len = jnp.minimum(bt, decode_end - t_start)
        buf_idx = block_id % 2
        next_buf_idx = (block_id + 1) % 2

        if use_gate_in_kernel:
            a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
            if dt_bias_ref is not None:
                dt_bias_tile = dt_bias_ref[...].astype(
                    jnp.float32)  # [H_v, num_lanes]
                if K > dt_bias_tile.shape[-1]:
                    dt_bias_val = jnp.concatenate(
                        [dt_bias_tile] * (K // dt_bias_tile.shape[-1]),
                        axis=-1)
                else:
                    dt_bias_val = dt_bias_tile

        # ── Step 1: Prefetch next bt-block's states ──
        next_t_start = t_start + bt
        next_block_len = jnp.maximum(
            jnp.minimum(bt, decode_end - next_t_start), 0)
        for i_t in range(bt):

            @pl.when(i_t < next_block_len)
            def _prefetch():
                next_si = state_indices_s[next_t_start + i_t]
                pltpu.make_async_copy(
                    state_hbm.at[pl.ds(next_si, 1), :, :, :],
                    h_bufs_s.at[next_buf_idx,
                                pl.ds(i_t, 1), :, :, :],
                    h_load_sems_s.at[next_buf_idx],
                ).start()

        # ── Step 2: Wait for current bt-block's state loads ──
        pltpu.make_async_copy(
            state_hbm.at[pl.ds(0, block_len), :, :, :],
            h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
            h_load_sems_s.at[buf_idx],
        ).wait()

        # ── Step 3: Compute ──
        q_all = q_ref[...].astype(jnp.float32)
        k_all = k_ref[...].astype(jnp.float32)
        v_all = v_ref[...].astype(jnp.float32)
        g_all = g_ref[...].astype(jnp.float32)

        if apply_silu:
            q_all = jax.nn.silu(q_all)
            k_all = jax.nn.silu(k_all)
            v_all = jax.nn.silu(v_all)

        if b_ref is not None:
            b_all = b_ref[...].astype(jnp.float32)
            if V > b_all.shape[-1]:
                beta_all = jax.nn.sigmoid(
                    jnp.concatenate([b_all] * (V // b_all.shape[-1]), axis=-1))
            else:
                beta_all = jax.nn.sigmoid(b_all)

        if use_qk_l2norm:
            q_all = q_all / jnp.sqrt(
                jnp.sum(q_all * q_all, axis=-1, keepdims=True) + 1e-6)
            k_all = k_all / jnp.sqrt(
                jnp.sum(k_all * k_all, axis=-1, keepdims=True) + 1e-6)
        q_all = q_all * scale

        qk_dot_all = jnp.sum(q_all * k_all, axis=-1, keepdims=True)

        if repeat_factor > 1:
            q_all = jnp.repeat(q_all, repeat_factor, axis=1)
            k_all = jnp.repeat(k_all, repeat_factor, axis=1)
            qk_dot_all = jnp.repeat(qk_dot_all, repeat_factor, axis=1)

        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _process_token():
                h0 = h_bufs_s[buf_idx, i_t].astype(jnp.float32)
                q_t = q_all[i_t]
                k_t = k_all[i_t]
                v_t = v_all[i_t]
                g_t = g_all[i_t]
                qk_dot = qk_dot_all[i_t]

                if use_gate_in_kernel:
                    g_val = g_t
                    if dt_bias_ref is not None:
                        g_val = g_val + dt_bias_val
                    if lower_bound is not None:
                        gk = lower_bound / (1.0 +
                                            jnp.exp(-(a_val[:, None] * g_val)))
                    else:
                        gk = -a_val[:, None] * jax.nn.softplus(g_val)
                else:
                    gk = g_t

                exp_gk = jnp.exp(gk)
                k_t_scaled = k_t * exp_gk

                kh = jax.lax.dot_general(
                    k_t_scaled.reshape(H_v, 1, K),
                    h0,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(H_v, V)

                v_diff = v_t - kh
                if b_ref is not None:
                    b_v = beta_all[i_t] * v_diff
                else:
                    b_v = v_diff

                q_t_scaled = q_t * exp_gk
                o_step1 = jax.lax.dot_general(
                    q_t_scaled.reshape(H_v, 1, K),
                    h0,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(H_v, V)

                o_t = o_step1 + qk_dot * b_v
                h_new = h0 * exp_gk[:, :, None] + k_t[:, :,
                                                      None] * b_v[:, None, :]

                o_ref[i_t] = o_t.astype(o_ref.dtype)
                h_bufs_s[buf_idx, i_t] = h_new.astype(h_bufs_s.dtype)

        # ── Step 4: Wait for stores from 2 blocks ago (same buffer set) ──
        prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
        prev_block_len = jnp.where(
            block_id >= 2,
            jnp.minimum(bt, decode_end - prev_t_start),
            0,
        )

        @pl.when(prev_block_len > 0)
        def _wait_prev_store():
            pltpu.make_async_copy(
                h_bufs_s.at[buf_idx,
                            pl.ds(0, prev_block_len), :, :, :],
                state_hbm.at[pl.ds(0, prev_block_len), :, :, :],
                h_store_sems_s.at[buf_idx],
            ).wait()

        # ── Step 5: Start storing current bt-block's states ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _start_store():
                si = state_indices_s[t_start + i_t]
                pltpu.make_async_copy(
                    h_bufs_s.at[buf_idx, pl.ds(i_t, 1), :, :, :],
                    state_hbm.at[pl.ds(si, 1), :, :, :],
                    h_store_sems_s.at[buf_idx],
                ).start()

    pltpu.emit_pipeline(
        _inner_kernel,
        grid=(nb_t, ),
        in_specs=[
            qk_spec,
            qk_spec,
            v_spec,
            g_spec,
            b_spec,
            a_log_spec,
            dt_bias_spec,
        ],
        out_specs=v_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        b_hbm,
        a_log_hbm,
        dt_bias_hbm,
        o_hbm,
        scratches=[h_bufs, state_indices_ref, h_load_sems, h_store_sems],
    )

    # ── Epilogue: drain outstanding stores ──
    last_buf_idx = (nb_t - 1) % 2
    other_buf_idx = nb_t % 2
    last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
    pltpu.make_async_copy(
        h_bufs.at[last_buf_idx,
                  pl.ds(0, last_block_len), :, :, :],
        state_hbm.at[pl.ds(0, last_block_len), :, :, :],
        h_store_sems.at[last_buf_idx],
    ).wait()

    other_block_len = jnp.where(
        nb_t >= 2,
        jnp.minimum(bt, decode_end - (nb_t - 2) * bt),
        0,
    )

    @pl.when(other_block_len > 0)
    def _drain_other():
        pltpu.make_async_copy(
            h_bufs.at[other_buf_idx,
                      pl.ds(0, other_block_len), :, :, :],
            state_hbm.at[pl.ds(0, other_block_len), :, :, :],
            h_store_sems.at[other_buf_idx],
        ).wait()


# ── Public API ───────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
        "apply_silu",
    ],
)
def fused_decoding_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    g: jax.Array,  # [T, H_v, K] float32
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None,  # [T, H_v, num_lanes] or None
    *,
    scale: float,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    lower_bound: float | None = None,
    apply_silu: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused recurrent GDN single-step decode.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        g: Per-key gating ``[T, H_v, K]``, float32.
        initial_state: State cache ``[num_states, H_v, K, V]``
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        b: Raw betas ``[T, H_v, num_lanes]`` (sigmoid applied inside kernel).
        scale: Scale factor.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v, num_lanes]`` float32.
        dt_bias: Per-head bias ``[H_v, num_lanes]`` float32.
        lower_bound: If set, use sigmoid gate instead of softplus.
        apply_silu: Apply SiLU activation to q, k, v inside the kernel.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]``.
    """
    T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = validate_gdn_inputs(
        q,
        k,
        v,
        g,
        initial_state,
        state_indices,
        b=b,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = get_default_block_sizes_decode(
        T,
        H_qk,
        H_v,
        K,
        V,
        dtype,
        initial_state.dtype,
        use_gate_in_kernel,
        dt_bias is not None,
        vmem_bytes_limit,
    )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    decode_end = distribution[0]
    grid_dim = jnp.where(decode_end > 0, 1, 0)

    n_b = b is not None
    n_gate = (A_log is not None) + (dt_bias is not None)

    scope_name = f"decoding_gdn-bt_{bt}"

    o, state = pl.pallas_call(
        functools.partial(
            _decode_kernel_main,
            H_qk=H_qk,
            H_v=H_v,
            K=K,
            V=V,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            bt=bt,
            apply_silu=apply_silu,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                *([any_spec] * 4),  # q, k, v, g
                any_spec if b is not None else None,  # b
                smem_spec,  # state_indices
                any_spec if A_log is not None else None,
                any_spec if dt_bias is not None else None,
                smem_spec,  # distribution
                any_spec,  # state_init
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                pltpu.VMEM((2, bt, H_v, K, V),
                           initial_state.dtype),  # h_bufs (double buffer)
                pltpu.SemaphoreType.DMA((2, )),  # h_load_sems
                pltpu.SemaphoreType.DMA((2, )),  # h_store_sems
            ],
        ),
        input_output_aliases={
            2: 0,  # v aliases o
            6 + n_b + n_gate: 1,  # initial_state aliases updated_state
        },
        out_shape=[
            jax.ShapeDtypeStruct((T, H_v, V), dtype),
            jax.ShapeDtypeStruct((num_states, H_v, K, V), initial_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=scope_name,
    )(
        q,
        k,
        v,
        g,
        b,
        state_indices,
        A_log,
        dt_bias,
        distribution,
        initial_state,
    )

    return o, state


def ragged_gated_delta_rule_decode_only(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
    apply_silu=False,
):
    """Adapter for decode-only branch matching ragged_gated_delta_rule interface.

    Internally reshapes inputs and delegates to :func:`fused_decoding_gdn`.

    Args:
        mixed_qkv: ``(num_tokens, 2*n_kq*d_k + n_v*d_v)`` post-conv/silu.
        b: ``(num_tokens, n_v)`` — raw beta (sigmoid applied in kernel).
        a: ``(num_tokens, n_v)`` — raw alpha (gate transform in kernel).
        recurrent_state: ``(num_states, n_v, d_k, d_v)``.
        A_log: ``(n_v,)`` float32.
        dt_bias: ``(n_v,)`` float32.
        query_start_loc: ``(num_seqs+1,)`` int32.
        state_indices: ``(num_seqs,)`` int32.
        distribution: ``(3,)`` int32 — ``(decode_end, prefill_end, mixed_end)``.
        has_initial_state: Ignored on decode path.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.
        apply_silu: Whether to apply silu in-kernel

    Returns:
        ``(updated_recurrent_state, output)`` where
        *updated_recurrent_state* is ``(num_states, n_v, d_k, d_v)`` and
        *output* is ``(num_tokens, n_v*d_v)``.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k

    q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
    k = mixed_qkv[..., key_dim:key_dim * 2].reshape(num_tokens, n_kq, d_k)
    v = mixed_qkv[..., key_dim * 2:].reshape(num_tokens, n_v, d_v)

    g = a
    if g.shape == (num_tokens, n_v):
        g = jnp.broadcast_to(g[..., None], (num_tokens, n_v, d_k))

    num_lanes = pltpu.get_tpu_info().num_lanes
    if b is not None:
        b = jnp.broadcast_to(b[:, :, None], (num_tokens, n_v, num_lanes))

    if A_log is not None:
        A_log = jnp.broadcast_to(A_log[:, None],
                                 (n_v, num_lanes)).astype(jnp.float32)

    if dt_bias is not None:
        dt_bias = jnp.broadcast_to(dt_bias[:, None],
                                   (n_v, num_lanes)).astype(jnp.float32)

    # (decode_end, prefill_end, mixed_end) → (decode_end, total)
    fused_distribution = jnp.stack([distribution[0], distribution[2]])

    scale = d_k**-0.5

    output, new_recurrent_state = fused_decoding_gdn(
        q,
        k,
        v,
        g,
        recurrent_state,
        state_indices,
        distribution=fused_distribution,
        b=b,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
        apply_silu=apply_silu,
    )

    output = output.reshape(num_tokens, n_v * d_v)
    return new_recurrent_state, output


# --- from fused_gdn_recurrent_kernel.py ---------------------------
def get_default_block_sizes_recurrent(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
    state_dtype=jnp.float32,
) -> int:
    """Choose bt to maximize VMEM utilization.

    The recurrent kernel uses a fixed ``(2, H_v, K, V)`` state double
    buffer of ``state_dtype`` regardless of bt.  Only pipeline tiles scale with bt.
    """
    ibits = dtypes.itemsize_bits(dtype)
    sbits = dtypes.itemsize_bits(state_dtype)

    # Fixed (in bits): h_bufs (2, H_v, K, V) of state_dtype — always 2 buffers
    num_lanes = pltpu.get_tpu_info().num_lanes
    fixed_bits = 2 * H_v * K * V * sbits
    if use_gate_in_kernel:
        fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
    if has_dt_bias:
        fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

    # bt-proportional (in bits): pipeline tiles (×2 for emit_pipeline double buffering)
    #   q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
    #   g(bt,H_v,K) float32                     -> H_v·K·32
    #   v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
    #   b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
    per_bt_bits = 2 * (2 * H_qk * K * ibits + H_v * K * 32 +
                       2 * H_v * V * ibits + H_v * num_lanes * ibits)

    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Round down to nearest power of 2
    return 1 << (bt.bit_length() - 1)


# ── Metadata ──


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNChunkIndices:
    num_blocks: jax.Array  # [1] int32
    block_id_to_seq_idx: jax.Array  # [max_num_blocks + 1] int32 (sentinel at end)
    block_id_to_t_offset: jax.Array  # [max_num_blocks + 1] int32


@jax.named_scope("calculate_chunk_indices")
def calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks, bt: int):
    """Pre-compute per-block metadata as a standalone Pallas kernel.

    Iterates over sequences and splits each into BT-sized work
    items.  A sequence boundary that falls mid-block creates two work
    items for that block (one per sequence).

    Returns a GDNChunkIndices with num_blocks, block_id_to_seq_idx, block_id_to_t_offset.
    """

    def _kernel(
        cu_seqlens_ref,
        distribution_ref,
        meta_out,
        *,
        bt: int,
    ):
        seq_start = distribution_ref[0]
        seq_end = distribution_ref[1]
        n_seqs = seq_end - seq_start

        @jax.named_scope("inner_block_loop")
        def inner_block_loop(blk_rel, carry, *, seq_idx, eos):
            num_blocks, t_cursor = carry
            block_id = num_blocks + blk_rel

            t_start = t_cursor + blk_rel * bt
            t_end = jnp.minimum(t_start + bt, eos)

            meta_out.block_id_to_seq_idx[block_id] = seq_idx
            meta_out.block_id_to_t_offset[block_id] = t_start
            meta_out.block_id_to_t_offset[block_id + 1] = t_end

            return num_blocks, t_cursor

        @jax.named_scope("outer_seq_loop")
        def outer_seq_loop(seq_rel, carry):
            num_blocks, t_cursor = carry
            seq_idx = seq_start + seq_rel
            eos = cu_seqlens_ref[seq_idx + 1]

            seq_len_from_cursor = eos - t_cursor
            n_seq_blocks = pl.cdiv(seq_len_from_cursor, bt)

            loop_fn = functools.partial(
                inner_block_loop,
                seq_idx=seq_idx,
                eos=eos,
            )
            jax.lax.fori_loop(0, n_seq_blocks, loop_fn, (num_blocks, t_cursor))

            return num_blocks + n_seq_blocks, eos

        first_token = cu_seqlens_ref[seq_start]
        num_blocks, _ = jax.lax.fori_loop(
            0,
            n_seqs,
            outer_seq_loop,
            (jnp.int32(0), first_token),
        )
        # Sentinel for look-ahead: block_id+1 reads -1 past the last block
        meta_out.block_id_to_seq_idx[num_blocks] = jnp.int32(-1)
        meta_out.num_blocks[0] = num_blocks

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    meta = pl.pallas_call(
        functools.partial(_kernel, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            out_specs=GDNChunkIndices(
                num_blocks=smem_spec,
                block_id_to_seq_idx=smem_spec,
                block_id_to_t_offset=smem_spec,
            ),
            grid=(1, ),
        ),
        out_shape=GDNChunkIndices(
            num_blocks=jax.ShapeDtypeStruct((1, ), jnp.int32),
            block_id_to_seq_idx=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                     jnp.int32),
            block_id_to_t_offset=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                      jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(cu_seqlens, distribution)

    return meta


# ── Index maps ──


class _MetadataIndexMaps:
    """Index maps driven by pre-computed metadata arrays."""

    def __init__(self, meta: GDNChunkIndices):
        self.meta = meta

    def token_map(self, block_id):
        t_start = self.meta.block_id_to_t_offset[block_id]
        t_end = self.meta.block_id_to_t_offset[block_id + 1]
        t_size = t_end - t_start
        return (pl.ds(t_start, t_size), 0, 0)


# ── Outer kernel ──


def _recurrent_gdn_main(
    meta,  # GDNChunkIndices (SMEM)
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K]
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    has_initial_state_ref,  # [max_num_req] int32 (SMEM); 0 = zero out h0
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, H_v, K, V] VMEM scratch (double buffer)
    h_load_sems,  # [2] DMA semaphores
    h_store_sems,  # [2] DMA semaphores
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    bt: int,
):
    num_blocks = meta.num_blocks[0]
    repeat_factor = H_v // H_qk
    # Build index maps from metadata
    idx_maps = _MetadataIndexMaps(meta)
    bounded_bt = pl.BoundedSlice(bt)

    qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), idx_maps.token_map)
    g_spec = pl.BlockSpec((bounded_bt, H_v, K), idx_maps.token_map)
    v_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
    o_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
    if b_hbm is not None:
        b_last = b_hbm.shape[2]
        b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), idx_maps.token_map)
    else:
        b_spec = None

    # ── Prologue: start h0 load for first sequence (don't wait) ──
    first_seq = meta.block_id_to_seq_idx[0]
    first_state_idx = state_indices_ref[first_seq]
    first_buf = first_seq % 2
    pltpu.make_async_copy(
        state_hbm.at[pl.ds(first_state_idx, 1), :, :, :],
        h_bufs.at[pl.ds(first_buf, 1), :, :, :],
        h_load_sems.at[first_buf],
    ).start()

    # ── Inner kernel ──
    def _inner_kernel_body(
            q_ref,  # [<=bt, H_qk, K]
            k_ref,  # [<=bt, H_qk, K]
            v_ref,  # [<=bt, H_v, V]
            g_ref,  # [<=bt, H_v, K]
            b_ref,  # [<=bt, H_v, num_lanes]
            a_log_ref,  # [H_v, num_lanes] or None
            dt_bias_ref,  # [H_v, num_lanes] or None
            o_ref,  # [<=bt, H_v, V]
            h_bufs_s,  # [2, H_v, K, V] VMEM scratch
            meta_s,  # GDNChunkIndices (SMEM)
            state_indices_s,  # [max_num_req] int32 (SMEM)
            has_initial_state_s,  # [max_num_req] int32 (SMEM)
            h_load_sems_s,  # [2] DMA semaphores
            h_store_sems_s,  # [2] DMA semaphores
    ):
        block_id = pl.program_id(0)
        seq_idx = meta_s.block_id_to_seq_idx[block_id]
        t_start = meta_s.block_id_to_t_offset[block_id]
        t_end = meta_s.block_id_to_t_offset[block_id + 1]
        block_len = t_end - t_start

        # Detect sequence start
        prev_seq_idx = meta_s.block_id_to_seq_idx[jnp.maximum(block_id - 1, 0)]
        is_new_seq = (block_id == 0) | (seq_idx != prev_seq_idx)

        # Look ahead: detect sequence end
        next_seq_idx = meta_s.block_id_to_seq_idx[block_id + 1]
        is_seq_end = seq_idx != next_seq_idx

        # Double-buffer index: alternate buffers per sequence
        buf_idx = seq_idx % 2
        safe_next_seq = jnp.maximum(next_seq_idx, 0)
        next_buf_idx = safe_next_seq % 2

        # Pool indices via state_indices
        state_idx = state_indices_s[seq_idx]
        safe_next_state_idx = state_indices_s[safe_next_seq]

        # ── Step 1: Prefetch next h0 & wait for current h0 load ──
        prefetch_cp = pltpu.make_async_copy(
            state_hbm.at[pl.ds(safe_next_state_idx, 1), :, :, :],
            h_bufs_s.at[pl.ds(next_buf_idx, 1), :, :, :],
            h_load_sems_s.at[next_buf_idx],
        )
        load_wait_cp = pltpu.make_async_copy(
            state_hbm.at[pl.ds(state_idx, 1), :, :, :],
            h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
            h_load_sems_s.at[buf_idx],
        )

        @pl.when(is_seq_end & (next_seq_idx >= 0))
        def _prefetch():
            prefetch_cp.start()

        @pl.when(is_new_seq)
        def _wait_h0():
            load_wait_cp.wait()

        # If the request has no prior recurrent state (brand-new prefill
        # landing on a freshly-allocated mamba slot), the DMA-loaded h0
        # is stale data from a previous tenant. Overwrite with zeros so
        # the recurrent update starts from zero, mirroring the chunked
        # path's `init_states_for_seqs = jnp.where(has_initial_state,
        # ..., 0)` and GPU's `initial_state[~has_initial_state, ...] = 0`
        # in `gdn_linear_attn._forward_core`.
        has_init = has_initial_state_s[seq_idx]

        @pl.when(is_new_seq & (has_init == 0))
        def _zero_h0():
            h_bufs_s[buf_idx] = jnp.zeros((H_v, K, V), dtype=h_bufs_s.dtype)

        # ── Step 2: Compute ──
        h = h_bufs_s[buf_idx].astype(jnp.float32)

        if use_gate_in_kernel:
            a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
            if dt_bias_ref is not None:
                dt_bias_tile = dt_bias_ref[...].astype(
                    jnp.float32)  # [H_v, num_lanes]
                if K > dt_bias_tile.shape[-1]:
                    dt_bias_val = jnp.concatenate(
                        [dt_bias_tile] * (K // dt_bias_tile.shape[-1]),
                        axis=-1)
                else:
                    dt_bias_val = dt_bias_tile

        def step(local_t, h):
            q_t = q_ref[local_t].astype(jnp.float32)
            k_t = k_ref[local_t].astype(jnp.float32)
            v_t = v_ref[local_t].astype(jnp.float32)
            g_t = g_ref[local_t].astype(jnp.float32)
            if b_ref is not None:
                b_tile = b_ref[local_t].astype(jnp.float32)  # [H_v, num_lanes]
                if V > b_tile.shape[-1]:
                    beta_t = jax.nn.sigmoid(
                        jnp.concatenate([b_tile] * (V // b_tile.shape[-1]),
                                        axis=-1))  # [H_v, V]
                else:
                    beta_t = jax.nn.sigmoid(
                        b_tile)  # [H_v, num_lanes] (== [H_v, V])

            if use_qk_l2norm:
                q_t = q_t / jnp.sqrt(
                    jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
                k_t = k_t / jnp.sqrt(
                    jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
            q_t = q_t * scale

            # GQA: repeat q/k from H_qk to H_v heads
            if repeat_factor > 1:
                q_t = jnp.repeat(q_t, repeat_factor, axis=0)
                k_t = jnp.repeat(k_t, repeat_factor, axis=0)

            if use_gate_in_kernel:
                if dt_bias_ref is not None:
                    g_t = g_t + dt_bias_val
                if lower_bound is not None:
                    gk = lower_bound / (1.0 + jnp.exp(-(a_val[:, None] * g_t)))
                else:
                    gk = -a_val[:, None] * jax.nn.softplus(g_t)
            else:
                gk = g_t

            # Same algebraic identity as fused_gdn_decode_kernel.py:
            # o = q @ h_pre + (q . k) * b_v
            # Lets MXU(o) and VPU(rank-1 update) run in parallel.
            h_pre = h * jnp.exp(gk[:, :, None])
            kh = jax.lax.dot_general(
                k_t.reshape(H_v, 1, K),
                h_pre,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).reshape(H_v, V)
            v_diff = v_t - kh
            b_v = beta_t * v_diff if b_ref is not None else v_diff

            o_step1 = jax.lax.dot_general(
                q_t.reshape(H_v, 1, K),
                h_pre,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).reshape(H_v, V)
            qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            o_t = o_step1 + qk_dot * b_v
            h = h_pre + k_t[:, :, None] * b_v[:, None, :]

            o_ref[local_t] = o_t.astype(o_ref.dtype)
            return h

        h = jax.lax.fori_loop(0, block_len, step, h, unroll=False)
        h_bufs_s[buf_idx] = h.astype(h_bufs_s.dtype)

        # ── Step 3: Wait prev store, start current store ──
        # Store updated state back at state_idx.
        store_cp = pltpu.make_async_copy(
            h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
            state_hbm.at[pl.ds(state_idx, 1), :, :, :],
            h_store_sems_s.at[buf_idx],
        )
        # Wait for prev store from same buffer (S-2's store).
        # Skip for the first 2 sequences — no prior store on this sem.
        has_prev_same_buf = (seq_idx - first_seq) >= 2

        @pl.when(is_seq_end & has_prev_same_buf)
        def _wait_prev_store():
            store_cp.wait()

        # Start current store
        @pl.when(is_seq_end)
        def _start_store():
            store_cp.start()

    # Run pipeline — None specs/inputs are passed through as None refs
    if use_gate_in_kernel and a_log_hbm is not None:
        a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
    else:
        a_log_spec = None
    dt_bias_spec = (pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _:
                                 (0, 0)) if dt_bias_hbm is not None else None)

    pltpu.emit_pipeline(
        _inner_kernel_body,
        grid=(num_blocks, ),
        in_specs=[
            qk_spec, qk_spec, v_spec, g_spec, b_spec, a_log_spec, dt_bias_spec
        ],
        out_specs=o_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        b_hbm,
        a_log_hbm,
        dt_bias_hbm,
        o_hbm,
        scratches=[
            h_bufs, meta, state_indices_ref, has_initial_state_ref,
            h_load_sems, h_store_sems
        ],
    )

    # ── Epilogue: wait for outstanding stores ──
    last_seq = meta.block_id_to_seq_idx[jnp.maximum(num_blocks - 1, 0)]
    last_buf = last_seq % 2
    other_buf = 1 - last_buf
    # Always wait for last seq's store
    pltpu.make_async_copy(
        h_bufs.at[pl.ds(0, 1), :, :, :],
        state_hbm.at[pl.ds(0, 1), :, :, :],
        h_store_sems.at[last_buf],
    ).wait()
    # If >= 2 seqs, also wait for the other sem
    drain_other = pltpu.make_async_copy(
        h_bufs.at[pl.ds(0, 1), :, :, :],
        state_hbm.at[pl.ds(0, 1), :, :, :],
        h_store_sems.at[other_buf],
    )

    @pl.when(last_seq != first_seq)
    def _drain_other():
        drain_other.wait()


# ── Public API ──


def fused_recurrent_gdn(
        q,  # [T, H_qk, K]
        k,  # [T, H_qk, K]
        v,  # [T, H_v, V]
        cu_seqlens,  # [N+1] int32
        g,  # [T, H_v, K] float32
        initial_state,  # [num_states, H_v, K, V] float32
        state_indices,  # [max_num_req] int32
        b,  # [T, H_v, num_lanes] or None
        has_initial_state,  # [max_num_req] int32 (0/1)
        *,
        scale,  # float
        use_qk_l2norm,  # bool
        use_gate_in_kernel=False,  # bool
        A_log=None,  # [H_v, num_lanes] or None
        dt_bias=None,  # [H_v, num_lanes] or None
        lower_bound=None,  # float or None
        distribution,  # [2] int32
):
    """Run the pre-computed-metadata recurrent GDN pallas kernel.

    ``has_initial_state[i]`` indicates whether request ``i``'s recurrent
    slot already holds a valid prior state (continuation, prefix-cache
    hit) or whether the slot is freshly allocated and its contents must
    be treated as zeros for this call. Mirrors the chunked path's
    `init_states_for_seqs = jnp.where(has_initial_state, ..., 0)` and
    GPU's `initial_state[~has_initial_state, ...] = 0` in
    ``gdn_linear_attn._forward_core``. Pass an all-ones array if the
    caller wants the previous (no-op) behaviour.
    """
    T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = (validate_gdn_inputs(
        q,
        k,
        v,
        g,
        initial_state,
        state_indices,
        b=b,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias))
    max_num_req = cu_seqlens.shape[0] - 1

    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = get_default_block_sizes_recurrent(H_qk,
                                 H_v,
                                 K,
                                 V,
                                 dtype,
                                 use_gate_in_kernel,
                                 dt_bias is not None,
                                 vmem_bytes_limit,
                                 state_dtype=initial_state.dtype)

    # Worst case: cdiv(T, bt) base blocks + up to max_num_req-1 boundary splits
    max_num_blocks = (T + bt - 1) // bt + max_num_req - 1

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    o_shape = jax.ShapeDtypeStruct((T, H_v, V), dtype)
    state_shape = jax.ShapeDtypeStruct((num_states, H_v, K, V),
                                       initial_state.dtype)

    meta = calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks,
                                   bt)

    n_seqs = distribution[1] - distribution[0]
    grid_dim = jnp.where(n_seqs > 0, 1, 0)

    n_b = (b is not None)
    n_gate = (A_log is not None) + (dt_bias is not None)

    scope_name = f"recurrent_gdn-bt_{bt}"

    o, state = pl.pallas_call(
        functools.partial(
            _recurrent_gdn_main,
            H_qk=H_qk,
            H_v=H_v,
            K=K,
            V=V,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            bt=bt,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                GDNChunkIndices(
                    num_blocks=smem_spec,
                    block_id_to_seq_idx=smem_spec,
                    block_id_to_t_offset=smem_spec,
                ),
                *([any_spec] * 4),  # q, k, v, g
                any_spec if b is not None else None,  # b
                smem_spec,  # state_indices
                any_spec if A_log is not None else None,
                any_spec if dt_bias is not None else None,
                any_spec,  # state_init (= initial_state)
                smem_spec,  # has_initial_state
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                # h_bufs match HBM dtype so the DMAs don't need conversion.
                # The compute path upcasts h_bufs to fp32 once per block
                # (h = h_bufs_s[buf_idx].astype(fp32) above) and carries it
                # as fp32 across the per-token fori_loop, so on-chip math is
                # fp32 regardless of HBM storage dtype.
                pltpu.VMEM((2, H_v, K, V),
                           initial_state.dtype),  # h_bufs (double buffer)
                pltpu.SemaphoreType.DMA((2, )),  # h_load_sems
                pltpu.SemaphoreType.DMA((2, )),  # h_store_sems
            ],
        ),
        # Aliases reference flat positional inputs. `meta` is a 3-leaf
        # pytree, so the absolute index of `v` is 5 (3 meta leaves + q, k)
        # and the absolute index of `initial_state` is 8 + n_b + n_gate.
        # `has_initial_state` is appended to the end and is not aliased.
        input_output_aliases={
            5: 0,
            8 + n_b + n_gate: 1
        },
        out_shape=[o_shape, state_shape],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=scope_name,
    )(
        meta,
        q,
        k,
        v,
        g,
        b,
        state_indices,
        A_log,
        dt_bias,
        initial_state,
        has_initial_state,
    )

    return o, state


# --- from fused_gdn_kernel_wrapper.py -----------------------------
def _dispatch_with_distribution(
    q,
    k,
    v,
    cu_seqlens,
    g,
    initial_state,
    state_indices,
    b,
    has_initial_state,
    *,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    lower_bound,
    distribution,
):
    """Dispatch to decode and recurrent kernels following the RPA pattern.

    Both kernels update the state cache in-place via ``input_output_aliases``.
    The decode kernel runs first, then its updated state and output are
    chained to the recurrent kernel.

    ``has_initial_state`` is consumed only by the recurrent kernel: decode
    tokens always have a valid prior state (a request must finish prefill
    before it can decode), so masking is unnecessary on the decode path.
    """
    # ── Decode kernel → updates state in-place ──
    o_d, state_1 = fused_decoding_gdn(
        q,
        k,
        v,
        g.astype(jnp.float32),
        initial_state.astype(jnp.float32),
        state_indices,
        distribution,
        b,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    # ── Recurrent kernel → updates state in-place ──
    o_r, state_2 = fused_recurrent_gdn(
        q,
        k,
        o_d,
        cu_seqlens,
        g.astype(jnp.float32),
        state_1,
        state_indices,
        b,
        has_initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o_r, state_2


# ── Public API ──


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
    ],
    donate_argnames=["v", "initial_state"],
)
def fused_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    cu_seqlens: jax.Array,  # [max_num_req+1] int32
    g: jax.Array,  # [T, H_v, K] or [T, H_v]
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None = None,  # [T, H_v] or None
    has_initial_state: jax.Array | None = None,  # [max_num_req] bool
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN forward pass.

    Supports GQA: ``H_v`` (value heads from ``v``) can be a multiple of
    ``H_qk`` (query/key heads from ``q``/``k``).  The kernel repeats
    q/k internally.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        cu_seqlens: Cumulative sequence lengths ``[max_num_req+1]``.
        g: Gating ``[T, H_v, K]`` or ``[T, H_v]`` (broadcast to K).
        initial_state: State cache ``[num_states, H_v, K, V]``.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        b: Raw betas ``[T, H_v]`` (sigmoid applied inside kernel). ``None`` means
          beta=1 (no beta gating).
        has_initial_state: Boolean tensor of shape ``[max_num_req]``. ``True``
          when the request's recurrent slot already holds a valid prior state
          (chunked-prefill continuation, prefix-cache hit, or running decode).
          ``False`` for brand-new prefills, whose slot is zeroed inside the
          recurrent kernel before the update so stale data from a previous tenant
          doesn't leak. ``None`` (default) is treated as all-True, preserving the
          pre-fix behaviour for callers that don't manage slot reuse.
        scale: Scale factor.  Default ``K ** -0.5``.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v]`` float32.
        dt_bias: Per-head bias ``[H_v]`` float32. Optional. Broadcast to ``[H_v,
          num_lanes]`` internally.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]`` with final states
        written back at the corresponding ``state_indices`` positions.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]

    # Broadcast g from [T, H_v] to [T, H_v, K] if needed.
    if g.shape == (T, H_v):
        g = jnp.broadcast_to(g[..., None], (T, H_v, K))
    elif g.shape != (T, H_v, K):
        raise ValueError(
            f"g shape {g.shape} must be [{T}, {H_v}, {K}] or [{T}, {H_v}]")

    # Validate pre-broadcast inputs.
    if b is not None and b.shape != (T, H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}]")
    if A_log is not None and A_log.shape != (H_v, ):
        raise ValueError(f"A_log shape {A_log.shape} must be [{H_v}]")
    if dt_bias is not None and dt_bias.shape != (H_v, ):
        raise ValueError(f"dt_bias shape {dt_bias.shape} must be [{H_v}]")

    cu_seqlens = cu_seqlens.astype(jnp.int32)
    state_indices = state_indices.astype(jnp.int32)

    if scale is None:
        scale = K**-0.5
    num_lanes = pltpu.get_tpu_info().num_lanes
    if b is not None:
        b = jnp.broadcast_to(b[:, :, None],
                             (T, H_v, num_lanes))  # [T, H_v, num_lanes]
    if dt_bias is not None:
        dt_bias = jnp.broadcast_to(dt_bias[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]
    distribution = distribution.astype(jnp.int32)

    if A_log is not None:
        A_log = jnp.broadcast_to(A_log[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]

    # Public contract is Boolean (matching the chunked / ref impls);
    # cast to int32 here for SMEM compatibility — the recurrent kernel
    # checks `has_init == 0` to decide whether to zero h0. Default to
    # all-True (no masking), matching the pre-fix behaviour.
    max_num_req = state_indices.shape[0]
    if has_initial_state is None:
        has_initial_state = jnp.ones((max_num_req, ), dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    o, state = _dispatch_with_distribution(
        q,
        k,
        v,
        cu_seqlens,
        g,
        initial_state,
        state_indices,
        b,
        has_initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o, state


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
    has_initial_state=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Adapter matching the ragged_gated_delta_rule_{ref,chunked} interface.

    Internally reshapes inputs and delegates to :func:`fused_gdn`.

    Args:
        mixed_qkv: ``(num_tokens, 2*n_kq*d_k + n_v*d_v)`` post-conv/silu.
        b: ``(num_tokens, n_v)`` — raw beta (sigmoid applied in kernel).
        a: ``(num_tokens, n_v)`` — raw alpha (gate transform in kernel).
        recurrent_state: ``(num_states, n_v, d_k, d_v)``.
        A_log: ``(n_v,)`` float32.
        dt_bias: ``(n_v,)`` float32.
        query_start_loc: ``(num_seqs+1,)`` int32.
        state_indices: ``(num_seqs,)`` int32.
        distribution: ``(3,)`` int32 — ``(decode_end, prefill_end, mixed_end)``.
        has_initial_state: Optional Boolean tensor of shape ``(max_reqs,)``.
          ``True`` when the request's slot already holds a valid prior recurrent
          state; ``False`` for brand-new prefills (the recurrent kernel zeros h0
          for those slots so stale data from a reused mamba slot doesn't leak).
          ``None`` (default) is treated as all-True, preserving the pre-PR-#2408
          behaviour. Pass it when you want the same stale-slot guard the chunked
          and ref impls already enforce.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.

    Returns:
        ``(updated_recurrent_state, output)`` where
        *updated_recurrent_state* is ``(num_states, n_v, d_k, d_v)`` and
        *output* is ``(num_tokens, n_v*d_v)``.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k

    q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
    k = mixed_qkv[..., key_dim:key_dim * 2].reshape(num_tokens, n_kq, d_k)
    v = mixed_qkv[..., key_dim * 2:].reshape(num_tokens, n_v, d_v)

    g = a

    # (decode_end, prefill_end, mixed_end) → (decode_end, total)
    fused_distribution = jnp.stack([distribution[0], distribution[2]])

    output, new_recurrent_state = fused_gdn(
        q,
        k,
        v,
        cu_seqlens=query_start_loc,
        g=g,
        initial_state=recurrent_state,
        state_indices=state_indices,
        distribution=fused_distribution,
        b=b,
        has_initial_state=has_initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    output = output.reshape(num_tokens, n_v * d_v)
    return new_recurrent_state, output


kernel = ragged_gated_delta_rule
