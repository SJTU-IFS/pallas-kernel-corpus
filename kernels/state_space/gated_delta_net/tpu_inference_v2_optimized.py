"""Standalone vLLM tpu-inference gated delta net kernels (v2).

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/gdn/v2/
  transformation: four upstream modules flattened into this file in dependency
    order -- compute_schedule_v2, recurrent_scan_impl, gdn_decode_kernel,
    recurrent_scan_v2.  Repo-local imports were dropped and
    `invert_triangular_matrix` was renamed per module (see below).

Same contract as the v1 kernels (``ragged_gated_delta_rule``), but v2 exposes
**two independent entry points** instead of one dispatching wrapper:

- ``ragged_gated_delta_rule_decode_only`` -- the decode-only path;
- ``recurrent_scan`` -- the chunked prefill / mixed path.

There is no v2 equivalent of v1's combined ``ragged_gated_delta_rule``, so a
caller picks the path rather than passing ``distribution`` to a dispatcher.
``kernel`` is bound to ``recurrent_scan``, the more general of the two.

**On the SiLU precondition.** The v1 wrapper silently expects post-SiLU input
(see ``baseline.PRE_SILU_INPUT``). v2's decode entry point makes it an explicit
``apply_silu`` argument, which is upstream resolving the same ambiguity the
corpus had to document by measurement. ``recurrent_scan`` has no such flag.

**Renamed on flattening.** ``invert_triangular_matrix`` is defined in both
``recurrent_scan_impl`` and ``recurrent_scan_v2`` with different bodies; they
are preserved as ``invert_triangular_matrix_impl`` and
``invert_triangular_matrix_scan``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/gdn/v2",
    "files": (
        "compute_schedule_v2.py",
        "recurrent_scan_impl.py",
        "gdn_decode_kernel.py",
        "recurrent_scan_v2.py",
    ),
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_gated_delta_rule",
    "launch_points": 2,
    "entry_points": ("ragged_gated_delta_rule_decode_only", "recurrent_scan"),
    "renamed_on_flatten": {
        "invert_triangular_matrix": (
            "invert_triangular_matrix_impl",
            "invert_triangular_matrix_scan",
        ),
    },
}

import dataclasses
import enum
import functools
import math
from typing import Any

import jax
from jax import lax
from jax._src import dtypes
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np

# --- from compute_schedule_v2.py ----------------------------------
def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Compute number of iterations in grid and work each iteration will do

  At high level
    - each iteration of grid is either prefill and or decode
    - grid moves in size of bt decode tokens (sequence) backwards starting from
    boundary
    - and prefill moves in chunk sized tokens forward from boundary to end
  Input characteristics
    - each sequence start and end may not be sublane aligned,
    boundary between decode and prefill maybe in shared sublane
    - sequence may not divide chunk size

  hardware req
    - offset for each block has to be sublane aligned

  So for this we have transition blocks at boundaries between prefill sequences,
  including first one with decode, token by token math is done here instead of
  chunk wise

  TODO: optimize table ,
    remove metadata which can be derived from other metadata or loop indices,
    like
        block offset can be derived from block idx and sequence start,
        block count can be derived from block idx and sequence start/end.
        also some metadata is only used for prefill or decode and can be stored
        in separate tables or encoded in same table with fewer bits.
        dtype of some metadata can be reduced to save space, for example
        block_is_first and block_is_last can be stored in 2 bits together,
        Sublane token by token metadata can be optimized by only storing
        boundaries
  """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (max_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)

    # =========================================================================
    # 1. Get each prefill sequence's effective start for chunkwise math
    # =========================================================================
    r_idx = jnp.arange(num_seqs)
    is_last_seq = r_idx == num_valid_seqs - 1
    seq_start = query_start_loc[:-1]
    seq_end = query_start_loc[1:]
    num_tokens = query_start_loc[num_valid_seqs]

    # create vector of sequence ends
    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )

    # if seq_len < sublane size
    is_decode_boundary = prev_seq_end == decode_tokens
    is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

    # compute the effective end of the rounded up to nearest sublane
    next_aligned_start = (seq_end // alignment) * alignment
    needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                        (~is_swallowed))

    is_decode_boundary = prev_seq_end == decode_tokens

    needs_start_transition = ((prev_seq_end % alignment != 0) & (~is_swallowed)
                              & is_decode_boundary)

    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)

    # Block counts per sequence
    num_regular_blocks = (effective_end - effective_start + chunk_size -
                          1) // chunk_size
    total_blocks_per_seq = (num_regular_blocks +
                            needs_transition.astype(jnp.int32) +
                            needs_start_transition.astype(jnp.int32))
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

    # Calculate the last perfectly aligned decode boundary
    is_pure_decode = seq_end <= decode_tokens
    total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)

    # Starting block index for each sequence
    base_idx = jnp.cumsum(total_blocks_per_seq) - total_blocks_per_seq
    total_prefill_blocks = jnp.sum(total_blocks_per_seq)

    # =========================================================================
    # 2. shows up as gathers
    # create block table
    # =========================================================================
    b_idx = jnp.arange(safe_max_blocks)
    prefill_valid_mask = b_idx < total_prefill_blocks

    # map grid index to sequence/request,
    # key for previous metadata arrays constructed to gather by sequence
    r_for_block = jnp.sum(b_idx[:, None] >= base_idx[None, :], axis=-1) - 1
    r_for_block = jnp.minimum(jnp.maximum(r_for_block, 0), num_seqs - 1)

    # index of block within blocks for a sequence
    local_b = b_idx - base_idx[r_for_block]

    start_trans_offset = (seq_start[r_for_block] // alignment) * alignment

    is_start_trans = needs_start_transition[r_for_block] & (local_b == 0)

    # Adjust local_b for regular blocks if there was a start transition
    adj_local_b = jnp.where(needs_start_transition[r_for_block], local_b - 1,
                            local_b)

    is_end_trans = needs_transition[r_for_block] & (
        adj_local_b == num_regular_blocks[r_for_block])

    reg_offset = effective_start[r_for_block] + adj_local_b * chunk_size
    reg_count = jnp.minimum(chunk_size,
                            effective_end[r_for_block] - reg_offset)
    #   reg_is_last = reg_offset + reg_count >= seq_end[r_for_block]
    #   reg_is_first = reg_offset == seq_start[r_for_block]

    trans_offset = next_aligned_start[r_for_block]

    # Apply predication
    block_offset = jnp.where(
        is_start_trans,
        start_trans_offset,
        jnp.where(is_end_trans, trans_offset, reg_offset),
    )

    block_count = jnp.where(
        is_start_trans,
        effective_start[r_for_block] - seq_start[r_for_block],
        jnp.where(is_end_trans, alignment, reg_count),
    )

    is_trans_block = is_start_trans | is_end_trans

    # =========================================================================
    # 3. Metadata for shared sublane tiles
    # =========================================================================
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    fixed_query_start_loc = jnp.where(valid_loc_mask, query_start_loc,
                                      last_valid_loc)
    glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

    # [safe_max_blocks, sublane size, num_seqs]
    valid_mask = glob_idxs < num_tokens
    t_reqs = (
        jnp.sum(glob_idxs[:, :, None] >= fixed_query_start_loc[None, None, :],
                axis=-1) - 1)
    # there could be padding in query_start_loc
    last_valid_seq = jnp.max(
        jnp.where(total_blocks_per_seq > 0, jnp.arange(num_seqs), -1))
    t_reqs = jnp.where(valid_mask, t_reqs, last_valid_seq)
    t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

    is_first_tok = (glob_idxs == query_start_loc[t_reqs]).astype(jnp.int32)
    is_last_tok = (glob_idxs == query_start_loc[t_reqs + 1] - 1).astype(
        jnp.int32)

    # =========================================================================
    # 4. Decode blocks metadata
    # =========================================================================
    decode_valid_mask = b_idx < num_decode_batches
    decode_batch_idx = jnp.where(decode_valid_mask,
                                 (num_decode_batches - 1) - b_idx, 0)
    decode_offsets = decode_batch_idx * BT
    decode_req_ids = decode_batch_idx * BT
    decode_counts = jnp.where(decode_valid_mask,
                              jnp.minimum(BT, decode_tokens - decode_offsets),
                              0)

    # Mask out invalid prefill
    prefill_valid_ints = prefill_valid_mask.astype(jnp.int32)
    block_offset = jnp.where(prefill_valid_mask, block_offset, 0)
    r_for_block = jnp.where(prefill_valid_mask, r_for_block, 0)
    block_count = jnp.where(prefill_valid_mask, block_count, 0)
    block_is_first = block_offset <= seq_start[r_for_block]
    block_is_last = (block_offset + block_count) >= seq_end[r_for_block]
    block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
    block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
    is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
    t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
    is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
    is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

    # =========================================================================
    # 5. Merge all
    # =========================================================================
    # Columns mapping:
    # 0: prefill_valid_ints - 1 if this grid block has valid prefill work,
    # .                  0 otherwise
    # 1: block_offset - start index of prefill start in tile, usually 0
    #                     but in shared sublane case its not
    # 2: r_for_block - request ID (sequence index) this prefill block belongs to
    # 3: block_count - number of valid tokens in this prefill block
    # 4: decode_valid_mask - 1 if this step has valid decode work, 0 otherwise
    # 5: decode_offsets - start index for the decode batch
    # 6: decode_req_ids - starting request ID in decode batch
    # 7: decode_counts - number of valid decode requests in this batch
    # 8: block_is_last - 1 if this is the last block for the request, 0 otherwise
    # 9: block_is_first - 1 if first block for request, 0 otherwise
    # 10: is_trans_block - 1 if this is a transition block, 0 otherwise
    cols = [
        prefill_valid_ints,  # 0
        block_offset,  # 1
        r_for_block,  # 2
        block_count,  # 3
        decode_valid_mask.astype(jnp.int32),  # 4
        decode_offsets,  # 5
        decode_req_ids,  # 6
        decode_counts,  # 7
        block_is_last.astype(jnp.int32),  # 8
        block_is_first.astype(jnp.int32),  # 9
        is_trans_block.astype(jnp.int32),  # 10
    ]

    # 11 to 11 + alignment - 1: Request ID for each token in the sublane tile
    for i in range(alignment):
        cols.append(t_reqs[:, i])  # e.g., 11-18 if alignment=8
    # 11 + alignment to 11 + 2*alignment - 1: 1 if token is first in request
    for i in range(alignment):
        cols.append(is_first_tok[:, i])  # e.g., 19-26
    # 11 + 2*alignment to 11 + 3*alignment - 1: 1 if token is last in request
    for i in range(alignment):
        cols.append(is_last_tok[:, i])  # e.g., 27-34

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks


# --- from recurrent_scan_impl.py ----------------------------------
# pylint: disable=invalid-name




def l2_normalize(x, eps=1e-6):
    rnorm = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * rnorm


# 1. Dataclasses for holding references to inputs/outputs and shared data.
# These are passed as arguments to the processor classes.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchRefs:
    """Inputs/Outputs specific to a single execution branch (prefill or decode)."""

    qkv: Any
    a_raw: Any
    b_raw: Any
    output: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SharedRefs:
    """Inputs/Outputs refs shared between both branches(prefill and decode)."""

    a_log: Any
    dt_bias: Any
    recurrent_state_in: Any
    recurrent_state_out: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrefillScratchRefs:
    """Scratch VMEM and semaphores allocated for prefill."""

    scratch: Any
    semaphore: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DecodeScratchRefs:
    """Scratch VMEM and semaphores allocated for decode."""

    state: Any
    load: Any
    store: Any
    output: Any
    read_semaphores: Any
    write_semaphore: Any


# 2.  Configuration Dataclasses (static)


@dataclasses.dataclass(frozen=True)
class ModelDims:
    """Dimensions of the Recurrent Scan model configuration."""

    n_kq: int
    n_v: int
    d_k: int
    d_v: int

    @property
    def key_dim(self) -> int:
        return self.n_kq * self.d_k

    @property
    def repeat_factor(self) -> int:
        return self.n_v // self.n_kq


@dataclasses.dataclass(frozen=True)
class TilingConfig:
    """Tiling dimensions for memory copy blocks."""

    C: int
    BT: int
    sublanesize: int


@dataclasses.dataclass(frozen=True)
class ScanConfig:
    """Configuration holding model dimensions and tiling options."""

    model: ModelDims
    tiling: TilingConfig
    use_qk_norm_in_gdn: bool
    decode_tokens: int


# 3. DMA Helper


class DMAHelper:
    """Manages asynchronous state copies and double-buffering semaphores."""

    def __init__(self, state_in, state_out, commit_scratch, semaphore):
        self.state_in = state_in
        self.state_out = state_out
        self.commit_scratch = commit_scratch
        self.sem = semaphore
        # double or n buffering
        self.has_multiple_slots = commit_scratch.shape[0] > 1

    def build_copy_in(self, slot: int, state_idx: int):
        target_slot = slot if self.has_multiple_slots else 0
        return pltpu.make_async_copy(
            src_ref=self.state_in.at[pl.ds(state_idx, 1)],
            dst_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            sem=self.sem.at[slot],
        )

    def commit_in(self, copy_op, slot: int, dst_ref, dst_slot: int):
        target_slot = slot if self.has_multiple_slots else 0
        copy_op.wait()
        dst_ref[dst_slot] = self.commit_scratch[target_slot].astype(
            dst_ref.dtype)

    def copy_out(self, slot: int, state_idx: int, src_scratch):
        target_slot = slot if self.has_multiple_slots else 0
        self.commit_scratch[target_slot] = src_scratch.astype(
            self.commit_scratch.dtype)
        copy_op = pltpu.make_async_copy(
            src_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            dst_ref=self.state_out.at[pl.ds(state_idx, 1)],
            sem=self.sem.at[slot],
        )
        copy_op.start()
        return copy_op

    def wait_out(self, slot: int, state_idx: int):
        target_slot = slot if self.has_multiple_slots else 0
        copy_op = pltpu.make_async_copy(
            src_ref=self.commit_scratch.at[pl.ds(target_slot, 1)],
            dst_ref=self.state_out.at[pl.ds(state_idx, 1)],
            sem=self.sem.at[slot],
        )
        copy_op.wait()


# 4. Schedule step helper
COL_PREFILL_VALID = 0
COL_PREFILL_OFFSET = 1
COL_PREFILL_REQ_ID = 2
COL_PREFILL_COUNT = 3
COL_DECODE_VALID = 4
COL_DECODE_OFFSET = 5
COL_DECODE_REQ_ID = 6
COL_DECODE_COUNT = 7
COL_IS_LAST_CHUNK = 8
COL_IS_FIRST_CHUNK = 9
COL_IS_TRANSITION = 10
COL_SUBLANE_REQ_IDS = 11


class ScheduleStep:
    """Unpacks and holds the scheduling metadata for the current step."""

    def __init__(self, schedule_table, step):
        self.step = step
        self.schedule_table = schedule_table
        self.prefill_valid = schedule_table[step, COL_PREFILL_VALID][...]
        self.prefill_offset = schedule_table[step, COL_PREFILL_OFFSET][...]
        self.prefill_req_id = schedule_table[step, COL_PREFILL_REQ_ID][...]
        self.prefill_count = schedule_table[step, COL_PREFILL_COUNT][...]

        self.decode_valid = schedule_table[step, COL_DECODE_VALID][...]
        self.decode_offset = schedule_table[step, COL_DECODE_OFFSET][...]
        self.decode_req_id = schedule_table[step, COL_DECODE_REQ_ID][...]
        self.decode_count = schedule_table[step, COL_DECODE_COUNT][...]

        self.is_last_chunk = schedule_table[step, COL_IS_LAST_CHUNK][...]
        self.is_first_chunk = schedule_table[step, COL_IS_FIRST_CHUNK][...]
        self.is_transition = schedule_table[step, COL_IS_TRANSITION][...]


# 4. Base Processor Class


class ScanProcessor:
    """Base class for executing step calculations."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
    ):
        self.cfg = config
        self.schedule = schedule
        self.state_indices = state_indices
        self.has_initial_state = has_initial_state


def invert_triangular_matrix_impl(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


class PrefillProcessor(ScanProcessor):
    """Handles prefill step processing."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
        refs: BranchRefs,
        shared: SharedRefs,
        scratch: PrefillScratchRefs,
        dma: DMAHelper,
    ):
        super().__init__(config, schedule, state_indices, has_initial_state)
        self.refs = refs
        self.shared = shared
        self.scratch = scratch
        self.dma = dma

    def process(self):
        is_trans = self.schedule.is_transition > 0
        jax.lax.cond(
            is_trans,
            lambda _: self._process_transition_prefill(),
            lambda _: self._process_regular_prefill(),
            operand=None,
        )

    def _process_regular_prefill(self):
        """Processes a regular prefill step without transition boundary overlaps."""
        prefill_req_id = self.schedule.prefill_req_id
        prefill_slot = prefill_req_id % 2
        init_has_init = self.has_initial_state[prefill_req_id][...]
        init_state_idx = self.state_indices[prefill_req_id][...]
        should_load_init = (self.schedule.is_first_chunk > 0) & (init_has_init
                                                                 > 0)
        should_zero_init = (self.schedule.is_first_chunk > 0) & (init_has_init
                                                                 == 0)

        init_copy_op = self.dma.build_copy_in(prefill_slot, init_state_idx)

        @pl.when(should_load_init)
        def _start_init_load():
            init_copy_op.start()

        @pl.when(should_zero_init)
        def _zero_init_state():
            self.scratch.scratch[prefill_slot] = jnp.zeros(
                (self.cfg.model.n_v, self.cfg.model.d_k, self.cfg.model.d_v),
                dtype=self.scratch.scratch.dtype,
            )

        key_dim = self.cfg.model.key_dim
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq
        C = self.cfg.tiling.C

        qkv_chunk = self.refs.qkv[...]
        qkv_chunk = jax.nn.silu(qkv_chunk)
        q = qkv_chunk[:, :key_dim]
        k = qkv_chunk[:, key_dim:2 * key_dim]
        v = qkv_chunk[:, 2 * key_dim:]

        a_raw_chunk = self.refs.a_raw[...]
        b_raw_chunk = self.refs.b_raw[...]

        a_raw_processed_T = a_raw_chunk[:, :n_v].T
        b_raw_processed_T = b_raw_chunk[:, :n_v].T

        beta_T = jax.nn.sigmoid(b_raw_processed_T)
        g_T = -jnp.exp(
            # in jax 10.0.1 we can avoid the cast to float32,
            # jax.errors.JaxRuntimeError: INTERNAL: Mosaic failed to compile TPU kernel: failed to legalize operation
            # 'math.log1p': %7302 = "math.log1p"(%7295) <{fastmath =
            # #arith.fastmath<none>}> : (vector<8x128x2xbf16>) -> vector<8x128x2xbf16>
            self.shared.a_log[...])[:, None] * jax.nn.softplus(
                # same issue with the cast here
                a_raw_processed_T + self.shared.dt_bias[...][:, None])
        g_T = jnp.maximum(g_T, -100.0)

        prefill_count = self.schedule.prefill_count
        mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
        q = jnp.where(mask_float[:, None] > 0, q, 0.0)
        k = jnp.where(mask_float[:, None] > 0, k, 0.0)
        g_T = jnp.where(mask_float[None, :] > 0, g_T, 0.0)
        v = jnp.where(mask_float[:, None] > 0, v, 0.0)
        beta_T = jnp.where(mask_float[None, :] > 0, beta_T, 0.0)

        q = q.reshape(C, n_kq, d_k)
        k = k.reshape(C, n_kq, d_k)
        v = v.reshape(C, n_v, d_v)

        if self.cfg.use_qk_norm_in_gdn:
            q = l2_normalize(q)
            k = l2_normalize(k)

        # Note: fusing transpose with (vmatpush.xpose) made it slower,
        # This has better instruction pipelining
        q_T = q.transpose(1, 0, 2)  # (n_kq, C, d_k)
        k_T = k.transpose(1, 0, 2)  # (n_kq, C, d_k)
        v_T = v.transpose(1, 0, 2)  # (n_v, C, d_v)

        repeat_factor = self.cfg.model.repeat_factor
        if repeat_factor > 1:
            q_T = jnp.repeat(q_T, repeat_factor, axis=0)
            k_T = jnp.repeat(k_T, repeat_factor, axis=0)

        scale = d_k**-0.5
        q_T = q_T * scale

        g_cumsum_list = []
        current_sum = jnp.zeros((n_v, ), dtype=jnp.float32)
        for i in range(C):
            current_sum = current_sum + g_T[:, i]
            g_cumsum_list.append(current_sum)
        g_cumsum_T = jnp.stack(g_cumsum_list, axis=1)  # shape (n_v, C)
        exp_g = jnp.exp(g_cumsum_T)[..., None]  # Precomputed for reuse
        k_beta = k_T * beta_T[..., None]

        # Concatenate along sequence dimension: (n_v, 2 * C, d_k)
        kbeta_q = jnp.concatenate([k_beta, q_T], axis=1)
        # Batch is n_v (axis 0), contract is d_k (axis 2).
        # Output shape: (n_v, 2 * C, C)
        S_both = jax.lax.dot_general(
            kbeta_q,
            k_T,
            (((2, ), (2, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )
        S = S_both[:, :C, :]
        S_q = S_both[:, C:, :]

        g_diff = g_cumsum_T[..., :, None] - g_cumsum_T[..., None, :]
        i_idx = jnp.arange(C)[:, None]
        j_idx = jnp.arange(C)[None, :]
        mask_float = (i_idx > j_idx).astype(jnp.float32)

        g_diff_safe = jnp.minimum(g_diff, 0.0)
        S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe),
                      0.0)

        mask_float_q = (i_idx >= j_idx).astype(jnp.float32)
        g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (
            1.0 - mask_float_q[None, ...]) * (-1e30)
        S_q = S_q * jnp.exp(g_diff_Sq)
        S_q = S_q * mask_float_q[None, ...]

        I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
        A_inv = invert_triangular_matrix_impl(I_plus_S, block_size=16)

        v_beta = v_T * beta_T[..., None]
        k_beta_g = k_beta * exp_g
        vk_in = jnp.concatenate(
            [
                v_beta,
                k_beta_g,
            ],
            axis=2,
        )  # (n_v, C, d_v + d_k)
        uw = jax.lax.dot_general(
            A_inv,
            vk_in,
            (((2, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.HIGHEST,
        )  # Output shape: (n_v, C, d_v + d_k)
        u = uw[..., :d_v]
        w = uw[..., d_v:]

        q_g = q_T * exp_g  # (n_v, C, d_k)

        @pl.when(should_load_init)
        def _finish_init_load():
            self.dma.commit_in(init_copy_op, prefill_slot,
                               self.scratch.scratch, prefill_slot)

        current_state = self.scratch.scratch[prefill_slot]  # (n_v, d_k, d_v)

        qw = jnp.concatenate([q_g, w], axis=1)  # (n_v, 2 * C, d_k)
        comb = jax.lax.dot_general(
            qw,
            current_state.astype(jnp.float32),
            (((2, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.DEFAULT,
        )  # Output shape: (n_v, 2 * C, d_v)
        attn_inter, v_prime = jnp.split(comb, 2, axis=1)

        v_new = u - v_prime
        term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
        o_c = attn_inter + term2  # (n_v, C, d_v)

        g_i_last_exp = exp_g[:, -1, None]
        g_diff_exp_state = jnp.exp(g_cumsum_T[..., -1, None] -
                                   g_cumsum_T)[..., None]
        k_i_g_diff = k_T * g_diff_exp_state
        update_term = jax.lax.dot_general(
            k_i_g_diff,
            v_new,
            (((1, ), (1, )), ((0, ), (0, ))),
            precision=jax.lax.Precision.DEFAULT,
        )  # Output shape: (n_v, d_k, d_v)
        h_new = current_state * g_i_last_exp + update_term

        self.scratch.scratch[prefill_slot] = h_new.astype(
            self.scratch.scratch.dtype)

        store_state_idx = self.state_indices[prefill_req_id][...]

        @pl.when(self.schedule.is_last_chunk > 0)
        def store_state():
            copy_op = self.dma.copy_out(prefill_slot, store_state_idx,
                                        self.scratch.scratch[prefill_slot])
            copy_op.wait()

        o_c_tr = o_c.transpose(1, 0, 2)
        o_c_flat = o_c_tr.reshape(C, n_v * d_v)

        mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
        o_c_flat_masked = o_c_flat * mask_float[:, None]
        self.refs.output[...] = o_c_flat_masked.astype(self.refs.output.dtype)

    def _process_transition_prefill(self):
        """Processes a transition prefill step with sublane stitching."""
        C_trans = self.cfg.tiling.sublanesize
        key_dim = self.cfg.model.key_dim
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq

        first_req_id = self.schedule.schedule_table[self.schedule.step,
                                                    COL_SUBLANE_REQ_IDS][...]
        first_is_first = self.schedule.schedule_table[self.schedule.step,
                                                      COL_SUBLANE_REQ_IDS +
                                                      C_trans][...]
        first_slot = first_req_id % 2
        first_has_init = self.has_initial_state[first_req_id][...]
        should_load_first = (first_is_first > 0) & (first_has_init > 0)
        first_state_idx = self.state_indices[first_req_id][...]

        first_copy_op = self.dma.build_copy_in(first_slot, first_state_idx)

        @pl.when(should_load_first)
        def _start_first_load():
            first_copy_op.start()

        qkv_chunk = self.refs.qkv[:C_trans, :]
        qkv_chunk = jax.nn.silu(qkv_chunk)
        q = qkv_chunk[:, :key_dim]
        k = qkv_chunk[:, key_dim:2 * key_dim]
        v = qkv_chunk[:, 2 * key_dim:]

        a_raw_chunk = self.refs.a_raw[...]
        b_raw_chunk = self.refs.b_raw[...]

        a_raw_processed_T = a_raw_chunk[:C_trans, :n_v].T
        b_raw_processed_T = b_raw_chunk[:C_trans, :n_v].T

        beta_chunk_T = jax.nn.sigmoid(b_raw_processed_T)
        g_chunk_T = -jnp.exp(
            self.shared.a_log[...])[:, None] * jax.nn.softplus(
                a_raw_processed_T + self.shared.dt_bias[...][:, None])
        g_chunk_T = jnp.maximum(g_chunk_T, -100.0)

        q = q.reshape(C_trans, n_kq, d_k)
        k = k.reshape(C_trans, n_kq, d_k)
        v = v.reshape(C_trans, n_v, d_v)

        if self.cfg.use_qk_norm_in_gdn:
            q = l2_normalize(q)
            k = l2_normalize(k)

        repeat_factor = self.cfg.model.repeat_factor
        if repeat_factor > 1:
            q = jnp.repeat(q, repeat_factor, axis=1)
            k = jnp.repeat(k, repeat_factor, axis=1)

        scale = d_k**-0.5
        q = q * scale

        @pl.when((first_is_first > 0) & (first_has_init == 0))
        def _zero_first_slot():
            self.scratch.scratch[first_slot] = jnp.zeros(
                (n_v, d_k, d_v), dtype=self.scratch.scratch.dtype)

        @pl.when(should_load_first)
        def _finish_first_load():
            self.dma.commit_in(first_copy_op, first_slot, self.scratch.scratch,
                               first_slot)

        h = self.scratch.scratch[first_slot]
        current_r = first_req_id
        sequence_valid = True
        exp_g_chunk_T = jnp.exp(g_chunk_T)

        for i in range(C_trans):
            t_req = self.schedule.schedule_table[self.schedule.step,
                                                 11 + i][...]
            t_is_first = self.schedule.schedule_table[self.schedule.step,
                                                      11 + C_trans + i][...]
            t_is_last = self.schedule.schedule_table[self.schedule.step,
                                                     11 + 2 * C_trans + i][...]

            is_new_seq = t_req != current_r
            sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

            is_decode_token = t_req < self.cfg.decode_tokens
            sequence_valid = jnp.where(is_decode_token, False, sequence_valid)

            c_slot = current_r % 2
            self.scratch.scratch[c_slot] = h

            def do_write(c_slot=c_slot, current_r=current_r, h=h):
                state_idx = self.state_indices[current_r][...]
                copy_op = self.dma.copy_out(c_slot, state_idx, h)
                copy_op.wait()
                return None

            is_current_r_prefill = current_r >= self.cfg.decode_tokens
            should_write = is_current_r_prefill & is_new_seq
            jax.lax.cond(should_write, do_write, lambda: None)

            t_slot = t_req % 2
            t_has_init = self.has_initial_state[t_req][...]

            def load_t_state(t_slot=t_slot, t_req=t_req):
                state_idx = self.state_indices[t_req][...]
                copy_op = self.dma.build_copy_in(t_slot, state_idx)
                copy_op.start()
                self.dma.commit_in(copy_op, t_slot, self.scratch.scratch,
                                   t_slot)

            should_load_t = (t_is_first > 0) & (t_has_init > 0)
            jax.lax.cond(should_load_t, load_t_state, lambda: None)

            should_zero = (t_is_first > 0) & (t_has_init == 0)

            def zero_t_slot(t_slot=t_slot, n_v=n_v, d_k=d_k, d_v=d_v):
                self.scratch.scratch[t_slot] = jnp.zeros(
                    (n_v, d_k, d_v), dtype=self.scratch.scratch.dtype)

            jax.lax.cond(should_zero, zero_t_slot, lambda: None)

            h = self.scratch.scratch[t_slot]
            current_r = t_req

            k_i = k[i, :, :]
            v_i = v[i, :, :]
            beta_i = beta_chunk_T[:, i]
            q_i = q[i, :, :]

            decay = exp_g_chunk_T[:, i][..., None]

            k_state = jnp.sum(k_i[..., None] * h, axis=1)
            v_diff = v_i - decay * k_state
            v_new = beta_i[:, None] * v_diff

            q_state = jnp.sum(q_i[..., None] * h, axis=1)
            q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

            out_i = decay * q_state + q_k * v_new

            k_v_new = k_i[..., None] * v_new[:, None, :]
            h_new = h * decay[..., None] + k_v_new

            h = jnp.where(sequence_valid, h_new, h)
            out_i = jnp.where(sequence_valid, out_i, 0.0)

            sequence_valid = jnp.where(t_is_last > 0, False, sequence_valid)

            self.refs.output[i, :] = out_i.reshape(n_v * d_v).astype(
                self.refs.output.dtype)

        final_slot = current_r % 2
        self.scratch.scratch[final_slot] = h

        is_current_r_prefill = current_r >= self.cfg.decode_tokens

        @pl.when(is_current_r_prefill)
        def do_final_write():
            state_idx = self.state_indices[current_r][...]
            copy_op = self.dma.copy_out(final_slot, state_idx, h)
            copy_op.wait()
            return None


class DecodeProcessor(ScanProcessor):
    """Handles batch decode step processing using double-buffering logic."""

    def __init__(
        self,
        config: ScanConfig,
        schedule: ScheduleStep,
        state_indices,
        has_initial_state,
        refs: BranchRefs,
        shared: SharedRefs,
        scratch: DecodeScratchRefs,
        dma: DMAHelper,
    ):
        super().__init__(config, schedule, state_indices, has_initial_state)
        self.refs = refs
        self.shared = shared
        self.scratch = scratch
        self.dma = dma

    def get_target_idx(self, b):
        safe_req_id = jnp.minimum(self.schedule.decode_req_id + b,
                                  self.state_indices.shape[0] - 1)
        return self.state_indices[safe_req_id][...]

    def process(self):
        """Processes decode steps in blocks."""
        decode_count = self.schedule.decode_count
        BT = self.cfg.tiling.BT
        n_v = self.cfg.model.n_v
        d_k = self.cfg.model.d_k
        d_v = self.cfg.model.d_v
        n_kq = self.cfg.model.n_kq
        key_dim = self.cfg.model.key_dim
        repeat_factor = self.cfg.model.repeat_factor
        use_qk_norm_in_gdn = self.cfg.use_qk_norm_in_gdn
        exp_a_log = jnp.exp(self.shared.a_log[...].astype(jnp.float32))
        dt_bias_f32 = self.shared.dt_bias[...].astype(jnp.float32)

        # Pre-loop: kick off async loads for iters 0 and 1.
        @pl.when(decode_count >= 1)
        def _preload_slot_0():
            tgt = self.get_target_idx(0)
            op = pltpu.make_async_copy(
                src_ref=self.shared.recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=self.scratch.load.at[pl.ds(0, 1)],
                sem=self.scratch.read_semaphores.at[0],
            )
            op.start()

        @pl.when(decode_count >= 2)
        def _preload_slot_1():
            tgt = self.get_target_idx(1)
            op = pltpu.make_async_copy(
                src_ref=self.shared.recurrent_state_in.at[pl.ds(tgt, 1)],
                dst_ref=self.scratch.load.at[pl.ds(1, 1)],
                sem=self.scratch.read_semaphores.at[1],
            )
            op.start()

        def process_decode_step(b, store_inflight):
            s0_inflight, s1_inflight = store_inflight
            is_valid = b < decode_count
            slot = b % 2
            using_slot_0 = slot == 0
            cur_slot_inflight = jax.lax.select(using_slot_0, s0_inflight,
                                               s1_inflight)

            @pl.when(is_valid)
            def do_work():
                # Wait for THIS iter's load.
                wait_load = pltpu.make_async_copy(
                    src_ref=self.shared.recurrent_state_in.at[pl.ds(0, 1)],
                    dst_ref=self.scratch.load.at[pl.ds(slot, 1)],
                    sem=self.scratch.read_semaphores.at[slot],
                )
                wait_load.wait()

                self.scratch.state[pl.ds(0, 1)] = self.scratch.load[pl.ds(
                    slot, 1)][...]

                # Prefetch load for iter b+2.
                next_b = b + 2

                @pl.when(next_b < decode_count)
                def _prefetch_next_load():
                    next_tgt = self.get_target_idx(next_b)
                    op = pltpu.make_async_copy(
                        src_ref=self.shared.recurrent_state_in.at[pl.ds(
                            next_tgt, 1)],
                        dst_ref=self.scratch.load.at[pl.ds(slot, 1)],
                        sem=self.scratch.read_semaphores.at[slot],
                    )
                    op.start()

                target_idx = self.get_target_idx(b)

                sublanesize = self.cfg.tiling.sublanesize
                b_aligned = (b // sublanesize) * sublanesize

                qkv_block_data = self.refs.qkv[
                    pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
                    qkv_block_data.dtype)[:, None]
                qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)

                # Fused SiLU
                qkv_row = jax.nn.silu(qkv_row)
                q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
                k = qkv_row[:, key_dim:2 * key_dim].reshape(n_kq, d_k)
                v = qkv_row[:, 2 * key_dim:].reshape(n_v, d_v)

                if use_qk_norm_in_gdn:
                    q = l2_normalize(q)
                    k = l2_normalize(k)

                # Head repetition
                if repeat_factor > 1:
                    q = jnp.repeat(q, repeat_factor, axis=0)
                    k = jnp.repeat(k, repeat_factor, axis=0)

                scale = d_k**-0.5
                q = q * scale

                g_block_new = self.refs.a_raw[pl.ds(b_aligned, sublanesize), :]
                beta_block_new = self.refs.b_raw[
                    pl.ds(b_aligned, sublanesize), :]

                mask_new = (jnp.arange(sublanesize) == (
                    b % sublanesize)).astype(g_block_new.dtype)[:, None]

                a_raw_new = jnp.sum(g_block_new * mask_new,
                                    axis=0,
                                    keepdims=True)[0, :n_v].astype(jnp.float32)
                b_raw_new = jnp.sum(beta_block_new * mask_new,
                                    axis=0,
                                    keepdims=True)[0, :n_v].astype(jnp.float32)

                # Compute gate
                curr_beta = jax.nn.sigmoid(b_raw_new)
                curr_g = -exp_a_log * jax.nn.softplus(a_raw_new + dt_bias_f32)
                curr_g = jnp.maximum(curr_g, -100.0)
                decay = jnp.exp(curr_g)

                current_state = self.scratch.state[0]

                # 1. Batched dot product: k @ state -> (n_v, d_v)
                k_state = jax.lax.dot_general(
                    k.reshape(n_v, 1, d_k),
                    current_state,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(n_v, d_v)

                decay_k_state = jnp.where(
                    jnp.isinf(k_state),
                    0.0,
                    decay[:, None] * k_state,
                )
                v_diff = v - decay_k_state
                v_new = curr_beta[:, None] * v_diff

                # 2. Batched dot product: q @ state -> (n_v, d_v)
                q_state = jax.lax.dot_general(
                    q.reshape(n_v, 1, d_k),
                    current_state,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(n_v, d_v)

                q_k = jnp.sum(
                    q * k,
                    axis=-1,
                    keepdims=True,
                )

                decay_q_state = jnp.where(
                    jnp.isinf(q_state),
                    0.0,
                    decay[:, None] * q_state,
                )
                out_step = decay_q_state + q_k * v_new

                # 3. Outer product and decay update for state -> (n_v, d_k, d_v)
                decay_state = jnp.where(
                    jnp.isinf(current_state),
                    0.0,
                    current_state * decay[:, None, None],
                )
                k_v_new = k[:, :, None] * v_new[:, None, :]
                new_state = decay_state + k_v_new
                self.scratch.store[slot] = new_state.astype(
                    self.scratch.store.dtype)

                # Accumulate output in scratchpad
                current_output = self.scratch.output[...]
                mask = (jnp.arange(BT) == b).astype(current_output.dtype)[:,
                                                                          None]
                new_output = jnp.where(
                    mask,
                    out_step.reshape(1,
                                     n_v * d_v).astype(current_output.dtype),
                    current_output,
                )
                self.scratch.output[...] = new_output

                # Async store. Before writing to decode_store_scratch[slot],
                # wait for the previous same-slot store DMA (from iter b-2)
                @pl.when(cur_slot_inflight > 0)
                def _wait_same_slot_store():
                    copy_op = pltpu.make_async_copy(
                        src_ref=self.scratch.store.at[pl.ds(slot, 1)],
                        dst_ref=self.shared.recurrent_state_out.at[pl.ds(0,
                                                                         1)],
                        sem=self.scratch.write_semaphore.at[slot],
                    )
                    copy_op.wait()

                copy_op = pltpu.make_async_copy(
                    src_ref=self.scratch.store.at[pl.ds(slot, 1)],
                    dst_ref=self.shared.recurrent_state_out.at[pl.ds(
                        target_idx, 1)],
                    sem=self.scratch.write_semaphore.at[slot],
                )
                copy_op.start()

            next_s0_inflight = jax.lax.select(
                is_valid & using_slot_0,
                jnp.int32(1),
                s0_inflight,
            )
            next_s1_inflight = jax.lax.select(
                is_valid & (~using_slot_0),
                jnp.int32(1),
                s1_inflight,
            )
            return (next_s0_inflight, next_s1_inflight)

        final_s0_inflight, final_s1_inflight = jax.lax.fori_loop(
            0,
            BT,
            process_decode_step,
            (jnp.int32(0), jnp.int32(0)),
        )

        # Drain any remaining async store DMAs
        @pl.when(final_s0_inflight > 0)
        def _drain_slot_0():
            temp_desc = pltpu.make_async_copy(
                src_ref=self.scratch.store.at[pl.ds(0, 1)],
                dst_ref=self.shared.recurrent_state_out.at[pl.ds(0, 1)],
                sem=self.scratch.write_semaphore.at[0],
            )
            temp_desc.wait()

        @pl.when(final_s1_inflight > 0)
        def _drain_slot_1():
            temp_desc = pltpu.make_async_copy(
                src_ref=self.scratch.store.at[pl.ds(1, 1)],
                dst_ref=self.shared.recurrent_state_out.at[pl.ds(0, 1)],
                sem=self.scratch.write_semaphore.at[1],
            )
            temp_desc.wait()

        mask = (jnp.arange(BT)
                < decode_count).astype(self.scratch.output.dtype)[:, None]
        decode_output_scratch_masked = self.scratch.output[...] * mask
        self.refs.output[...] = decode_output_scratch_masked


# --- from gdn_decode_kernel.py ------------------------------------
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


def get_default_block_sizes(
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

    Accounts for state scratch ``(bt, H_v, K, V)`` ``state_dtype``, optional
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
    distribution_ref,  # [3] int32 (SMEM)
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
            h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
            h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
            h_load_sems_s.at[buf_idx],
        ).wait()

        # ── Step 3: Compute ──
        # Inputs are sliced and processed inside the loop to avoid vreg spill to vmem

        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _process_token():
                h0 = h_bufs_s[buf_idx, i_t].astype(jnp.float32)

                # Slice and process inputs for current token
                q_t = q_ref[i_t]
                k_t = k_ref[i_t]
                v_t = v_ref[i_t]
                g_t = g_ref[i_t]

                if apply_silu:
                    q_t = jax.nn.silu(q_t)
                    k_t = jax.nn.silu(k_t)
                    v_t = jax.nn.silu(v_t)

                if use_qk_l2norm:
                    q_t = q_t / jnp.sqrt(
                        jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
                    k_t = k_t / jnp.sqrt(
                        jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
                q_t = q_t * scale

                qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)

                if repeat_factor > 1:
                    q_t = jnp.repeat(q_t, repeat_factor, axis=0)
                    k_t = jnp.repeat(k_t, repeat_factor, axis=0)
                    qk_dot = jnp.repeat(qk_dot, repeat_factor, axis=0)

                if b_ref is not None:
                    b_t = b_ref[i_t].astype(jnp.float32)
                    if V > b_t.shape[-1]:
                        beta_t = jax.nn.sigmoid(
                            jnp.concatenate([b_t] * (V // b_t.shape[-1]),
                                            axis=-1))
                    else:
                        beta_t = jax.nn.sigmoid(b_t)

                if use_gate_in_kernel:
                    g_val = g_t
                    if dt_bias_ref is not None:
                        g_val = g_val + dt_bias_val
                    if lower_bound is not None:
                        gk = lower_bound / (1.0 +
                                            jnp.exp(-(a_val[:, None] * g_val)))
                    else:
                        gk = -a_val[:, None] * jax.nn.softplus(
                            g_val.astype(jnp.float32)).astype(g_val.dtype)
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
                    b_v = beta_t * v_diff
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
                h_bufs_s.at[buf_idx,
                            pl.ds(0, prev_block_len), :, :, :],
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
        h_bufs.at[last_buf_idx,
                  pl.ds(0, last_block_len), :, :, :],
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
            h_bufs.at[other_buf_idx,
                      pl.ds(0, other_block_len), :, :, :],
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
    initial_state: jax.Array,  # [num_states, H_v, K, V] float32
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [3] int32
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
        initial_state: State cache ``[num_states, H_v, K, V]`` float32.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[3]`` — ``(decode_end, prefill_end, mixed_end)``.
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
    bt = get_default_block_sizes(
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

    scale = d_k**-0.5

    output, new_recurrent_state = fused_decoding_gdn(
        q,
        k,
        v,
        g,
        recurrent_state,
        state_indices,
        distribution=distribution,
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


# --- from recurrent_scan_v2.py ------------------------------------
# pylint: disable=invalid-name





def invert_triangular_matrix_scan(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


def inner_kernel(
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Prefill chunk
    prefill_qkv_ref,
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Decode batch
    decode_qkv_ref,
    # VMEM: (C, 128). Raw a values for Prefill chunk
    prefill_a_raw_ref,
    # VMEM: (BT, 128). Raw a values for Decode batch
    decode_a_raw_ref,
    # VMEM: (C, 128). Raw b values for Prefill chunk
    prefill_b_raw_ref,
    # VMEM: (BT, 128). Raw b values for Decode batch
    decode_b_raw_ref,
    # VMEM: (n_v,). A_log for gate computation
    a_log_ref,
    # VMEM: (n_v,). dt_bias for gate computation
    dt_bias_ref,
    # VMEM: (C, n_v * d_v). Scanned outputs for prefill
    prefill_output_ref,
    # VMEM: (BT, n_v * d_v). Scanned outputs for decode
    decode_output_ref,
    # SMEM: (max_blocks, 8). Schedule table
    schedule_table,
    # SMEM: (max_reqs,). State indices
    state_indices,
    # SMEM: (max_reqs,). Whether each request has prior recurrent state
    has_initial_state,
    *,
    # HBM: (B, n_v, d_k, d_v). All recurrent states
    recurrent_state_in,
    recurrent_state_out,
    # Chunk size for prefill
    C: int,
    # Batch size for decode
    BT: int,
    #  Number of key/query heads
    n_kq: int,
    #  Number of value heads
    n_v: int,
    #  Key dimension
    d_k: int,
    #  Value dimension
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    # VMEM scratchpad: (2, n_v, d_k, d_v). To carry state across chunks
    # (double buffered)
    prefill_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). Per-iter safe-copy of the loaded
    # state. Required as a separate buffer from decode_load_scratch because
    # the prefetch DMA for iter b+2 writes to the same slot concurrently with
    # this iter's compute; this buffer isolates the reads. Stored as bf16;
    # per-head fp32 cast happens in VREG inside the compute loop.
    decode_state_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). Aliased to slot 0 of
    # decode_store_scratch in _run_with_scratch (decode drains its stores
    # before prefill runs, so the slot is free for prefill bf16 staging).
    state_commit_scratch,
    # VMEM scratchpad: (2, n_v, d_k, d_v). Double-buffered staging for
    # fully-async decode loads. iter b lands in slot (b % 2).
    decode_load_scratch,
    # VMEM scratchpad: (2, n_v, d_k, d_v). Double-buffered staging for
    # fully-async decode stores. iter b uses slot (b % 2).
    decode_store_scratch,
    # VMEM scratchpad: (BT, n_v * d_v). To hold decode outputs before DMA
    decode_output_scratch,
    # Array of C semaphores for decode state loads
    decode_read_semaphores,
    # 2 semaphores (one per decode_store_scratch slot) for async decode stores
    decode_write_semaphore,
    # 1 semaphore for prefill DMA (stores only)
    prefill_semaphore,
    # Number of decode tokens (requests) in the batch
    decode_tokens,
):
    """Inner kernel for recurrent scan processing both prefill and decode.

  This function is called for each step in the schedule table and dispatches
  work to either `process_decode` or
  `process_regular_prefill`/`process_transition_prefill`.
  """
    step = pl.program_id(0)

    # Instantiate helper config structures
    model_dims = ModelDims(n_kq=n_kq,
                                               n_v=n_v,
                                               d_k=d_k,
                                               d_v=d_v)
    tiling_cfg = TilingConfig(C=C,
                                                  BT=BT,
                                                  sublanesize=sublanesize)
    config = ScanConfig(
        model=model_dims,
        tiling=tiling_cfg,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        decode_tokens=decode_tokens,
    )

    schedule = ScheduleStep(schedule_table, step)

    shared_refs = SharedRefs(
        a_log=a_log_ref,
        dt_bias=dt_bias_ref,
        recurrent_state_in=recurrent_state_in,
        recurrent_state_out=recurrent_state_out,
    )

    prefill_refs = BranchRefs(
        qkv=prefill_qkv_ref,
        a_raw=prefill_a_raw_ref,
        b_raw=prefill_b_raw_ref,
        output=prefill_output_ref,
    )
    prefill_scratch_refs = PrefillScratchRefs(
        scratch=prefill_scratch,
        semaphore=prefill_semaphore,
    )
    prefill_dma = DMAHelper(
        state_in=recurrent_state_in,
        state_out=recurrent_state_out,
        commit_scratch=state_commit_scratch,
        semaphore=prefill_semaphore,
    )

    prefill_processor = PrefillProcessor(
        config=config,
        schedule=schedule,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        refs=prefill_refs,
        shared=shared_refs,
        scratch=prefill_scratch_refs,
        dma=prefill_dma,
    )

    decode_refs = BranchRefs(
        qkv=decode_qkv_ref,
        a_raw=decode_a_raw_ref,
        b_raw=decode_b_raw_ref,
        output=decode_output_ref,
    )
    decode_scratch = DecodeScratchRefs(
        state=decode_state_scratch,
        load=decode_load_scratch,
        store=decode_store_scratch,
        output=decode_output_scratch,
        read_semaphores=decode_read_semaphores,
        write_semaphore=decode_write_semaphore,
    )
    decode_dma_in = DMAHelper(
        state_in=recurrent_state_in,
        state_out=recurrent_state_out,
        commit_scratch=decode_load_scratch,
        semaphore=decode_read_semaphores,
    )
    decode_processor = DecodeProcessor(
        config=config,
        schedule=schedule,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        refs=decode_refs,
        shared=shared_refs,
        scratch=decode_scratch,
        dma=decode_dma_in,
    )

    # READ table

    prefill_valid = schedule_table[step,
                                   COL_PREFILL_VALID][...]
    decode_valid = schedule_table[step,
                                  COL_DECODE_VALID][...]
    decode_offset = schedule_table[step,
                                   COL_DECODE_OFFSET][...]
    prefill_offset = schedule_table[
        step, COL_PREFILL_OFFSET][...]
    is_transition = schedule_table[step,
                                   COL_IS_TRANSITION][...]

    # 2. Decode Branch
    @pl.when(decode_valid > 0)
    def decode_wrapper():
        decode_processor.process()
        return None

    # Prefill Branch
    # Process prefill if there is valid prefill work in this step
    @pl.when(prefill_valid > 0)
    def process_prefill():
        prefill_processor.process()
        return None

    # For transition block at boundary of decode and prefill we will have
    # overlap:
    # - Decode block BT contains prefill tokens
    # - Sublane size transition prefill block contains some decode tokens in the
    #   sublane
    # So we need to stitch the outputs so they don't overwrite each other in the
    # global index. We exchange decode and prefill outputs so:
    # - Prefill output ref has decode token outputs at decode token indexes in its
    #   out ref
    # - Decode output ref has prefill token outputs at prefill token indexes in
    #   its out ref
    def do_stitch():
        local_start = prefill_offset - decode_offset
        local_split = decode_tokens - prefill_offset

        # Need to hint compiler, or it complains in DMA added by emit pipeline
        safe_local_start = pl.multiple_of(local_start, sublanesize)

        decode_overlap = decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :]
        prefill_arr = prefill_output_ref[pl.ds(0, sublanesize), :]

        # 3. Build sublane size mask
        iota = jax.lax.broadcasted_iota(jnp.int32, (sublanesize, ), 0)
        is_decode_mask = (iota < local_split).astype(jnp.int32)[:, None]

        # 4. Merge
        merged_overlap = jnp.where(is_decode_mask, decode_overlap, prefill_arr)

        decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :] = merged_overlap
        prefill_output_ref[pl.ds(0, sublanesize), :] = merged_overlap

        return None

    is_first_block = pl.program_id(0) == 0
    needs_stitching = (is_transition > 0) & is_first_block & (decode_valid > 0)
    jax.lax.cond(needs_stitching, do_stitch, lambda: None)


def get_qkv_index_map_v2(
    step,
    schedule_table,
    valid_col,
    offset_col,
    alignment=16,
    block_size=64,
    sink_offset=0,
):
    valid = schedule_table[step, valid_col][...]
    offset = schedule_table[step, offset_col][...]
    offset = pl.multiple_of(offset, alignment)

    safe_offset = jnp.where(valid > 0, offset, sink_offset)
    safe_offset = pl.multiple_of(safe_offset, alignment)

    return (pl.ds(safe_offset, block_size), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    alignment=16,
    sink_offset=0,
):
    """Creates block specs for recurrent scan kernel."""

    prefill_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=COL_PREFILL_VALID,
        offset_col=COL_PREFILL_OFFSET,
        alignment=alignment,
        block_size=chunk_size,
        sink_offset=sink_offset,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=COL_DECODE_VALID,
        offset_col=COL_DECODE_OFFSET,
        alignment=alignment,
        block_size=BT,
        sink_offset=sink_offset,
    )

    prefill_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), d),
        index_map=prefill_qkv_index_map,
    )
    decode_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), d),
        index_map=decode_qkv_index_map,
    )

    prefill_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
        index_map=prefill_qkv_index_map,
    )
    decode_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), n_v * d_v),
        index_map=decode_qkv_index_map,
    )

    a_log_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    dt_bias_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    prefill_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )
    prefill_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )

    return [
        prefill_qkv_spec,
        decode_qkv_spec,
        prefill_a_raw_spec,
        decode_a_raw_spec,
        prefill_b_raw_spec,
        decode_b_raw_spec,
        a_log_spec,
        dt_bias_spec,
    ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    has_initial_state_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    schedule_table_ref,
    decode_tokens_ref,
    total_blocks_ref,
    recurrent_state_ref,
    output_ref,
    *,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
):
    """Fused kernel for recurrent scan."""
    decode_tokens = decode_tokens_ref[0]
    total_blocks = total_blocks_ref[0]

    d = mixed_qkv_ref.shape[-1]
    pad_size = max(C, BT)
    sink_offset = mixed_qkv_ref.shape[0] - pad_size

    in_specs, out_specs = create_block_specs(
        schedule_table_ref,
        C,
        BT,
        d,
        n_v,
        d_v,
        alignment=sublanesize,
        sink_offset=sink_offset,
    )

    def _run_with_scratch(
        scratch_ref,
        decode_state_scratch_ref,
        decode_load_scratch_ref,
        decode_store_scratch_ref,
        decode_output_scratch_ref,
        decode_read_sems,
        decode_write_sem,
        prefill_sem,
    ):
        # Alias state_commit_scratch to slot 0 of decode_store_scratch.
        # Decode drains its store DMAs before prefill runs in any step, so
        # the slot is free to use as prefill's bf16 HBM staging.
        state_commit_scratch_ref = decode_store_scratch_ref.at[pl.ds(0, 1)]

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                inner_kernel,
                C=C,
                BT=BT,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                sublanesize=sublanesize,
                prefill_scratch=scratch_ref,
                decode_state_scratch=decode_state_scratch_ref,
                decode_output_scratch=decode_output_scratch_ref,
                state_commit_scratch=state_commit_scratch_ref,
                decode_load_scratch=decode_load_scratch_ref,
                decode_store_scratch=decode_store_scratch_ref,
                decode_read_semaphores=decode_read_sems,
                decode_write_semaphore=decode_write_sem,
                prefill_semaphore=prefill_sem,
                decode_tokens=decode_tokens,
                recurrent_state_in=aliased_recurrent_state_ref,
                recurrent_state_out=recurrent_state_ref,
            ),
            grid=(total_blocks, ),
            in_specs=in_specs,
            out_specs=out_specs,
        )

        pipeline_func(
            mixed_qkv_ref,
            mixed_qkv_ref,
            a_raw_ref,
            a_raw_ref,
            b_raw_ref,
            b_raw_ref,
            a_log_ref,
            dt_bias_ref,
            output_ref,
            output_ref,
            scratches=[
                schedule_table_ref,
                state_indices_ref,
                has_initial_state_ref,
            ],
        )

    pl.run_scoped(
        # TODO: Move this to outer pallas call and get rid of run_scoped
        _run_with_scratch,
        pltpu.VMEM((2, n_v, d_k, d_v),
                   jnp.float32),  # prefill_scratch (double buffered)
        pltpu.VMEM((1, n_v, d_k, d_v),
                   recurrent_state_ref.dtype),  # decode_state_scratch
        # state_commit_scratch aliased to slot 0 of decode_store_scratch in
        # _run_with_scratch; no separate allocation.
        pltpu.VMEM((2, n_v, d_k, d_v), recurrent_state_ref.dtype
                   ),  # decode_load_scratch (double-buffered)
        pltpu.VMEM(
            (2, n_v, d_k, d_v), recurrent_state_ref.dtype
        ),  # decode_store_scratch (double-buffered; slot 0 also used as prefill's state_commit staging)
        pltpu.VMEM((BT, n_v * d_v),
                   mixed_qkv_ref.dtype),  # decode_output_scratch
        pltpu.SemaphoreType.DMA(
            (2, )),  # decode_read_semaphores (one per slot)
        pltpu.SemaphoreType.DMA(
            (2, )),  # decode_write_semaphore (one per slot)
        pltpu.SemaphoreType.DMA((2, )),  # prefill_semaphore
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "BT",
        "use_qk_norm_in_gdn",
        "vmem_limit_bytes",
        "race_detect_enable",
    ],
)
def recurrent_scan(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 128,
    BT: int = 128,
    use_qk_norm_in_gdn: bool = True,
    has_initial_state: jax.Array | None = None,
    vmem_limit_bytes: int | None = None,
    race_detect_enable: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused recurrent scan kernel for GDN on TPU v7.

  Args:
    mixed_qkv: jax.Array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
      Packed Query, Key, and Value tokens.
    b: jax.Array of shape [num_tokens, n_v]. Input for beta gate.
    a: jax.Array of shape [num_tokens, n_v]. Input for g gate.
    recurrent_state: jax.Array of shape [max_reqs, n_v, d_k, d_v]. Current
      recurrent states.
    A_log: jax.Array of shape [n_v]. Log of parameter A.
    dt_bias: jax.Array of shape [n_v]. Bias for dt.
    query_start_loc: jax.Array of shape [num_requests + 1]. Start indices of
      each request in mixed_qkv.
    state_indices: jax.Array of shape [num_requests] or larger. Mapping from
      request ID to state index.
    distribution: jax.Array of shape [3]. Contains [decode_tokens,
      total_tokens].
    n_kq: Number of query/key heads.
    n_v: Number of value heads.
    d_k: Dimension of query/key features.
    d_v: Dimension of value features.
    chunk_size: Block size for processing (default 128).
    BT: Block size for decode requests (default 128).
    use_qk_norm_in_gdn: Whether to use QK normalization.
    vmem_limit_bytes: Per-kernel scoped VMEM ceiling passed to Mosaic.
    race_detect_enable: If True, run the kernel under Pallas interpret mode with
      DMA/buffer race detection enabled.

  Returns:
    A tuple containing:
      - Updated recurrent state of shape [max_reqs, n_v, d_k, d_v].
      - The mixed_qkv array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
  """
    if has_initial_state is None:
        has_initial_state = jnp.zeros(state_indices.shape[0], dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    num_tokens = mixed_qkv.shape[0]
    tpu_info = pltpu.get_tpu_info()
    sublanesize = 4 // mixed_qkv.itemsize * tpu_info.num_sublanes

    # Default the scoped VMEM ceiling. This value could be tuned for different state cache numerics and chunk sizes.
    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.8)

    # Pad token dimension so invalid pipeline steps DMA into a safe sink area.
    # Sink offset must be aligned to sublanesize for Mosaic tile compatibility.
    block_size = max(chunk_size, BT)
    sink_offset = ((num_tokens + sublanesize - 1) // sublanesize) * sublanesize
    pad_rows = sink_offset + block_size - num_tokens
    mixed_qkv = jnp.pad(mixed_qkv, ((0, pad_rows), (0, 0)))

    # Pad raw a and b to (num_tokens + pad_rows, 128) for sublanes
    a_padded = jnp.pad(a, ((0, pad_rows), (0, 128 - n_v)))
    b_padded = jnp.pad(b, ((0, pad_rows), (0, 128 - n_v)))

    # decode_tokens: scalar, number of decode tokens.
    # Assuming length 1 per decode request, this is also the number of decode
    # requests.
    decode_tokens = distribution[0]
    schedule_table, total_blocks = (
        compute_schedule_table_v2(
            query_start_loc,
            decode_tokens,
            distribution[2],
            num_tokens,
            chunk_size,
            BT,
            alignment=sublanesize,
        ))

    # sublane,128
    decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
    total_blocks_arr = jnp.expand_dims(total_blocks, 0)

    grid_spec = pl.GridSpec(
        grid=(1, ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
    )

    updated_recurrent_state, output_padded = pl.pallas_call(
        functools.partial(
            fused_kernel,
            C=chunk_size,
            BT=BT,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            sublanesize=sublanesize,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
            jax.ShapeDtypeStruct((sink_offset + block_size, n_v * d_v),
                                 mixed_qkv.dtype),
        ),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},
        interpret=(pltpu.InterpretParams(
            detect_races=True) if race_detect_enable else False),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(
        mixed_qkv,
        recurrent_state,
        state_indices,
        has_initial_state,
        a_padded,
        b_padded,
        A_log,
        dt_bias,
        schedule_table,
        decode_tokens_arr,
        total_blocks_arr,
    )
    return updated_recurrent_state, output_padded[:num_tokens]


kernel = recurrent_scan
