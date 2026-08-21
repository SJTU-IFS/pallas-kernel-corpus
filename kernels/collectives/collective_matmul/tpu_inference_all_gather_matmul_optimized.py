# SPDX-License-Identifier: Apache-2.0
"""Standalone vLLM tpu-inference all gather matmul.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  paths:
    tpu_inference/kernels/collectives/util.py
    tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py
    tpu_inference/kernels/collectives/all_gather_matmul.py
  transformation: the modules above were concatenated in dependency order and
    the repo-local imports between them removed. The `util`/`all_gather_matmul_tuned_block_sizes` module
    qualifiers were dropped with a scope-aware pass, so a local name that
    happens to match is left alone. No kernel body
    was touched.

Entry point: ``all_gather_matmul`` (also exported as ``kernel``); the audited Pallas
launch is ``_all_gather_matmul_call``.

Contract ``all_gather_matmul``.
An all-gather fused into a matmul. Rather than gathering every
shard and then multiplying, each device multiplies the shard it
holds while the next one is still in flight -- the communication
hides under the MXU work instead of preceding it.

**NOT VALIDATED, and not for want of trying.** This kernel requires
exactly 8 devices -- upstream's own test skips below that. The
kernel indexes a ring with `lax.axis_index` and exchanges shards
with its left and right neighbours over 14 send/recv semaphores.
This corpus was built and validated on a **v6e-1** -- a single chip, a single
device -- where that ring has no neighbours and the topology has one node. So
the file is carried here with its provenance intact and recorded UNVALIDATED
with zero migrated launch points. It is a hardware requirement, not unfinished
work: no amount of TPU time on a one-device machine would close it.

Native shape: none declared.
"""

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    # `path` is the entry-point module, matching every other file in the
    # corpus; `also_inlines` lists what was concatenated ahead of it.
    "path": "tpu_inference/kernels/collectives/all_gather_matmul.py",
    "also_inlines": ['tpu_inference/kernels/collectives/util.py', 'tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py'],
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "all_gather_matmul",
    "family": "collective_matmul",
    "launch_points": 1,
    "native_shape": None,
    "requires_devices": 8,
    "validated": False,
}

# ---- flattened from tpu_inference/kernels/collectives/util.py ----

import functools

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
    """Performs a barrier with neighbors on the global barrier semaphore.

  Optionally performs a second barrier, which prevents a potential race
  when reusing the same collective_id across kernel invocations.

  Args:
    left_neighbor: Left neighbor device id.
    right_neighbor: Right neighbor device id.
    double_barrier: Whether to perform a second barrier.
  """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped,
                           second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in [left_neighbor, right_neighbor]:
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor, ),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, 2)

# ---- flattened from tpu_inference/kernels/collectives/all_gather_matmul_tuned_block_sizes.py ----

import re

import jax

# key:
#    - tpu_version
#    - m
#    - n
#    - k
#    - dtype
#    - tp_size
# value:
#    - bn
#    - bk
TUNED_BLOCK_SIZES = {
    # go/keep-sorted start
    (6, 1024, 51200, 5120, 'bfloat16', 8): (6400, 2560),
    (6, 1024, 57344, 8192, 'bfloat16', 8): (7168, 8192),
    (6, 2048, 51200, 5120, 'bfloat16', 8): (1280, 5120),
    (6, 2048, 57344, 8192, 'bfloat16', 8): (1024, 8192),
    (6, 4096, 51200, 5120, 'bfloat16', 8): (3200, 5120),
    (6, 8192, 51200, 5120, 'bfloat16', 8): (1280, 5120),
    # go/keep-sorted end
}


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        return -1
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]

    # v6: "TPU v6"
    # v7: "TPU7x"
    assert kind[:3] == 'TPU', kind
    return int(re.search(r'\d+', kind).group())


def get_key(
    m,
    n,
    k,
    dtype,
    tp_size,
):
    """Returns the key for the given parameters."""
    return (
        get_tpu_version(),
        m,
        n,
        k,
        dtype,
        tp_size,
    )


def get_tuned_block_sizes(m, n, k, dtype_name, tp_size):
    """Returns the tuned block sizes for the given parameters."""
    key = get_key(m, n, k, dtype_name, tp_size)
    return TUNED_BLOCK_SIZES.get(key, (None, None))

# ---- flattened from tpu_inference/kernels/collectives/all_gather_matmul.py ----

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


P = jax.sharding.PartitionSpec


def _cdiv(x, y):
    return (x + y - 1) // y


# TODO(chengjiyao): try unrolling the loop instead of using pallas_call grid
# TODO(chengjiyao): try m tiling
# TODO(chengjiyao): try using [bm, bk] and [bk, bn] scratches memory shape for
# large bm
# TODO(chengjiyao): try splitting to two parts when n_per_device is large:
# output_0, gatherd_x = ag-matmul(x, y_0)
# output_1 = matmul(gatherd_x, y_1)
# output = concat(output_0, output_1)
# TODO(chengjiyao): investigate the register spilling
def _all_gather_kernel(
    # Inputs
    x_hbm_ref,  # [m_per_device, k]
    y_hbm_ref,  # [k, n_per_device]
    # Outputs
    o_hbm_ref,  # [m, n_per_device]
    x_hbm_scratch_ref,  # [num_devices - 1, m_per_device, k]
    # Scratches
    x_local_copy_sem,  # []
    y_local_copy_sem,  # []
    o_local_copy_sem,  # []
    send_sems,  # [2, num_devices - 1] for left and right
    recv_sems,  # [2, num_devices - 1] for left and right
    x_vmem_scratch_ref,  # [2, m_per_device, k]
    y_vmem_scratch_ref,  # [k, n_per_device]
    o_vmem_scratch_ref,  # [2, m_per_device, bn]
    acc_vmem_scratch_ref,  # [m_per_device, bn] of jnp.float32
    axis_name: str,
    bn: int,
    bk: int,
    debug_mode=False,
    rhs_transpose: bool = False,
):
    """Pallas kernel for all-gather.

  Args:
    x_hbm_ref: LHS of the matmul before all-gather.
    y_hbm_ref: RHS of the matmul.
    o_hbm_ref: Output of the matmul.
    x_hbm_scratch_ref: Scratch memory for LHS of the matmul.
    x_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    y_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    o_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
    send_sem: DMA semaphore for the remote send.
    capacity_sem: Capacity semaphore for the remote send.
    recv_sems: DMA semaphore for the remote receive.
    x_vmem_scratch_ref: Scratch memory for LHS of the matmul.
    y_vmem_scratch_ref: Scratch memory for RHS of the matmul.
    o_vmem_scratch_ref: Scratch memory for output of the matmul.
  """
    num_devices = pl.num_programs(0) - 2
    grid_n = pl.num_programs(1)
    grid_k = pl.num_programs(2)
    outer_step = pl.program_id(0)
    bn_i = pl.program_id(1)
    bk_i = pl.program_id(2)
    global_step_id = outer_step * grid_n * grid_k + bn_i * grid_k + bk_i
    mxu_total_steps = num_devices * grid_n * grid_k
    gn_by_gk = grid_n * grid_k
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
    x_hbm_receiving_slot = outer_step
    x_hbm_working_slot = outer_step - 1
    x_vmem_receiving_slot = outer_step % 2
    x_vmem_working_slot = (global_step_id - 1) // gn_by_gk % 2
    o_receiving_slot = lax.rem((global_step_id + grid_k - 1) // grid_k, 2)
    o_working_slot = 1 - o_receiving_slot
    m_per_device, _ = x_hbm_ref.shape
    m_per_device_per_direction = m_per_device // 2

    def debug_print(msg, *args):
        if debug_mode:

            @pl.when(my_id == 0)
            def _debug_print():
                pl.debug_print(msg, *args)

    def _start_or_wait_copy(
        op: jax._src.pallas.mosaic.primitives.AsyncCopyDescriptor,
        wait: bool = False,
    ):
        if wait:
            op.wait()
        else:
            op.start()

    def _do_first_x_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first x local copy, x_vmem_receiving_slot={},"
            " bk_i={}",
            int(wait),
            x_vmem_receiving_slot,
            bk_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[:, k_slice],
            dst_ref=x_vmem_scratch_ref.at[x_vmem_receiving_slot, :, k_slice],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_left_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent x left local copy,"
            " x_hbm_working_slot={}, x_vmem_receiving_slot={}, bk_i={}",
            int(wait),
            x_hbm_working_slot,
            x_vmem_receiving_slot,
            bk_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot,
                :m_per_device_per_direction,
                k_slice,
            ],
            dst_ref=x_vmem_scratch_ref.at[
                x_vmem_receiving_slot,
                :m_per_device_per_direction,
                k_slice,
            ],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_right_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent x right local copy,"
            " x_hbm_working_slot={}, x_vmem_receiving_slot={}, bk_i={}",
            int(wait),
            x_hbm_working_slot,
            x_vmem_receiving_slot,
            bk_i,
        )
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            dst_ref=x_vmem_scratch_ref.at[
                x_vmem_receiving_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_y_local_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do y local copy, bk_i={}, bn_i={}",
            int(wait),
            bk_i,
            bn_i,
        )
        k_slice = pl.ds(bk_i * bk, bk)
        n_slice = pl.ds(bn_i * bn, bn)
        if rhs_transpose:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[n_slice, k_slice],
                dst_ref=y_vmem_scratch_ref.at[n_slice, k_slice],
                sem=y_local_copy_sem,
            )
        else:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[k_slice, n_slice],
                dst_ref=y_vmem_scratch_ref.at[k_slice, n_slice],
                sem=y_local_copy_sem,
            )
        _start_or_wait_copy(y_local_copy_op, wait)

    def _do_first_left_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first left remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot,
                                         0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_first_right_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do first right remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[
                x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_subsequent_left_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent left remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot,
                                         0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot,
                                         0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_subsequent_right_remote_copy(wait: bool = False):
        debug_print(
            "[AGMM debug, wait={}] do subsequent right remote copy,"
            " x_hbm_receiving_slot={}, x_hbm_working_slot={}",
            int(wait),
            x_hbm_receiving_slot,
            x_hbm_working_slot,
        )
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot, m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[
                x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor, ),
            device_id_type=pl.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_mxu():
        working_global_step_id = global_step_id - 1
        working_bk_i = working_global_step_id % grid_k
        working_bn_i = working_global_step_id % gn_by_gk // grid_k
        debug_print(
            "[AGMM debug] do mxu, x_vmem_working_slot={}, o_receiving_slot={},"
            " working_bk_i={}, working_bn_i={}",
            x_vmem_working_slot,
            o_receiving_slot,
            working_bk_i,
            working_bn_i,
        )
        k_slice = pl.ds(working_bk_i * bk, bk)
        n_slice = pl.ds(working_bn_i * bn, bn)

        if grid_k == 1:
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot][...]
                rhs = y_vmem_scratch_ref.at[n_slice, :][...]
                o_vmem_scratch_ref.at[o_receiving_slot][...] = lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
            else:
                o_vmem_scratch_ref.at[o_receiving_slot][...] = jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot][...],
                    y_vmem_scratch_ref.at[:, n_slice][...],
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
        else:
            # TODO(chengjiyao): optimize the vstore
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot, :,
                                            k_slice][...]
                rhs = y_vmem_scratch_ref.at[n_slice, k_slice][...]
                acc_vmem_scratch_ref[...] += lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1, ), (1, )), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            else:
                acc_vmem_scratch_ref[...] += jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot, :,
                                          k_slice][...],
                    y_vmem_scratch_ref.at[k_slice, n_slice][...],
                    preferred_element_type=jnp.float32,
                )

            @pl.when(working_bk_i == grid_k - 1)
            def _update():
                debug_print(
                    "[AGMM debug] update, o_receiving_slot={}",
                    o_receiving_slot,
                )
                o_vmem_scratch_ref.at[o_receiving_slot][
                    ...] = acc_vmem_scratch_ref[...].astype(
                        x_vmem_scratch_ref.dtype)
                # TODO(chengjiyao): based on the kyuyeunk' suggestion:
                # this logic can be more optimized. right now it does this.
                # line 316 performs dot
                # line 316 loads from acc_vmem_scartch_ref
                # line 316 adds resulting dot with acc_vmem_scratch_ref
                # line 316 stores result into acc_vmem_scratch_ref
                # line 335 loads from acc_vmem_scratch_ref again.
                # line 338 zero initializes & stores it to acc_vmem_scratch_ref
                # better way would be

                # perform dot
                # if working_bk_i != 0, load from acc_vmem_scratch_ref and add result
                # from previous step. If not, skip this process.
                # if working_bk_i == gk - 1, store the result from step 2 into
                # o_vmem_scratch_ref, if not, store it into acc_vmem_scratch_ref
                acc_vmem_scratch_ref[...] = jnp.zeros_like(
                    acc_vmem_scratch_ref)

    def _do_o_local_copy(wait: bool = False):
        working_global_step_id = global_step_id - grid_k - 1
        working_bn_i = (working_global_step_id % gn_by_gk) // grid_k
        n_slice = pl.ds(working_bn_i * bn, bn)
        offset = (global_step_id - 2) // gn_by_gk
        left_o_idx = (my_id + offset) % num_devices
        left_o_idx = left_o_idx * 2
        right_o_idx = (my_id - offset + num_devices) % num_devices
        right_o_idx = right_o_idx * 2 + 1
        debug_print(
            "[AGMM debug, wait={}] do o local copy, o_working_slot={},"
            " left_o_idx={}, right_o_idx={}, working_bn_i={}",
            int(wait),
            o_working_slot,
            left_o_idx,
            right_o_idx,
            working_bn_i,
        )
        o_left_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[
                o_working_slot, :m_per_device_per_direction],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * left_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        o_right_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[o_working_slot,
                                          m_per_device_per_direction:],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * right_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        _start_or_wait_copy(o_left_local_copy_op, wait)
        _start_or_wait_copy(o_right_local_copy_op, wait)

    ### ------- Kernel start ------- ###
    # TODO(chengjiyao): explore a fine-grained way to do the waits and signal

    debug_print(
        "===== starting a grid, outer_step={}, bn_i={}, bk_i={} =====",
        outer_step,
        bn_i,
        bk_i,
    )

    @pl.when(global_step_id == 0)
    @jax.named_scope("_start_first_remote_copy")
    def _start_first_remote_copy():
        if grid_k > 1:
            acc_vmem_scratch_ref[...] = jnp.zeros_like(acc_vmem_scratch_ref)
        # Barrier with both neighbors at the start, since we will be
        # communicating with both.
        local_barrier(left_neighbor, right_neighbor)
        _do_first_left_remote_copy(wait=False)
        _do_first_right_remote_copy(wait=False)

    cond_start_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % gn_by_gk == 0,
    )

    @pl.when(cond_start_subsequent_remote_copy)
    @jax.named_scope("_start_subsequent_remote_copy")
    def _start_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=False)
        _do_subsequent_right_remote_copy(wait=False)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_start_first_local_x_copy")
    def _start_first_x_local_copy():
        _do_first_x_local_copy(wait=False)

    cond_subsequent_x_local_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices), bn_i == 0)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_start_subsequent_x_local_copy")
    def _start_subsequent_x_local_copy():
        _do_subsequent_x_left_local_copy(wait=False)
        _do_subsequent_x_right_local_copy(wait=False)

    @pl.when(outer_step == 0)
    @jax.named_scope("_start_y_local_copy")
    def _start_y_local_copy():
        _do_y_local_copy(wait=False)

    def _get_start_o_local_copy_cond():
        if grid_k == 1:
            return jnp.logical_and(global_step_id >= 2, global_step_id
                                   < mxu_total_steps + 2)
        else:
            return jnp.logical_and(
                jnp.logical_and(
                    global_step_id >= grid_k + 1,
                    global_step_id < mxu_total_steps + grid_k + 1,
                ),
                global_step_id % grid_k == 1,
            )

    @pl.when(_get_start_o_local_copy_cond())
    @jax.named_scope("_start_o_local_copy")
    def _start_o_local_copy():
        _do_o_local_copy(wait=False)

    @pl.when(
        jnp.logical_and(global_step_id >= 1, global_step_id
                        < 1 + mxu_total_steps))
    @jax.named_scope("_mxu")
    def _mxu():
        _do_mxu()

    def _get_wait_o_local_copy_cond():
        if grid_k == 1:
            return jnp.logical_and(global_step_id >= 2, global_step_id
                                   < mxu_total_steps + 2)
        else:
            return jnp.logical_and(
                jnp.logical_and(
                    global_step_id >= grid_k + 1,
                    global_step_id < mxu_total_steps + grid_k + 1,
                ),
                global_step_id % grid_k == 0,
            )

    @pl.when(_get_wait_o_local_copy_cond())
    @jax.named_scope("_wait_o_local_copy")
    def _wait_o_local_copy():
        _do_o_local_copy(wait=True)

    @pl.when(outer_step == 0)
    @jax.named_scope("_wait_y_local_copy")
    def _wait_y_local_copy():
        _do_y_local_copy(wait=True)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_wait_first_x_local_copy")
    def _wait_first_x_local_copy():
        _do_first_x_local_copy(wait=True)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_wait_subsequent_x_local_copy")
    def _wait_subsequent_x_local_copy():
        _do_subsequent_x_left_local_copy(wait=True)
        _do_subsequent_x_right_local_copy(wait=True)

    @pl.when(global_step_id == gn_by_gk - 1)
    @jax.named_scope("_wait_first_remote_copy")
    def _wait_first_remote_copy():
        _do_first_left_remote_copy(wait=True)
        _do_first_right_remote_copy(wait=True)

    cond_wait_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % gn_by_gk == gn_by_gk - 1,
    )

    @pl.when(cond_wait_subsequent_remote_copy)
    @jax.named_scope("_wait_subsequent_remote_copy")
    def _wait_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=True)
        _do_subsequent_right_remote_copy(wait=True)

    ### ------- Kernel end ------- ###


# FIXME(chengjiyao): make it accurate for the cases of quantization
def get_vmem_estimate_bytes(
    m,
    n,
    k,
    bn,
    acc_bytes,
    tp_size,
    x_dtype,
    y_dtype,
    out_dtype,
):
    """Returns the total vmem bytes used by the kernel."""
    m_per_device = m // tp_size
    n_per_device = n // tp_size
    y_vmem_bytes = (n_per_device * k * dtypes.itemsize_bits(y_dtype) // 8)
    total_bytes = (
        2 * m_per_device * k * dtypes.itemsize_bits(x_dtype) // 8
        # x_vmem_scratch_ref
        + y_vmem_bytes  # y_vmem_scratch_ref
        + 2 * m * bn * dtypes.itemsize_bits(out_dtype) // 8
        # o_vmem_scratch_ref
        + acc_bytes  # acc_vmem_scratch_ref, jnp.float32
    )
    return total_bytes


def validate_inputs(x, y, tp_size, rhs_transpose=False):
    """Validates the inputs to the all_gather_matmul kernel."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
    if x.dtype != y.dtype:
        raise ValueError(
            f"Input dtypes must match, got {x.dtype} and {y.dtype}.")
    m, k = x.shape
    if rhs_transpose:
        n, k_from_y = y.shape
    else:
        k_from_y, n = y.shape
    if k != k_from_y:
        raise ValueError(
            "Incompatible shapes for matmul: contracting dimension mismatch:"
            f" {x.shape} and {y.shape}.")

    if k % 128 != 0:
        raise ValueError(f"k ({k}) must be divisible by 128.")

    if n % 128 != 0:
        raise ValueError(f"n ({n}) must be divisible by 128.")

    m_per_device_per_direction = m // tp_size // 2
    if m_per_device_per_direction % 8 != 0:
        raise ValueError(f"m ({m}) must be divisible by {{tp_size * 2 * 8}}.")

    if m % (tp_size * 2) != 0:
        raise ValueError(
            f"x.shape[0] ({m}) must be divisible by tp_size * 2 ({tp_size * 2})'."
        )
    if n % tp_size != 0:
        raise ValueError(
            f"y.shape[{0 if rhs_transpose else 1}] ({n}) must be divisible by"
            f" tp_size ({tp_size}) on axis '{tp_size}'.")


def all_gather_matmul(
    x: jax.Array,
    y: jax.Array,
    mesh: jax.sharding.AbstractMesh,
    axis_name: str,
    collective_id: int | None = 0,
    bn: int | None = None,
    bk: int | None = None,
    rhs_transpose: bool = False,
):
    """Performs all-gather on the input tensor and then a matmul.

  Args:
    x: LHS of the matmul before all-gather.
    y: RHS of the matmul.
    mesh: JAX mesh.
    axis_name: Name of the axis to all-gather over.
    collective_id: An integer used for barrier semaphore allocation.
    bn: Number of blocks in the n dimension.
    bk: Number of blocks in the k dimension.
    rhs_transpose: If True, y is transposed.

  Returns:
    all-gather(x, axis=0) @ y
  """
    tp_size = mesh.shape[axis_name]
    validate_inputs(x, y, tp_size, rhs_transpose)
    m, k = x.shape
    if rhs_transpose:
        n, _ = y.shape
        y_in_spec = P(axis_name, None)
    else:
        _, n = y.shape
        y_in_spec = P(None, axis_name)
    m_per_device = m // tp_size
    n_per_device = n // tp_size
    tuned_bn, tuned_bk = (
        get_tuned_block_sizes(
            m, n, k,
            jnp.dtype(x.dtype).name, tp_size))
    if bn is None:
        bn = tuned_bn if tuned_bn is not None else n
    if bk is None:
        bk = tuned_bk if tuned_bk is not None else k
    grid_n = _cdiv(n_per_device, bn)
    grid_k = _cdiv(k, bk)
    acc_shape = (m_per_device, bn)
    # NOTE(chengjiyao): acc buffer is not used in the grid_k == 1 case.
    if grid_k == 1:
        acc_shape = (8, 128)
    acc_bytes = (acc_shape[0] * acc_shape[1] *
                 dtypes.itemsize_bits(jnp.float32)) // 8
    y_vmem_shape = (n_per_device, k) if rhs_transpose else (k, n_per_device)
    estimated_vmem_bytes = get_vmem_estimate_bytes(
        m,
        n,
        k,
        bn,
        acc_bytes,
        tp_size,
        x.dtype,
        y.dtype,
        x.dtype,
    )
    out_shape = [
        jax.ShapeDtypeStruct((m, n_per_device), x.dtype),  # output
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, k),
                             x.dtype),  # x HBM scratch
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,  # x_local_copy_sem
            pltpu.SemaphoreType.DMA,  # y_local_copy_sem
            pltpu.SemaphoreType.DMA,  # o_local_copy_sem
            pltpu.SemaphoreType.DMA(
                (2, tp_size - 1)),  # left and right send semaphores
            pltpu.SemaphoreType.DMA((
                2,
                tp_size - 1,
            )),  # left and right recv semaphores
            pltpu.VMEM((2, m_per_device, k), x.dtype),  # x vmem scratch
            pltpu.VMEM(y_vmem_shape, y.dtype),  # y vmem scratch
            pltpu.VMEM((2, m_per_device, bn), x.dtype),  # output vmem scratch
            pltpu.VMEM(acc_shape, jnp.float32),  # acc vmem scratch
        ),
        grid=(tp_size + 2, grid_n, grid_k),
    )
    flops = 2 * m * k * n_per_device
    bytes_accessed = x.dtype.itemsize * (m * k + k * n_per_device +
                                         m * n_per_device)
    cost_estimate = pl.CostEstimate(flops=flops,
                                    bytes_accessed=bytes_accessed,
                                    transcendentals=0)

    @jax.jit(static_argnames=["bn", "bk", "rhs_transpose"])
    def _all_gather_matmul_call(x, y, bn, bk, rhs_transpose):
        return pl.pallas_call(
            functools.partial(
                _all_gather_kernel,
                bn=bn,
                bk=bk,
                axis_name=axis_name,
                rhs_transpose=rhs_transpose,
            ),
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=estimated_vmem_bytes + 8 * 1024 * 1024,
            ),
            cost_estimate=cost_estimate,
            name=get_kernel_name(bn, bk, rhs_transpose),
        )(x, y)[0]

    shard_map_kernel = jax.jit(
        jax.shard_map(
            functools.partial(
                _all_gather_matmul_call,
                bn=bn,
                bk=bk,
                rhs_transpose=rhs_transpose,
            ),
            mesh=mesh,
            in_specs=(P(axis_name, None), y_in_spec),
            out_specs=P(None, axis_name),
            check_vma=False,
        ), )

    return shard_map_kernel(x, y)


def get_kernel_name(bn: int, bk: int, rhs_transpose: bool):
    return (
        f"all_gather_matmul_kernel_bn_{bn}_bk_{bk}_rhs_transpose_{rhs_transpose}"
    )


kernel = all_gather_matmul
