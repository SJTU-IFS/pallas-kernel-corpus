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
"""Standalone vLLM tpu-inference hierarchical reduce scatter.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  paths:
    tpu_inference/kernels/collectives/hierrs_sc/config.py
    tpu_inference/kernels/collectives/hierrs_sc/topology.py
    tpu_inference/kernels/collectives/hierrs_sc/dma_pipeline.py
    tpu_inference/kernels/collectives/hierrs_sc/kernel.py
    tpu_inference/kernels/collectives/hierrs_sc/wrapper.py
  transformation: the modules above were concatenated in dependency order and
    the repo-local imports between them removed. No kernel body
    was touched.

Entry point: ``hierarchical_reduce_scatter_local`` (also exported as ``kernel``); the audited Pallas
launch is ``hierarchical_reduce_scatter_local``.

Contract ``hierarchical_reduce_scatter``.
Reduce-scatter by recursive halving, on SparseCore rather than
the TensorCore -- so the reduction runs beside the model's own
compute instead of competing with it. Two-stage pipelining
overlaps the intra-die and inter-chip hops with local adds.

**NOT VALIDATED, and not for want of trying.** This kernel requires
a multi-chip topology. The kernel runs on SparseCore and
pipelines Die-to-Die against Chip-to-Chip ICI, with devices
ordered by physical topology coordinates.
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
    "path": "tpu_inference/kernels/collectives/hierrs_sc/wrapper.py",
    "also_inlines": ['tpu_inference/kernels/collectives/hierrs_sc/config.py', 'tpu_inference/kernels/collectives/hierrs_sc/topology.py', 'tpu_inference/kernels/collectives/hierrs_sc/dma_pipeline.py', 'tpu_inference/kernels/collectives/hierrs_sc/kernel.py'],
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "hierarchical_reduce_scatter",
    "family": "collective_matmul",
    "launch_points": 1,
    "native_shape": None,
    "requires_devices": None,
    "validated": False,
}

# ---- flattened from tpu_inference/kernels/collectives/hierrs_sc/config.py ----

import dataclasses
import math

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Config:
    # yapf: disable
    """Dimensions and sharding sizes.

  ================================================================================================
              FIGURE 1: PHASE 1 TENSOR PARTITIONING (PIPELINE MICRO-BATCHES)
  ================================================================================================
              |<--------------------------------- hidden_dim_size -------------------------------->|
              |                                         |                                          |
  Iteration:  |<---------------- mb_size -------------->|<---------------- mb_size --------------->| (x num_micro_batches)
              +-----------------------------------------+------------------------------------------+ ---
  chunk_    ^ |                                         |                                          |  ^
  size      | |         Device 0, Microbatch 0          |          Device 0, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  | num_
  chunk_    ^ |                                         |                                          |  | tokens
  size      | |         Device 1, Microbatch 0          |          Device 1, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  |
              |                  ...                    |                    ...                   |  |
              +-----------------------------------------+------------------------------------------+  v
                                                                                                     ---
  ================================================================================================

  ================================================================================================
            FIGURE 2: MACRO PHASE 2 TENSOR PARTITIONING (ICI CROSS-CHIP SCATTER)
  ================================================================================================
  Zooming into a SINGLE Micro-batch column (`mb_size`) to show how Phase 2 slices the chunks further
  for Reduce-Scatter across the i-th hypercube network dimensions.

              |<-------------------------------- mb_size ---------------------------------------->|
              |                                         |                                         |
  ICI Phase 2:|<------------ hc_chunk_size ------------>|<------------ hc_chunk_size ------------>| (x num_hcube_dims)
              +-----------------------------------------+-----------------------------------------+ ---
  chunk_    ^ |                                         |                                         |  ^
  size      | |    Device 0, RS through 0th axis        |    Device 0, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  | num_
  chunk_    ^ |                                         |                                         |  | tokens
  size      | |    Device 1, RS through 0th axis        |    Device 1, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  |
              |                  ...                    |                    ...                  |  |
              +-----------------------------------------+-----------------------------------------+  v
  ================================================================================================

  ================================================================================================
                  FIGURE 3: CORE AND SUBCORE PARTITIONING
  ================================================================================================
  Zooming into a SINGLE grid cell (one Device's Chunk) to show how it physically maps
  onto the cores and subcores for accumulation.

              |<-------------------------- mb_size or hc_chunk_size -------------------------->|
              |                                                                                |
  Subcore Col:|<-- {p1,p2}_col_cs ->|<-- {p1,p2}_col_cs ->|<-- {p1,p2}_col_cs ->|     ...      |
              +=====================+=====================+=====================+==============+ ---
  subcore_  ^ |                     |                     |                     |              |  ^
  chunk_    | |  Core 0, Subcore 0  |  Core 0, Subcore 0  |  Core 0, Subcore 0  |              |  |
  size      v +---------------------+---------------------+---------------------+--------------+  |
  subcore_  ^ |                     |                     |                     |              |  | core_
  chunk_    | |  Core 0, Subcore 1  |  Core 0, Subcore 1  |  Core 0, Subcore 1  |              |  | chunk_
  size      v +---------------------+---------------------+---------------------+--------------+  | size
              |         ...         |         ...         |         ...         |              |  |
              +=====================+=====================+=====================+==============+  v
  subcore_  ^ |                     |                     |                     |              |  ^
  chunk_    | |  Core 1, Subcore 0  |  Core 1, Subcore 0  |  Core 1, Subcore 0  |              |  |
  size      v +---------------------+---------------------+---------------------+--------------+  |
  subcore_  ^ |                     |                     |                     |              |  | core_
  chunk_    | |  Core 1, Subcore 1  |  Core 1, Subcore 1  |  Core 1, Subcore 1  |              |  | chunk_
  size      v +---------------------+---------------------+---------------------+--------------+  | size
              |         ...         |         ...         |         ...         |              |  |
              +=====================+=====================+=====================+==============+  v
  ================================================================================================
  LEGEND:
  - mb_size            = hidden_dim_size // num_micro_batches
  - hc_chunk_size      = mb_size // num_hcube_dims
  - core_chunk_size    = chunk_size // num_cores
  - subcore_chunk_size = core_chunk_size // num_subcores_row
  - p1_col_cs          = mb_size // num_subcores_col
  - p2_col_cs          = hc_chunk_size // num_subcores_col
  ================================================================================================
  """
    # yapf: enable

    # Total number of devices executing this kernel
    num_devices: int
    # Total hidden size dimension (e.g., 4096)
    hidden_dim_size: int
    # Local sequence slice per device (= num_tokens // num_devices)
    chunk_size: int
    # Number of tokens
    num_tokens: int
    # Input data type (e.g., bfloat16)
    dtype: jnp.dtype
    # Pipelining unrolling factor for overlapping ALU/DMA. If None, determined by
    # heuristic.
    _num_micro_batches: int | None = None

    def __post_init__(self):
        assert self.cores_per_chip == 2, (
            "This kernel architecture strictly supports 2 cores per chip, but"
            f" found {self.cores_per_chip}.")
        assert (self.num_chips & (self.num_chips - 1)
                ) == 0, f"num_chips {self.num_chips} must be a power of 2"
        assert (self.num_hcube_dims
                >= 1), f"num_hcube_dims {self.num_hcube_dims} must be >= 1"
        assert self.hidden_dim_size % self.num_micro_batches == 0, (
            f"hidden_dim_size {self.hidden_dim_size} must be divisible by "
            f"num_micro_batches {self.num_micro_batches}")
        assert self.sc_info is not None, "Cannot find sc_info"

    @property
    def cores_per_chip(self) -> int:
        """Number of physical tensor cores per chip on the current TPU architecture."""
        return pltpu.get_tpu_info(
        ).chip_version.num_physical_tensor_cores_per_chip

    @property
    def num_chips(self) -> int:
        """Number of physical TPU chips in the mesh (num_devices // cores_per_chip)"""
        return self.num_devices // self.cores_per_chip

    @property
    def packing_factor(self) -> int:
        """Returns the number of array elements packed into a single 32-bit (4-byte) word."""
        return 4 // self.dtype.itemsize

    @property
    def num_hcube_dims(self) -> int:
        """ICI hypercube logical network dimensions (log2(num_chips))"""
        return int(math.log2(self.num_chips))

    @property
    def num_micro_batches(self) -> int:
        """Pipelining unrolling factor for overlapping ALU/DMA.

    If not set, use the best value based on the empirical results.
    """
        if self._num_micro_batches is not None:
            return self._num_micro_batches
        if self.num_tokens >= 4096:
            return 8
        elif self.num_tokens >= 2048:
            return 4
        elif self.num_tokens >= 256:
            return 2
        else:
            return 1

    @property
    def mb_size(self) -> int:
        """Micro batch slice size"""
        return self.hidden_dim_size // self.num_micro_batches

    @property
    def hc_chunk_size(self) -> int:
        """Phase 2 (C2C) hypercube chunk slice size"""
        return self.mb_size // self.num_hcube_dims

    @property
    def sc_info(self):
        return pltpu.get_tpu_info().sparse_core

    @property
    def num_cores(self) -> int:
        # Use all cores to maximize aggregate HBM memory bandwidth.
        return self.sc_info.num_cores

    @property
    def num_subcores_col(self) -> int:
        """Number of subcore columns used for column-wise hidden size partitioning"""
        # Shard column as much as possible as long as the chunk's width >= 128, which is for dma alignment.
        return min(self.sc_info.num_lanes, self.hc_chunk_size // 128)

    @property
    def num_subcores_row(self) -> int:
        """Number of subcore rows used for row-wise sequence partitioning"""
        # Rows are sharded both core and remaining subcores.
        return min(
            self.sc_info.num_lanes // self.num_subcores_col,
            self.chunk_size // (self.num_cores * self.packing_factor),
        )

    @property
    def num_subcores(self) -> int:
        return self.num_subcores_row * self.num_subcores_col

    @property
    def core_chunk_size(self) -> int:
        """Sequence slice size assigned to each physical core on a device"""
        return self.chunk_size // self.num_cores

    @property
    def subcore_chunk_size(self) -> int:
        """Sequence slice size assigned to each subcore row"""
        return self.core_chunk_size // self.num_subcores_row

    @property
    def subcore_col_chunk_size_p1(self) -> int:
        """Column slice size for Phase 1 DMA"""
        return self.mb_size // self.num_subcores_col

    @property
    def subcore_col_chunk_size_p2(self) -> int:
        """Column slice size for Phase 2 DMA"""
        return self.hc_chunk_size // self.num_subcores_col

# ---- flattened from tpu_inference/kernels/collectives/hierrs_sc/topology.py ----

import jax
from jax.experimental import pallas as pl




class Topology:
    """Abstracts device indexing logic to find neighbors/partners."""

    def __init__(self, axis_name: str):
        self.cur_id = jax.lax.axis_index(axis_name)
        self.cur_chip_id = self.cur_id // 2
        self.cur_chiplet_bit = self.cur_id % 2
        self.partner_id = jax.lax.select(self.cur_chiplet_bit == 0,
                                         self.cur_id + 1, self.cur_id - 1)

    def get_device_id(self, chip_id, chiplet_bit):
        """Returns the global device ID from physical chip `chip_id` and chiplet coordinate `chiplet_bit` (0 or 1)."""
        return chip_id * 2 + chiplet_bit

    def get_neighbor_chip_id(self, dim):
        """Returns the physical chip ID of the logical neighbor in hypercube dimension `dim`.

    For example, on a 2D hypercube of 4 physical chips (IDs: 0, 1, 2, 3):
    - If current chip is 0 (binary 00):
      - Neighbor along dimension 0 is: 0 ^ (1 << 0) = 1 (binary 01).
      - Neighbor along dimension 1 is: 0 ^ (1 << 1) = 2 (binary 10).
    """
        return self.cur_chip_id ^ (1 << dim)

    def get_neighbor_device_id(self, dim):
        """Returns the ID of the neighbor device along hypercube dimension `dim` sharing the same chiplet position.

    For example, on a 2D hypercube of 4 chips (IDs 0-3) containing 8 logical
    devices (IDs 0-7):
    - If current device is 0 (physical chip 0, chiplet bit 0):
      - Neighbor along dimension 0 is: get_device_id(neighbor_chip=1, chiplet=0)
      = 2.
      - Neighbor along dimension 1 is: get_device_id(neighbor_chip=2, chiplet=0)
      = 4.
    """
        return self.get_device_id(self.get_neighbor_chip_id(dim),
                                  self.cur_chiplet_bit)


class ChunkLocator:
    """Encapsulates sequence and HBM indexing math for SparseCore Reduce-Scatter."""

    def __init__(
            self,
            config: Config,
            topo: Topology,
            core_idx: jax.Array,  # integer scalar.
            subcore_idx: jax.Array | None,  # integer scalar. None for SCS.
    ):
        self.config = config
        self.topo = topo
        self.core_idx = core_idx
        if subcore_idx is not None:
            self.subcore_row_idx = subcore_idx // config.num_subcores_col
            self.subcore_col_idx = subcore_idx % config.num_subcores_col
        else:
            self.subcore_row_idx = None
            self.subcore_col_idx = None
        self.mb_stride = config.num_hcube_dims * config.hc_chunk_size

    def _get_row_slice(self, chunk_idx, for_tec):
        """Returns a row slice for `chunk_idx` of core-level size,

    or subcore-level if `for_tec` is True.
    """
        row_offset = (chunk_idx * self.config.chunk_size +
                      self.core_idx * self.config.core_chunk_size)
        if for_tec:
            row_offset += self.subcore_row_idx * self.config.subcore_chunk_size
            row_size = self.config.subcore_chunk_size
        else:
            row_size = self.config.core_chunk_size
        return pl.ds(pl.multiple_of(row_offset, 8), row_size)

    def _get_col_slice(self, base_col_offset, col_size, col_chunk_size,
                       for_tec):
        """Returns a column slice from `base_col_offset` of width `col_size`,

    or sharded to `col_chunk_size` if `for_tec` is True.
    """
        if for_tec:
            col_offset = base_col_offset + self.subcore_col_idx * col_chunk_size
            col_width = col_chunk_size
        else:
            col_offset = base_col_offset
            col_width = col_size
        return pl.ds(col_offset, col_width)

    def get_phase1_slice(self, chunk_idx, mb_idx, *, for_tec=False):
        """Returns a 2D HBM slice for Phase 1 (D2D) for `chunk_idx` and `mb_idx`,

    mapped to subcore if `for_tec` is True.
    """
        return (
            self._get_row_slice(chunk_idx, for_tec),
            self._get_col_slice(
                mb_idx * self.config.mb_size,
                self.config.mb_size,
                self.config.subcore_col_chunk_size_p1,
                for_tec,
            ),
        )

    def get_phase2_slice(self,
                         chunk_idx,
                         mb_idx,
                         hcube_dim_idx,
                         *,
                         for_tec=False):
        """Returns a 2D HBM slice for Phase 2 (C2C) for `chunk_idx`, `mb_idx`

    and `hcube_dim_idx`, mapped to subcore if `for_tec` is True.
    """
        return (
            self._get_row_slice(chunk_idx, for_tec),
            self._get_col_slice(
                mb_idx * self.config.mb_size +
                hcube_dim_idx * self.config.hc_chunk_size,
                self.config.hc_chunk_size,
                self.config.subcore_col_chunk_size_p2,
                for_tec,
            ),
        )

    def get_phase1_chunk_idx(self, device_id, chip_idx):
        """Calculates the chunk index processed by `device_id` for `chip_idx`.

    In Phase 1, global token chunks are sharded across the topology. A device
    processes token chunks corresponding to all physical chips `chip_idx` in
    the mesh, filtered by its own chiplet position (even/odd device ID).
    """
        chiplet_bit = device_id % 2
        return chip_idx * 2 + chiplet_bit

    def get_phase1_chunk_idxes(self, device_id):
        """Returns all global chunk indices processed by the chiplet group of device `device_id`."""
        chiplet_bit = device_id % 2
        return [
            chip_idx * 2 + chiplet_bit
            for chip_idx in range(self.config.num_chips)
        ]

    def get_phase2_chunk_idx(self, device_id, step_idx, chunk_group_idx,
                             hcube_dim_idx):
        """Calculates the chunk index owned by a device `device_id` for chunk group `chunk_group_idx` during Phase 2 (C2C RS).

    During Phase 2, devices perform a hypercube reduction. At step `step_idx` of
    the hypercube reduction, the topology is partitioned into independent
    parallel sub-cubes/groups of devices exchanging along hypercube dimension
    `hcube_dim_idx`.
    """
        dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
        chip_id = device_id // 2
        my_dim_bit = (chip_id >> dim) & 1

        prev_dims = [(hcube_dim_idx + j) % self.config.num_hcube_dims
                     for j in range(step_idx)]
        future_dims = [(hcube_dim_idx + j) % self.config.num_hcube_dims
                       for j in range(step_idx + 1, self.config.num_hcube_dims)
                       ]

        my_base_chunk_idx = self.get_hcube_chunk_idx(device_id,
                                                     chunk_group_idx,
                                                     future_dims, prev_dims,
                                                     dim, my_dim_bit)
        chiplet_bit = device_id % 2
        return my_base_chunk_idx * 2 + chiplet_bit

    def get_hcube_chunk_idx(
        self,
        device_id,
        chunk_group_idx,
        future_dims,
        prev_dims,
        target_dim,
        dim_val,
    ):
        """Calculates the mapped HBM chunk index for the hypercube communication ring of device `device_id` at iteration `chunk_group_idx` along active dimension `target_dim` with bit value `dim_val`, given the processed dimensions `prev_dims` and unprocessed dimensions `future_dims`."""
        chip_id = device_id // 2
        base = 0
        for d in prev_dims:
            bit = (chip_id >> d) & 1
            base |= bit << d
        for bit_pos, d in enumerate(future_dims):
            bit = (chunk_group_idx >> bit_pos) & 1
            base |= bit << d
        base |= dim_val << target_dim
        return base

# ---- flattened from tpu_inference/kernels/collectives/hierrs_sc/dma_pipeline.py ----

import dataclasses
import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc



def _accumulate(
    *,
    config: Config,
    target,
    addend,
    num_rows,
    num_cols,
    col_step=16,
):
    """Performs in-place element-wise addition: `target += addend`.

  Both `target` and `addend` are 2D references of shape (`num_rows`,
  `num_cols`).
  """
    packing = config.packing_factor

    @plsc.parallel_loop(0, num_cols, step=col_step)
    def _loop(c_in):
        c_slice = pl.ds(c_in, col_step)

        num_iters = num_rows // packing
        for i in range(num_iters):
            r = i * packing
            r_idx = r if packing == 1 else pl.ds(r, packing)
            val = target[r_idx, c_slice] + addend[r_idx, c_slice]
            target[r_idx, c_slice] = val


@dataclasses.dataclass(frozen=True)
class RemoteDmaManager:
    """Handles remote DMA (ICI) copies on SCS."""

    config: Config
    topo: Topology
    core_idx: jax.Array
    p1_send_sem: jax.Ref = dataclasses.field(kw_only=True)
    p2_send_sem: jax.Ref = dataclasses.field(kw_only=True)
    p1_recv_sem: jax.Ref = dataclasses.field(kw_only=True)
    p2_recv_sem: jax.Ref = dataclasses.field(kw_only=True)
    locator: ChunkLocator = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "locator",
            ChunkLocator(self.config,
                         self.topo,
                         self.core_idx,
                         subcore_idx=None),
        )

    @jax.named_scope("start_phase1_d2d_copies")
    def start_phase1_d2d_copies(
        self,
        *,
        mb_idx,
        src,
        dst,
    ):
        """Triggers remote D2D copies from `src` to `dst` for the micro-batch index `mb_idx`."""
        recv_slot = mb_idx % 2
        for c, send_chunk_idx in enumerate(
                self.locator.get_phase1_chunk_idxes(self.topo.partner_id)):
            row_slice, col_slice = self.locator.get_phase1_slice(
                send_chunk_idx, mb_idx)
            slice_src = src.at[row_slice, col_slice]
            slice_dst = dst.at[row_slice, col_slice]
            pltpu.async_remote_copy(
                slice_src,
                slice_dst,
                self.p1_send_sem.at[recv_slot, c],
                self.p1_recv_sem.at[recv_slot, c],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

    @jax.named_scope("start_phase2_c2c_copies")
    def start_phase2_c2c_copies(
        self,
        *,
        mb_idx,
        step_idx,
        src,
        dst,
    ):
        """Triggers remote C2C copies from `src` to `dst` for `mb_idx` and `step_idx`."""
        slot_idx = mb_idx % 2
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)

        @pl.loop(0, num_chunk_groups)
        def _(chunk_group_idx):

            @pl.loop(0, num_hcube_dims)
            def _(hcube_dim_idx):
                dim = (hcube_dim_idx + step_idx) % num_hcube_dims
                neighbor_device_id = self.topo.get_neighbor_device_id(dim)
                neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neighbor_device_id,
                    step_idx,
                    chunk_group_idx=chunk_group_idx,
                    hcube_dim_idx=hcube_dim_idx,
                )

                row_slice, col_slice = self.locator.get_phase2_slice(
                    neighbor_chunk_idx, mb_idx, hcube_dim_idx)
                slice_src = src.at[step_idx, row_slice, col_slice]
                slice_dst = dst.at[row_slice, col_slice]
                pltpu.async_remote_copy(
                    slice_src,
                    slice_dst,
                    self.p2_send_sem.at[slot_idx, step_idx, chunk_group_idx,
                                        hcube_dim_idx],
                    self.p2_recv_sem.at[slot_idx, step_idx, chunk_group_idx,
                                        hcube_dim_idx],
                    device_id=neighbor_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

    @jax.named_scope("wait_phase1_d2d_copies")
    def wait_phase1_d2d_copies(self,
                               mb_idx,
                               src,
                               dst,
                               *,
                               wait_send: bool = False):
        """Waits for Phase 1 (D2D) remote copies from `src` to `dst` for the index `mb_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
        recv_slot = mb_idx % 2
        p1_recv_sem_slice = self.p1_recv_sem.at[recv_slot]
        p1_send_sem_slice = self.p1_send_sem.at[recv_slot]

        dummy_send = src.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.mb_size),
        ]
        dummy_recv = dst.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.mb_size),
        ]
        my_id = self.topo.cur_id
        for c, _ in enumerate(self.locator.get_phase1_chunk_idxes(my_id)):
            dma = pltpu.make_async_remote_copy(
                src_ref=dummy_send,
                dst_ref=dummy_recv,
                send_sem=p1_send_sem_slice.at[c],
                recv_sem=p1_recv_sem_slice.at[c],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            if wait_send:
                dma.wait_send()
            else:
                dma.wait_recv()

    @jax.named_scope("wait_phase2_c2c_copies")
    def wait_phase2_c2c_copies(self,
                               mb_idx,
                               step_idx,
                               src,
                               dst,
                               *,
                               wait_send: bool = False):
        """Waits for Phase 2 (C2C) remote copies from `src` to `dst` for `mb_idx` and `step_idx`.

    If `wait_send` is True, it blocks until the local send completes, releasing
    the source buffer slice; otherwise, it blocks until the remote receive
    completes.
    """
        slot_idx = mb_idx % 2
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
        p2_recv_sem_slice = self.p2_recv_sem.at[slot_idx, step_idx]
        p2_send_sem_slice = self.p2_send_sem.at[slot_idx, step_idx]

        dummy_send = src.at[
            step_idx,
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.hc_chunk_size),
        ]
        dummy_recv = dst.at[
            pl.ds(0, self.config.core_chunk_size),
            pl.ds(0, self.config.hc_chunk_size),
        ]

        @pl.loop(0, num_chunk_groups)
        def _(chunk_group_idx):

            @pl.loop(0, num_hcube_dims)
            def _(hcube_dim_idx):
                dim = (hcube_dim_idx + step_idx) % num_hcube_dims
                neighbor_device_id = self.topo.get_neighbor_device_id(dim)

                dma = pltpu.make_async_remote_copy(
                    src_ref=dummy_send,
                    dst_ref=dummy_recv,
                    send_sem=p2_send_sem_slice.at[chunk_group_idx,
                                                  hcube_dim_idx],
                    recv_sem=p2_recv_sem_slice.at[chunk_group_idx,
                                                  hcube_dim_idx],
                    device_id=neighbor_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                if wait_send:
                    dma.wait_send()
                else:
                    dma.wait_recv()


@dataclasses.dataclass(frozen=True)
class LocalDmaManager:
    """Handles local (HBM <-> VMEM) DMA copies and pipelines on TEC."""

    config: Config
    topo: Topology
    core_idx: jax.Array  # integer scalar.
    subcore_idx: jax.Array  # integer scalar.
    locator: ChunkLocator = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "locator",
            ChunkLocator(self.config, self.topo, self.core_idx,
                         self.subcore_idx),
        )

    def _local_subcore_copy(self, src_ref, dst_ref, sem):
        """Issues asynchronous DMAs to copy a subcore chunk between `src_ref` and `dst_ref` using `sem`."""
        if self.config.subcore_chunk_size % 8 == 0:
            pltpu.make_async_copy(src_ref, dst_ref, sem).start()
        else:
            # If the number of rows is not a multiple of 8, it iterates over
            # rows to satisfy layout alignment requirements.
            src_32b = src_ref.bitcast(jax.numpy.uint32)
            dst_32b = dst_ref.bitcast(jax.numpy.uint32)
            for i in range(src_32b.shape[0]):
                pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :],
                                      sem).start()

    def _wait_subcore_copies(self, src_dummy, dst_dummy, sem, num_copies=1):
        """Waits for `num_copies` subcore chunks to complete on `sem` using dummy async copies."""
        for _ in range(num_copies):
            if self.config.subcore_chunk_size % 8 == 0:
                pltpu.make_async_copy(src_dummy, dst_dummy, sem).wait()
            else:
                src_32b = src_dummy.bitcast(jax.numpy.uint32)
                dst_32b = dst_dummy.bitcast(jax.numpy.uint32)
                for i in range(src_32b.shape[0]):
                    pltpu.make_async_copy(src_32b.at[i, :], dst_32b.at[i, :],
                                          sem).wait()

    # TODO: Remove this and use emit_pipeline instead once correctness bug is
    # fixed that triggers when there are two src inputs.
    def _accumulate_pipeline(
        self,
        num_iters: int,
        get_src1_fn,
        get_src2_fn,
        get_out_fn,
        dtype,
        max_col_size,
    ):
        """Double-buffered accumulation of chunks across `num_iters` iterations."""
        if num_iters == 0:
            return

        @functools.partial(
            pl.run_scoped,
            src1_vmem_ref=pltpu.VMEM(
                (2, max(2, self.config.subcore_chunk_size), max_col_size),
                dtype),
            src2_vmem_ref=pltpu.VMEM(
                (2, max(2, self.config.subcore_chunk_size), max_col_size),
                dtype),
            src_sems=pltpu.SemaphoreType.DMA((num_iters, )),
            out_sems=pltpu.SemaphoreType.DMA((num_iters, )),
        )
        def _run(src1_vmem_ref, src2_vmem_ref, src_sems, out_sems):

            def copy_in_fn(idx, slot):
                src1 = get_src1_fn(idx)
                src2 = get_src2_fn(idx)
                r, c = src1.shape
                self._local_subcore_copy(src1, src1_vmem_ref.at[slot, :r, :c],
                                         src_sems.at[idx])
                self._local_subcore_copy(src2, src2_vmem_ref.at[slot, :r, :c],
                                         src_sems.at[idx])

            def wait_in_fn(idx, slot):
                src1 = get_src1_fn(idx)
                r, c = src1.shape
                with jax.named_scope("wait"):
                    self._wait_subcore_copies(
                        src1,
                        src1_vmem_ref.at[0, :r, :c],
                        src_sems.at[idx],
                        num_copies=2,
                    )

            def compute_fn(idx, slot):
                src1 = get_src1_fn(idx)
                r, c = src1.shape
                with jax.named_scope("accumulate"):
                    _accumulate(
                        config=self.config,
                        target=src1_vmem_ref.at[slot],
                        addend=src2_vmem_ref.at[slot],
                        num_rows=r,
                        num_cols=c,
                    )

            def copy_out_fn(idx, slot):
                out = get_out_fn(idx)
                r, c = out.shape
                self._local_subcore_copy(src1_vmem_ref.at[slot, :r, :c], out,
                                         out_sems.at[idx])

            def wait_out_fn(idx, slot):
                out = get_out_fn(idx)
                r, c = out.shape
                self._wait_subcore_copies(
                    src1_vmem_ref.at[0, :r, :c],
                    out,
                    out_sems.at[idx],
                    num_copies=1,
                )

            def _slot(idx):
                return idx % 2

            copy_in_fn(0, 0)

            @pl.loop(0, num_iters - 1)
            def _(idx):

                @pl.when(idx >= 1)
                def _():
                    wait_out_fn(idx - 1, _slot(idx - 1))

                copy_in_fn(idx + 1, _slot(idx + 1))
                wait_in_fn(idx, _slot(idx))
                compute_fn(idx, _slot(idx))
                copy_out_fn(idx, _slot(idx))

            last_idx = num_iters - 1
            wait_in_fn(last_idx, _slot(last_idx))
            compute_fn(last_idx, _slot(last_idx))
            copy_out_fn(last_idx, _slot(last_idx))

            if last_idx >= 1:
                wait_out_fn(last_idx - 1, _slot(last_idx - 1))
            wait_out_fn(last_idx, _slot(last_idx))

    def run_phase1_accumulate_pipeline(
        self,
        *,
        mb_idx,
        src1_ref,
        src2_ref,
        out_ref,
    ):
        """Executes Phase 1 pipelined accumulation for `mb_idx`."""
        num_iters = self.config.num_chips

        def get_src1_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return src1_ref.at[rs, cs]

        def get_src2_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return src2_ref.at[rs, cs]

        def get_out_fn(idx):
            chunk_idx = self.locator.get_phase1_chunk_idx(
                self.topo.cur_id, idx)
            rs, cs = self.locator.get_phase1_slice(chunk_idx,
                                                   mb_idx,
                                                   for_tec=True)
            return out_ref.at[0].at[rs, cs]

        self._accumulate_pipeline(
            num_iters,
            get_src1_fn,
            get_src2_fn,
            get_out_fn,
            src1_ref.dtype,
            self.config.subcore_col_chunk_size_p1,
        )

    def run_phase2_accumulate_pipeline(
        self,
        *,
        mb_idx,
        step_idx,
        src1_ref,
        src2_ref,
        final_out_ref,
    ):
        """Executes Phase 2 pipelined reduction for `mb_idx` and `step_idx`."""
        num_hcube_dims = self.config.num_hcube_dims
        num_chunk_groups = 1 << (num_hcube_dims - 1 - step_idx)
        num_iters = num_chunk_groups * num_hcube_dims
        is_last_step = step_idx == num_hcube_dims - 1

        def get_chunk_idx(idx):
            chunk_group_idx = idx // num_hcube_dims
            hcube_dim_idx = idx % num_hcube_dims
            return self.locator.get_phase2_chunk_idx(
                self.topo.cur_id,
                step_idx,
                chunk_group_idx=chunk_group_idx,
                hcube_dim_idx=hcube_dim_idx,
            )

        def get_src1_fn(idx):
            chunk_idx = get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            return src1_ref.at[step_idx, rs, cs]

        def get_src2_fn(idx):
            chunk_idx = get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            return src2_ref.at[step_idx + 1, rs, cs]

        def get_out_fn(idx):
            chunk_idx = 0 if is_last_step else get_chunk_idx(idx)
            hcube_dim_idx = idx % num_hcube_dims
            rs, cs = self.locator.get_phase2_slice(chunk_idx,
                                                   mb_idx,
                                                   hcube_dim_idx,
                                                   for_tec=True)
            final_dst = final_out_ref if is_last_step else src1_ref.at[step_idx
                                                                       + 1]
            return final_dst.at[rs, cs]

        self._accumulate_pipeline(
            num_iters,
            get_src1_fn,
            get_src2_fn,
            get_out_fn,
            src1_ref.dtype,
            self.config.subcore_col_chunk_size_p2,
        )

# ---- flattened from tpu_inference/kernels/collectives/hierrs_sc/kernel.py ----

import jax
from jax.experimental import pallas as pl





# ==============================================================================
#                 HIERARCHICAL REDUCE-SCATTER TIMELINE (D2D + C2C Step 0)
# ==============================================================================
# Hardware Execution Mapping:
#   - SCS (SparseCore Sequencer): Controls all asynchronous RDMA transfers
#     (D2D/C2C).
#   - TEC (Tile Core): Conducts all mathematical operations (Accumulation).
#
# Time -> t0            t1                      t2                      t3
#         | Prologue    |  Loop m=0             |  Loop m=1             |
#         |             |                       |                       |
# [SCS]   |             |                       |                       |
# D2D/DMA [A]====[B]    |  [D]====[E]           |  [D]====[E]           |
# (P1)    | P1 MB0      |  | P1 MB1             |  | P1 MB2             |
#         |             |  |                    |  |                    |
# [SCS]   |             |                       |                       |
# C2C     |             [C]=====================[G]                     |
# (P2)    |             |       P2 MB0          |                       |
#         |             |                       [F]=====================[G]
#         |             |                       |       P2 MB1          |
#         |             |                       |                       [F]====>
#         |             |                       |                       | P2 MB2
# [TEC]   |             |                       |                       |
# Accum   |      [B]====|          [E]====|     [G]====|         [I]====|
#         |        AC P1|            AC P1|     | AC P2|           AC P1|
#         |        (MB0)|            (MB1)|     | (MB0)|           (MB2)|
# ==============================================================================


def scs_kernel(
    # Inputs
    x_ref: jax.Ref,
    # Outputs.
    _: jax.Ref,  # output_ref, unused for SCS.
    running_sum_ref: jax.Ref,
    recv_buf_ref: jax.Ref,
    *,
    config: Config,
    axis_name: str | tuple[str, ...],
    # Scratch
    scs_to_tec: jax.Ref,
    tec_to_scs: jax.Ref,
    p1_recv_sem: jax.Ref,
    p2_recv_sem: jax.Ref,
    p1_send_sem: jax.Ref,
    p2_send_sem: jax.Ref,
    **unused_scratch,
):
    """Executes SparseCore Sequencer (SCS) execution logic for Reduce-Scatter.

  SCS is in charge of handling D2D and C2C ICI operations.
  """
    core_idx = jax.lax.axis_index("core")
    topo = Topology(axis_name)
    dma_manager = RemoteDmaManager(
        config,
        topo,
        core_idx,
        p1_send_sem=p1_send_sem,
        p2_send_sem=p2_send_sem,
        p1_recv_sem=p1_recv_sem,
        p2_recv_sem=p2_recv_sem,
    )

    def _signal_and_wait_tec(mb_idx, step):
        slot = mb_idx % 2
        for s in range(config.num_subcores):
            pl.semaphore_signal(scs_to_tec.at[slot, step],
                                device_id={"subcore": s})
        pl.semaphore_wait(tec_to_scs.at[slot, step], value=config.num_subcores)

    ############################################################################
    #                             PROLOGUE                                     #
    ############################################################################

    # [Step A - P1 MB0]: Start computing the initial prologue for pipeline by
    # firing D2D transfer for the very first micro-batch (MB), copying the first
    # `mb_size` block (MB 0) from local HBM over to the partner chiplet.
    dma_manager.start_phase1_d2d_copies(
        mb_idx=0,
        src=x_ref,
        dst=recv_buf_ref.at[0],
    )
    # [Step B - P1 MB0 Done]: Wait/block until the D2D transfer for MB 0
    # successfully done and the data is available in `recv_buf_ref[0]`, ensuring
    # the TEC has full data to begin accumulate phase 1.
    dma_manager.wait_phase1_d2d_copies(mb_idx=0,
                                       src=x_ref,
                                       dst=recv_buf_ref.at[0])

    # [Accumulate P1 MB0]: Signal the TEC to start local accumulation for MB 0
    # using the D2D data just received, and block SCS execution until TEC finishes
    # updating `running_sum_ref`.
    _signal_and_wait_tec(mb_idx=0, step=0)

    # [Step C - P2-S0 MB0]: Kick off Phase 2's first step ICI transfers. Copies
    # the partially reduced `running_sum_ref` out to the neighor chip, storing the
    # data into `recv_buf_ref[1]`.
    dma_manager.start_phase2_c2c_copies(
        mb_idx=0,
        step_idx=0,
        src=running_sum_ref,
        dst=recv_buf_ref.at[1],
    )

    ############################################################################
    #                  MAIN PIPELINE LOOP (P1 + P2 Step 0)                     #
    ############################################################################
    @pl.loop(0, config.num_micro_batches - 1)
    def step0_loop(mb_idx):
        # [Step D - P1 MB i+1]: While SEC+TEC processes the CURRENT micro-batch,
        # let's say MB i, it asynchronously triggers D2D transfers for the NEXT
        # micro-batch, (MB i+1), perfectly overlapping D2D with C2C and
        # accumulation.
        dma_manager.start_phase1_d2d_copies(
            mb_idx=mb_idx + 1,
            src=x_ref,
            dst=recv_buf_ref.at[0],
        )
        # [Step E - P1 MB i+1 Done]: Block until the D2D transfer for MB i+1 is
        # complete. This is safe because this isn't on the critical path of the
        # pipeline.
        dma_manager.wait_phase1_d2d_copies(mb_idx=mb_idx + 1,
                                           src=x_ref,
                                           dst=recv_buf_ref.at[0])

        # [Accumulate P1 MB i+1]: Signal the TEC to start accumulating MB i+1, and
        # wait here until the subcores finish accumulation and storing the reduced
        # chunk into `running_sum_ref`.
        _signal_and_wait_tec(mb_idx=mb_idx + 1, step=0)

        # [Step F - P2-S0 MB i+1]: Immediately queue Phase 2 Step 0 ICI transfers
        # for the NEXT micro-batch (MB i+1).
        dma_manager.start_phase2_c2c_copies(
            mb_idx=mb_idx + 1,
            step_idx=0,
            src=running_sum_ref,
            dst=recv_buf_ref.at[1],
        )

        # [Step G - P2-S0 MB i Done]: Block until the Phase 2 Step 0 ICI transfers
        # for the CURRENT micro-batch (MB i) is complete. We expect the newly
        # reduced chunk to arriving into `recv_buf_ref[1]`, then signal and wait for
        # the TEC to accumulate those chunks.
        dma_manager.wait_phase2_c2c_copies(mb_idx=mb_idx,
                                           step_idx=0,
                                           src=running_sum_ref,
                                           dst=recv_buf_ref.at[1])
        _signal_and_wait_tec(mb_idx=mb_idx, step=1)

    ############################################################################
    #                 EPILOGUE AND PRE-START P2-S1                             #
    ############################################################################
    last_mb_idx = config.num_micro_batches - 1
    # [P2-S1 MB 0]: Pre-start Phase 2 Step 1 Ring ICI transfers for MB 0,
    # overlapping with final MB execution for better pipeline saturation.
    # This will be done later if there's only 1 MB due to data depedency,
    # specifically accumulation for P2-S0 MB0 is not yet done.
    if config.num_micro_batches > 1:
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=1,
            src=running_sum_ref,
            dst=recv_buf_ref.at[2],
        )
    # [Step G - P2-S0 MB Last Done]: Wait/block until the Phase 2 Step 0 hypercube
    # transfers finish arriving for the last micro-batch.
    dma_manager.wait_phase2_c2c_copies(
        mb_idx=last_mb_idx,
        step_idx=0,
        src=running_sum_ref,
        dst=recv_buf_ref.at[1],
    )
    # [Accumulate P2-S0 MB Last]: Signal the TEC to start accumulating the
    # last MB, and wait here until subcores finish accumulation.
    _signal_and_wait_tec(mb_idx=last_mb_idx, step=1)

    # [P2-S1 MB 0]: When there's only 1 micro-batch, start ICI (Phase 2) for step
    # 1 once accumulation for Phase 2 Step 0 is done.
    if config.num_micro_batches == 1:
        dma_manager.start_phase2_c2c_copies(
            mb_idx=0,
            step_idx=1,
            dst=recv_buf_ref.at[2],
            src=running_sum_ref,
        )

    ############################################################################
    #                           PHASE 2 STEP 1+ LOOP                           #
    ############################################################################
    for step_idx in range(1, config.num_hcube_dims):
        #
        # PIPELINE ICI and ACCUMULATION (P2 Step i)
        #
        @pl.loop(0, config.num_micro_batches - 1)
        def step_loop(mb_idx):
            # [P2-S_step MB i Done]: Block until ICI transfers finish for the CURRENT
            # MB (MB i)
            dma_manager.wait_phase2_c2c_copies(
                mb_idx=mb_idx,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
            )
            # [P2-S_step MB i+1]: Immediately queue Phase 2 for ICI transfers
            # for the NEXT micro-batch across the ICI ring to minimize latency
            # stalling.
            dma_manager.start_phase2_c2c_copies(
                mb_idx=mb_idx + 1,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
            )
            # [Accumulate P2-S_step MB i]: Signal the TEC to accumulate the newly
            # arrived chunk, and wait here until the subcores finish accumulation.
            _signal_and_wait_tec(mb_idx, step_idx + 1)

        #
        # EPILOGUE AND PRE-START P2 STEP i+1
        #
        last_mb_idx = config.num_micro_batches - 1
        # [P2-S_step MB Last Done]: Block until the ICI transfers finish for the
        # last MB and the reduced chunk is available.
        dma_manager.wait_phase2_c2c_copies(
            mb_idx=last_mb_idx,
            step_idx=step_idx,
            src=running_sum_ref,
            dst=recv_buf_ref.at[step_idx + 1],
        )

        # [P2-S_step+1 MB 0]: Pre-start Phase 2 Step i+1 ICI for MB 0, overlapping
        # with final MB accumulation for maximize ICI bandwidth saturation.
        # This will be done later if there's only 1 MB due to data depedency,
        # specifically accumulation for P2-S_i MB0 is not yet done.
        if config.num_micro_batches > 1 and step_idx < config.num_hcube_dims - 1:
            dma_manager.start_phase2_c2c_copies(
                mb_idx=0,
                step_idx=step_idx + 1,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 2],
            )
        # [Accumulate P2-S_step MB Last]: Signal the TEC to accumulate the received
        # chunk, and wait here until the subcores finish acumulation.
        _signal_and_wait_tec(last_mb_idx, step_idx + 1)

        # [P2-S_step+1 MB 0]: When there's only 1 micro-batch, start ICI for the
        # NEXT step once accumulation for the current step's MB0 is done.
        if config.num_micro_batches == 1 and step_idx < config.num_hcube_dims - 1:
            dma_manager.start_phase2_c2c_copies(
                mb_idx=0,
                step_idx=step_idx + 1,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 2],
            )

    ############################################################################
    #                 CLEAN UP UN-WAITED SEND SEMAPHORES                       #
    ############################################################################
    # Resolve un-waited send semaphores
    @pl.loop(0, config.num_micro_batches)
    def wait_p1_sends_loop(mb_idx):
        dma_manager.wait_phase1_d2d_copies(mb_idx=mb_idx,
                                           src=x_ref,
                                           dst=recv_buf_ref.at[0],
                                           wait_send=True)

    @pl.loop(0, config.num_hcube_dims)
    def wait_p2_sends_step_loop(step_idx):

        @pl.loop(0, config.num_micro_batches)
        def wait_p2_sends_mb_loop(mb_idx):
            dma_manager.wait_phase2_c2c_copies(
                mb_idx=mb_idx,
                step_idx=step_idx,
                src=running_sum_ref,
                dst=recv_buf_ref.at[step_idx + 1],
                wait_send=True,
            )


def tec_kernel(
    # Inputs
    x_ref: jax.Ref,
    # Outputs
    output_ref: jax.Ref,
    running_sum_ref: jax.Ref,
    recv_buf_ref: jax.Ref,
    *,
    config: Config,
    axis_name: str | tuple[str, ...],
    # Scratch
    scs_to_tec: jax.Ref,
    tec_to_scs: jax.Ref,
    **unused_scratch,
):
    """Kernel execution impl that runs on the Tile Core (TEC).

  TEC is responsible for handling the computation logic needed for reduce
  scatter, which is basically the local accumulation of the incoming chunks of
  data (running sum) and corresponding chunks.
  """

    num_hcube_dims = config.num_hcube_dims
    core_idx = jax.lax.axis_index("core")
    subcore_idx = jax.lax.axis_index("subcore")

    topo = Topology(axis_name)
    dma_manager = LocalDmaManager(
        config,
        topo,
        core_idx,
        subcore_idx,
    )

    ############################################################################
    #                             PROLOGUE                                     #
    ############################################################################
    # [Step B - P1 MB0 Done]: Wait/block until SCS confirms the very first D2D
    # transfer (MB 0) is done and the data is available in `recv_buf_ref[0]`.
    pl.semaphore_wait(scs_to_tec.at[0, 0], value=1)
    # [Accumulate P1 MB0]: Conduct local accumulation for MB 0, which is
    # equivalent to (running_sum_ref = x_ref + recv_buf_ref). And signal SCS that
    # accumulation is done.
    dma_manager.run_phase1_accumulate_pipeline(
        mb_idx=0,
        src1_ref=x_ref,
        src2_ref=recv_buf_ref.at[0],
        out_ref=running_sum_ref,
    )
    pl.semaphore_signal(tec_to_scs.at[0, 0])

    ############################################################################
    #                  MAIN PIPELINE LOOP (P1 + P2 Step 0)                     #
    ############################################################################
    @pl.loop(0, config.num_micro_batches - 1)
    def step0_loop(mb_idx):
        curr_slot = mb_idx % 2
        next_slot = 1 - curr_slot

        # [Step E - P1 MB i+1 Done]: Block until SCS confirms the overlapped D2D
        # payload (MB i+1) has safely arrived and is available in `recv_buf_ref[0]`.
        pl.semaphore_wait(scs_to_tec.at[next_slot, 0], value=1)
        # [Step E - Accumulate P1 MB i+1]: Accumulate the recevied Phase 1 chunks
        # for MB i+1, which is equivalent to (running_sum_ref = x_ref +
        # recv_buf_ref). And signal SCS that accumulation is done.
        dma_manager.run_phase1_accumulate_pipeline(
            mb_idx=mb_idx + 1,
            src1_ref=x_ref,
            src2_ref=recv_buf_ref.at[0],
            out_ref=running_sum_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[next_slot, 0])

        # [Step G - P2-S0 MB i Done]: Block until the Phase 2 Step 0 ICI payload has
        # fully arrived for the CURRENT micro-batch (MB i).
        pl.semaphore_wait(scs_to_tec.at[curr_slot, 1], value=1)
        # [Step G - Accumulate P2-S0 MB i]: Accumulate the received Phase 2 Step 0
        # chunks for MB i, which is equivalent to (running_sum_ref += recv_buf_ref).
        # And signal SCS that accumulation is done.
        dma_manager.run_phase2_accumulate_pipeline(
            mb_idx=mb_idx,
            step_idx=0,
            src1_ref=running_sum_ref,
            src2_ref=recv_buf_ref,
            final_out_ref=output_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[curr_slot, 1])

    last_mb_idx = config.num_micro_batches - 1
    curr_slot = last_mb_idx % 2
    # [Step G - P2-S0 MB Last Done]: Block until the Phase 2 Step 0 ICI network
    # payload has fully arrived for the final micro-batch.
    pl.semaphore_wait(scs_to_tec.at[curr_slot, 1], value=1)
    # [Accumulate P2-S0 MB Last]: Accumulate the received Phase 2 Step 0 chunks
    # for MB Last, which is equivalent to (running_sum_ref += recv_buf_ref). And
    # signal SCS that accumulation is done.
    dma_manager.run_phase2_accumulate_pipeline(
        mb_idx=last_mb_idx,
        step_idx=0,
        src1_ref=running_sum_ref,
        src2_ref=recv_buf_ref,
        final_out_ref=output_ref,
    )
    pl.semaphore_signal(tec_to_scs.at[curr_slot, 1])

    ############################################################################
    #                     PHASE 2 STEP 1+ ACCUMULATION LOOP                    #
    ############################################################################
    def do_phase2_step(mb_idx, step_idx):
        curr_slot = mb_idx % 2
        # [P2-S_step MB i Done]: Block until Phase 2 Step `step_idx` ICI payload has
        # safely arrived for the CURRENT micro-batch (MB i).
        pl.semaphore_wait(scs_to_tec.at[curr_slot, 1 + step_idx], value=1)
        # [Accumulate P2-S_step MB i]: Accumulate the received Phase 2 Step
        # `step_idx` chunks for MB i, which is equivalent to (running_sum_ref +=
        # recv_buf_ref). And signal SCS that accumulation is done.
        dma_manager.run_phase2_accumulate_pipeline(
            mb_idx=mb_idx,
            step_idx=step_idx,
            src1_ref=running_sum_ref,
            src2_ref=recv_buf_ref,
            final_out_ref=output_ref,
        )
        pl.semaphore_signal(tec_to_scs.at[curr_slot, 1 + step_idx])

    for step_idx in range(1, num_hcube_dims):

        @pl.loop(0, config.num_micro_batches)
        def step_loop(mb_idx):
            do_phase2_step(mb_idx, step_idx)

# ---- flattened from tpu_inference/kernels/collectives/hierrs_sc/wrapper.py ----

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc



def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
) -> jax.Array:
    """Performs hierarchical Reduce-Scatter on SparseCore.

  This performs hierarchical recursive halving algorithm to perform Reduce
  scatter operation on SparseCore. It uses a 2-stage pipelined execution to
  overlap Die-to-Die ICI, Chip-to-Chip ICI, and local compute:

  Args:
    local_x: Local input shard of shape `[num_tokens, hidden_dim]`, representing
      the partial sum that the current device owns.
    num_devices: Total number of devices forming the reduction ring. Devices
      must be ordered by physical topology coordinates, with the chiplet
      dimension positioned as the last coordinate.
    num_micro_batches: Number of micro-batches to split the hidden dimension
      into for pipelining. If None, determined by heuristic.
    axis_name: Mesh axis name mapped explicitly out to the enclosing
      `jax.lax.shard_map` context.

  Returns:
    The reduced output array shard of shape `[num_tokens // num_devices,
    hidden_dim]`.
  """
    num_tokens, hidden_dim_size = local_x.shape
    chunk_size_orig = num_tokens // num_devices
    # Pad the local input to the minimum chunk size.
    min_chunk_size = pltpu.get_tpu_info().get_sublane_tiling(local_x.dtype)
    pad_amount = max(0, min_chunk_size - chunk_size_orig)
    reshaped_x = local_x.reshape(num_devices, -1, hidden_dim_size)
    padded_x = jnp.pad(
        reshaped_x,
        ((0, 0), (0, pad_amount), (0, 0)),
    )
    local_x = padded_x.reshape(-1, hidden_dim_size)
    num_tokens = local_x.shape[0]

    chunk_size = num_tokens // num_devices

    config = Config(
        num_devices=num_devices,
        hidden_dim_size=hidden_dim_size,
        chunk_size=chunk_size,
        num_tokens=num_tokens,
        dtype=local_x.dtype,
        _num_micro_batches=num_micro_batches,
    )

    scs_mesh = plsc.ScalarSubcoreMesh(axis_name="core",
                                      num_cores=config.num_cores)
    tec_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=config.num_cores,
        num_subcores=config.num_subcores,
    )

    out, _, _ = pl.kernel(
        interpret=False,
        body=[
            # SCS (SparseCore Sequencer) exclusively manages async D2D and C2C ICI
            # operations. TEC (Tile Core) strictly executes vector ALU
            # instructions for the accumulation math. They run concurrently as
            # decoupled cores, maintaining pipeline synchronization dynamically
            # utilizing hardware semaphores to gracefully hand-off buffers between
            # copies and compute.
            functools.partial(scs_kernel, config=config, axis_name=axis_name),
            functools.partial(tec_kernel, config=config, axis_name=axis_name),
        ],
        mesh=[scs_mesh, tec_mesh],
        out_type=(
            # output_ref
            jax.ShapeDtypeStruct((config.chunk_size, hidden_dim_size),
                                 local_x.dtype),
            # running_sum_ref[i, ...]: The accumulated result at each step (i=0 is reserved for Phase 1)
            jax.ShapeDtypeStruct((config.num_hcube_dims, *local_x.shape),
                                 local_x.dtype),
            # recv_buf_ref[i, ...]: The received data from peer at each step (i=0 is reserved for Phase 1)
            jax.ShapeDtypeStruct((config.num_hcube_dims + 1, *local_x.shape),
                                 local_x.dtype),
        ),
        scratch_types=dict(
            scs_to_tec=pltpu.SemaphoreType.REGULAR(
                (2, config.num_hcube_dims + 1)) @ tec_mesh,
            tec_to_scs=pltpu.SemaphoreType.REGULAR(
                (2, config.num_hcube_dims + 1)) @ scs_mesh,
            p1_send_sem=pltpu.SemaphoreType.DMA(
                (2, config.num_chips)) @ scs_mesh,
            p2_send_sem=pltpu.SemaphoreType.DMA((
                2,
                config.num_hcube_dims,
                config.num_chips // config.cores_per_chip,
                config.num_hcube_dims,
            )) @ scs_mesh,
            p1_recv_sem=pltpu.SemaphoreType.DMA(
                (2, config.num_chips)) @ scs_mesh,
            p2_recv_sem=pltpu.SemaphoreType.DMA((
                2,
                config.num_hcube_dims,
                config.num_chips // config.cores_per_chip,
                config.num_hcube_dims,
            )) @ scs_mesh,
        ),
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True, ),
    )(local_x)

    out = out[:chunk_size_orig, :]
    return out


kernel = hierarchical_reduce_scatter_local
