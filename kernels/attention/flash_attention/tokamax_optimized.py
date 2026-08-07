"""Standalone Tokamax Splash Attention TPU Pallas implementation.

Source:
  repository: https://github.com/openxla/tokamax
  commit: 927e3f94e8ffe0430cf38bd1423112bb2f69ec66
  paths:
    tokamax/_src/ops/experimental/tpu/splash_attention/
      splash_attention_mask.py
      splash_attention_mask_info.py
      base.py
      splash_attention_kernel.py
  transformation: repo-local mask, mask-info, base, and kernel modules were
    flattened in dependency order; module qualifiers were removed; Python 3.12
    type-alias statements were made Python 3.11-compatible.

**Two Pallas launch points**, not three.  This lineage fuses the dQ computation
into the dKV kernel -- one ``pallas_call`` returns ``dq_unreduced, dk, dv`` --
where the JAXBench and MaxText ``attention/`` splash kernels launch a separate
backward-dQ kernel.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/openxla/tokamax",
    "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
    "path": "tokamax/_src/ops/experimental/tpu/splash_attention",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "splash_mha_hsd",
    "launch_points": 2,
}

IMPLEMENTATION = "tokamax"


# ---- flattened from splash_attention_mask.py ----

# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Mini-mask creation library."""

from collections.abc import Callable
import dataclasses
from typing import Any, Self

import numpy as np

# mypy: ignore-errors


class Mask:
  """A base class for splash attention masks."""

  @property
  def shape(self) -> tuple[int, ...]:
    raise NotImplementedError

  def __getitem__(self, idx) -> np.ndarray:
    raise NotImplementedError

  def __bool__(self) -> bool:
    raise NotImplementedError(
        'Conversion to bool is unsupported. Could be caused by using logical'
        ' instead of bitwise operations on masks.'
    )

  def __or__(self, other: Self) -> Self:
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalOr(self, other)  # pyrefly: ignore[bad-return]

  def __and__(self, other: Self) -> Self:
    if self.shape != other.shape:
      raise ValueError(
          f'Invalid shape for other: {other.shape}, expected: {self.shape}'
      )
    return LogicalAnd(self, other)  # pyrefly: ignore[bad-return]


def make_causal_mask(shape: tuple[int, int], offset: int = 0) -> np.ndarray:
  """Makes a causal attention mask.

  Args:
    shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.

  Returns:
    The causal mask.
  """
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  return (q_idx[:, None] + offset >= kv_idx[None, :]).astype(np.bool_)


def make_local_attention_mask(
    shape: tuple[int, int],
    window_size: tuple[int | None, int | None],
    *,
    offset: int = 0,
) -> np.ndarray:
  """Makes a local attention mask."""
  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)
  mask = np.ones((q_seq_len, kv_seq_len), dtype=np.bool_)
  left, right = window_size
  if left is not None:
    mask = mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
  if right is not None:
    mask = mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])
  return mask.astype(np.bool_)


def make_chunk_attention_mask(
    shape: tuple[int, int], chunk_size: int
) -> np.ndarray:
  """Makes a chunked causal attention mask.

  Args:
    shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """
  if chunk_size <= 0:
    raise ValueError('chunk_size must be positive')

  q_seq_len, kv_seq_len = shape
  q_idx = np.arange(q_seq_len, dtype=np.int32)
  kv_idx = np.arange(kv_seq_len, dtype=np.int32)

  # chunk mask calculation
  same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
  mask = same_chunk & (q_idx[:, None] >= kv_idx[None, :])
  return mask


def make_random_mask(
    shape: tuple[int, int], sparsity: float, seed: int
) -> np.ndarray:
  """Makes a random attention mask."""
  np.random.seed(seed)
  return np.random.binomial(n=1, p=1.0 - sparsity, size=shape).astype(np.bool_)


@dataclasses.dataclass(slots=True)
class LogicalOr(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] | self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


@dataclasses.dataclass(slots=True)
class LogicalAnd(Mask):
  left: Mask
  right: Mask

  def __init__(self, left: Mask, right: Mask):
    if left.shape != right.shape:
      raise ValueError('Masks must have the same shape')
    self.left = left
    self.right = right

  @property
  def shape(self) -> tuple[int, ...]:
    return self.left.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.left[idx] & self.right[idx]

  def __hash__(self):
    return hash((type(self),) + (self.left, self.right))


class _ComputableMask(Mask):
  """Superclass for all masks that can be computed inside the kernel using a callable object.

  This subclass is designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention
  kernel, avoiding the memory cost of materializing the full
  (sequence_length, sequence_length) boolean mask array, which can be excessive
  for long sequences.

  Attributes:
    _shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
    q_sequence: Indices of Q sequence. q_sequence is reused across __getitem__
      calls which is important for compile-time performance.
    mask_function: Function used by the SplashAttention kernel to compute the
      mask rather than loading it.
  """

  _shape: tuple[int, int]
  q_sequence: np.ndarray
  mask_function: Callable[..., Any]

  def __init__(
      self,
      shape: tuple[int, int],
      mask_function: Callable[..., Any],
      shard_count: int = 1,
  ):
    self._shape = shape
    self.mask_function = mask_function
    q_seq_len = self.shape[0]

    if q_seq_len % (shard_count * shard_count) != 0:
      raise ValueError(
          f'Shard count squared ({shard_count * shard_count}) must'
          f' divide Q seq_len ({self.shape[0]}) evenly.'
      )

    self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    q_slice = _fill_slice(q_slice, self.shape[0])
    kv_slice = _fill_slice(kv_slice, self.shape[1])

    rows = self.q_sequence[q_slice]
    cols = np.arange(kv_slice.start, kv_slice.stop)

    return self.mask_function(rows[:, None], cols[None, :])

  def __eq__(self, other: object):
    raise NotImplementedError()

  def __hash__(self):
    raise NotImplementedError()


class CausalMask(_ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      offset: int = 0,
      shard_count: int = 1,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      # When evaluating the mask in _process_mask we typically work with numpy
      # array views.
      # Avoid the addition when possible to avoid instantiating an actual array.
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class ChunkedCausalMask(_ComputableMask):
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens
  attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.


  Attributes:
    chunk_size: The size of each attention chunk.
  """

  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError('chunk_size must be positive')
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      # Condition 1: Same chunk
      same_chunk = (q_ids // self.chunk_size) == (kv_ids // self.chunk_size)

      # Condition 2: Causal
      causal = q_ids >= kv_ids

      return same_chunk & causal

    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.chunk_size,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


class LocalMask(_ComputableMask):
  """Lazy local mask, prevents model from attending to tokens outside window.

  Attributes:
    window_size: Size of the two sides of the local window (None identifies no
      limit for the given side).
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  window_size: tuple[int | None, int | None]
  offset: int

  def __init__(
      self,
      shape: tuple[int, int],
      window_size: tuple[int | None, int | None],
      offset: int,
      shard_count: int = 1,
  ):
    self.window_size = window_size
    self.offset = offset

    def local_mask_function(q_ids, kv_ids):
      """Computes the local attention mask for the given slice indices."""
      left_size, right_size = self.window_size

      assert q_ids.ndim == 2
      assert kv_ids.ndim == 2

      if left_size is None and right_size is None:
        return np.ones((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

      # Avoid the addition when possible to avoid instantiating an actual array.
      if offset != 0:
        shifted_q_ids = q_ids + self.offset
      else:
        shifted_q_ids = q_ids

      mask = None
      if left_size is not None:
        mask = shifted_q_ids - left_size <= kv_ids
      if right_size is not None:
        if mask is None:
          mask = shifted_q_ids + right_size >= kv_ids
        else:
          mask &= shifted_q_ids + right_size >= kv_ids
      return mask

    super().__init__(
        shape=shape,
        mask_function=local_mask_function,
        shard_count=shard_count,
    )

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return False

    return (
        self.shape == other.shape
        and self.window_size == other.window_size
        and self.offset == other.offset
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash((
        type(self),
        self.shape,
        self.window_size,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
    ))


@dataclasses.dataclass(slots=True)
class NumpyMask(Mask):
  """A mask backed by a dense numpy array."""

  array: np.ndarray

  def __post_init__(self):
    if self.array.ndim != 2:
      raise ValueError('Expected a 2-dim array')

    if self.array.dtype != np.bool_:
      raise ValueError('Mask must be a boolean array')

  @property
  def shape(self) -> tuple[int, ...]:
    return self.array.shape

  def __getitem__(self, idx) -> np.ndarray:
    return self.array[idx]

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return np.array_equal(self.array, other.array, equal_nan=True)

  def __hash__(self):
    return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
  assert inp_slice.step is None or inp_slice.step == 1
  start = 0 if inp_slice.start is None else inp_slice.start
  stop = size if inp_slice.stop is None else inp_slice.stop
  assert start >= 0
  assert stop <= size
  return slice(start, stop, None)


@dataclasses.dataclass(frozen=True, slots=True)
class FullMask(Mask):
  """Lazy full mask, allows all tokens to attend to all other tokens."""

  # TODO: Transform FullMask into a _ComputableMask.

  _shape: tuple[int, int]

  def __post_init__(self):
    if not isinstance(self.shape, tuple):
      raise ValueError(f'Unsupported shape type: {type(self.shape)}')

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i, j = idx
    if not isinstance(i, slice) or not isinstance(j, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')
    i = _fill_slice(i, self.shape[0])
    j = _fill_slice(j, self.shape[1])
    return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape

  def __hash__(self):
    return hash((type(self), self.shape))


# ---- flattened from splash_attention_mask_info.py ----

# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Mini-mask creation library."""

import collections
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# mypy: ignore-errors

lax = jax.lax
MaskCallable = Any


def find_bounds(
    arr: jax.Array | np.ndarray,
) -> tuple[jax.Array | np.ndarray | None, jax.Array | np.ndarray | None]:
  # Find the first and last block of a row to determine when to initialize/store
  # the output.

  if arr is None:
    return None, None

  bounds_start = (arr != jnp.roll(arr, shift=1, axis=-1)).astype(jnp.int32)
  bounds_end = (arr != jnp.roll(arr, shift=-1, axis=-1)).astype(jnp.int32)
  bounds_start = bounds_start.at[0].set(1)
  bounds_end = bounds_end.at[-1].set(1)

  return bounds_start, bounds_end


# Logic for processing NumPy masks for kernels
class MaskInfo(NamedTuple):
  """Contains runtime masking information for the Splash attention kernel.

  The arrays, mask_next and block_mask are placed in TPU
  scalar-memory. This is a scarse resource so the mask creation logic attempts
  to shrink the data-type of these arrays to the smallest possible one.
  This can be: np.int32, np.int16 or np.int8.

  Attributes:
    mask_next: An integer[num_active_blocks] NumPy array where each entry
      contains the next mask block index in `partial_mask_blocks` to prefetch.
    active_rows: An integer[num_active_blocks] NumPy array where each entry
      contains the row index of the corresponding active block in the original
      mask.
    active_cols: An integer[num_active_blocks] NumPy array where each entry
      contains the column index of the corresponding active block in the
      original mask.
    block_mask: An integer[num_active_blocks] NumPy array where each entry is
      either 1 or 2. 1 means the corresponding block is full and 2 means the
      corresponding block is partially masked.
    num_active_blocks: An integer[] NumPy array whose entries are the sizes of
      the corresponding blocks in the original mask.
    partial_mask_blocks: An int8[num_partial_blocks, block_q, block_kv] NumPy
      array that contains the blocks of the original mask that contained both
      zeros and ones. The entries in `mask_next` point to indices in the first
      axis of this array.
    q_sequence: A i32[q_sequence_length] NumPy array. When using causal masking,
      this contains the list of indices that correspond to q tokens. For plain
      causal this is just np.arange(q_sequence_length).
  """

  mask_next: np.ndarray | jax.Array | None
  active_rows: np.ndarray | jax.Array | None
  active_cols: np.ndarray | jax.Array | None
  block_mask: np.ndarray | jax.Array | None
  num_active_blocks: np.ndarray | jax.Array | None
  partial_mask_blocks: np.ndarray | jax.Array | None
  q_sequence: np.ndarray | None


def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
  """Downcast numpy array.

  If possible, downcast the data-type of the input array to the smallest numpy
  type (among np.int16 and np.int8) that fits the content of the array.

  Args:
    array: the array to downcast

  Returns:
    The downcasted array.

  Raises:
    ValueError: if the input array is not np.int32 or if its elements are not
    all positive.
  """
  if array.dtype != np.int32:
    raise ValueError(f'Expected int32 input, but got {array.dtype}.')

  if not np.all(array >= -1):
    # Allow -1 for padding.
    raise ValueError('Expected non-negative array.')

  if array.size == 0:
    return array

  max_value = np.max(array)

  if max_value <= np.iinfo(np.int8).max:
    return array.astype(np.int8)
  elif max_value <= np.iinfo(np.int16).max:
    return array.astype(np.int16)
  else:
    return array.astype(np.int32)


def _check_mask(mask: Mask) -> None:
  """Check that the given mask is valid.

  A row of all zeros along the kv dimension would result in a division by zero
  when computing the softmax. This function is meant to protect against that
  case.

  Args:
    mask: the mask to check.

  Raises:
    ValueError: the mask is invalid.
  """

  assert len(mask.shape) == 2

  exception_message = (
      'Some rows of the mask (along the kv dimension) are all zeros.\nThis is'
      ' would result in a division by zero when computing the attention'
      ' softmax.'
  )

  is_row_non_zero = np.zeros(mask.shape[0], dtype=np.bool_)
  for col in range(mask.shape[1]):
    # Mask only supports slice indices.
    is_row_non_zero = np.logical_or(
        is_row_non_zero,
        mask[(slice(0, mask.shape[0]), slice(col, col + 1))][:, 0],
    )
  if not is_row_non_zero.all():
    raise ValueError(exception_message)


class _HashableNDArray:
  """Helper to make a numpy array hashable: can be added associative containers.

  Attributes:
    array: The underlying numpy array.
  """

  __slots__ = ('array', '_hash')
  array: np.ndarray

  def __init__(self, array: np.ndarray):
    self.array = array
    self._hash = hash(array.tobytes())

  def __hash__(self):
    return self._hash

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, _HashableNDArray):
      return NotImplemented
    return np.array_equal(self.array, other.array, equal_nan=True)


def _generate_shard_metadata(
    block_mask: np.ndarray,
    partial_blocks: np.ndarray,
    is_dkv: bool,
    return_dynamic_grid: bool,
):
  if is_dkv:
    block_mask = block_mask.mT
    partial_blocks = partial_blocks.mT

  if return_dynamic_grid:
    active_mask = block_mask > 0
    if is_dkv:
      # If an entire row is masked then that kv output tile won't be visited.
      # We extend the grid to visit these tiles to initialize them.
      active_mask[:, 0] |= ~active_mask.any(axis=1)
    active_indices = np.argwhere(active_mask)
    active_rows = active_indices[:, 0].astype(np.int32)
    active_cols = active_indices[:, 1].astype(np.int32)
    block_mask = block_mask[active_mask > 0]
    grid_size = active_rows.size
  else:
    active_indices = np.ndindex(block_mask.shape)
    active_rows = active_cols = grid_size = None

  partial_coords = np.argwhere(partial_blocks != -1)
  if partial_coords.size > 0:
    mask_next = []
    mask_coords_iter = iter([tuple(c) for c in partial_coords])
    first_m = coord_m = next(mask_coords_iter)

    for idx in active_indices:
      is_next_mask = tuple(idx) > tuple(coord_m)
      if is_next_mask:
        try:
          coord_m = next(mask_coords_iter)  # type: ignore
        except StopIteration:
          coord_m = first_m
      mask_next.append(partial_blocks[coord_m])
  else:
    mask_next = np.full(block_mask.size, -1, dtype=np.int32)

  mask_next = np.array(mask_next, dtype=np.int32)
  flat_block_mask = block_mask.flatten()

  return active_rows, active_cols, mask_next, flat_block_mask, grid_size


def _process_dynamic_mask(
    mask: jax.Array,
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
) -> MaskInfo:
  """Process a dynamic mask to compute it's local sparsity data.

  Note that this operates on a single shard of the mask.

  Args:
    mask: [q_seq_len, kv_seq_len] jax.Array representing a dense mask to
      process.
    block_shape: A Tuple[int, int] representing the shape of the Pallas grid
      block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the scalar-memory data of MaskInfo to
      a data type smaller than np.int32 (if possible).

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """
  if len(mask.shape) != 2:
    raise ValueError(f'Expected a 2-dim mask, instead got: {mask.shape}.')

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  # Tile the last 2 dimensions of the mask into 2D tiles of size `block_shape`.
  mask_blocks = (
      mask.reshape(
          q_blocks_count,
          q_block_size,
          kv_blocks_count,
          kv_block_size,
      )
      .swapaxes(-2, -3)
      .astype(partial_mask_blocks_dtype)
  )

  any_mask = jnp.any(mask_blocks, axis=(-1, -2)).astype(np.int32)
  all_mask = jnp.all(mask_blocks, axis=(-1, -2)).astype(np.int32)
  block_mask = any_mask + all_mask

  block_ids = jnp.arange(block_mask.size, dtype=np.int32).reshape(
      block_mask.shape
  )
  if is_dkv:
    block_mask = block_mask.swapaxes(-1, -2)
    block_ids = block_ids.swapaxes(-1, -2)
    mask_blocks = mask_blocks.swapaxes(-1, -2)

  active_mask = block_mask > 0
  if is_dkv:
    # If an entire row is masked then that kv output tile won't be visited.
    # We extend the grid to visit these tiles to initialize them.
    empty_rows = jnp.all(block_mask == 0, axis=-1)
    first_col = jnp.arange(block_mask.shape[1]) == 0
    active_mask |= (empty_rows[:, None] & first_col)

  num_active_blocks = active_mask.flatten().sum(keepdims=True)
  active_indices = jnp.argwhere(
      active_mask, size=active_mask.size, fill_value=-1
  )
  active_rows = active_indices[:, 0].astype(np.int32)
  active_cols = active_indices[:, 1].astype(np.int32)

  block_mask = block_mask[active_rows, active_cols]
  mask_next = block_ids.at[active_rows, active_cols].get(
      wrap_negative_indices=False
  )
  mask_next = jnp.where(block_mask == 1, mask_next, 0)

  # Mask out the blocks that aren't active.
  mask = (jnp.arange(block_mask.size) < num_active_blocks).astype(np.int32)
  block_mask = block_mask * mask

  # Collapsing because the block ids are linearized.
  mask_blocks = lax.collapse(mask_blocks, 0, 2)

  def _downcast(array: jax.Array, max_value: int) -> jax.Array:
    if array.size == 0:
      return array

    if array.dtype != np.int32:
      raise ValueError(f'Expected int32 input, but got {array.dtype}.')

    if max_value <= np.iinfo(np.int8).max:
      return array.astype(np.int8)
    elif max_value <= np.iinfo(np.int16).max:
      return array.astype(np.int16)
    else:
      return array.astype(np.int32)

  if downcast_smem_data:
    block_mask = block_mask.astype(np.int8)  # values are in the range [0, 1, 2]
    mask_next = _downcast(mask_next, q_blocks_count * kv_blocks_count)

  return MaskInfo(
      mask_next=mask_next,
      active_rows=active_rows,
      active_cols=active_cols,
      block_mask=block_mask,
      num_active_blocks=num_active_blocks,
      partial_mask_blocks=mask_blocks,
      q_sequence=None,
  )


# When used in a transformer network with multiple layers, the SplashAttention
# kernel is created several times with the same mask. Cache MaskInfo to avoid
# blowing up compile times. Ideally the size of the cache should be determined
# by the client.
@functools.lru_cache(maxsize=12)
def _process_mask(
    mask: Mask,  # [q_seq_len, kv_seq_len]
    block_shape: tuple[int, int],
    is_dkv: bool,
    *,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int = 1,
    kv_seq_shards: int = 1,
    return_dynamic_grid: bool = True,
) -> tuple[MaskInfo, MaskCallable | None]:
  """Transform a dense mask into a sparse representation.

  The number Q sequence shards are needed to create a MaskInfo
  object that is partitionable (with shard_map) along that dimension.
  Args:
    mask: Dense mask to process.
    block_shape: Shape of the Pallas grid block.
    is_dkv: True if we are processing the dKV mask
    downcast_smem_data: If True, downcast the SMEM data of MaskInfo to a data
      type smaller if possible.
    q_seq_shards: Number of Q sequence shards of the mesh in which the kernel is
      launched.

  Returns:
    `MaskInfo`, a sparse representation of the dense mask.
    `MaskCallable`: a callable that, given Q and KV indices, returns
      the value of the mask at those coordinates.

  Raises:
    ValueError: if the input mask is invalid or the block sizes are not
    compatible with the mask sizes.
  """

  if len(mask.shape) != 2:
    raise ValueError(f'Expected a 2-dim mask, instead got: {mask.shape=}')

  q_seq_len, kv_seq_len = mask.shape
  q_block_size, kv_block_size = block_shape
  q_blocks_count, q_mod = divmod(q_seq_len, q_block_size)
  kv_blocks_count, kv_mod = divmod(kv_seq_len, kv_block_size)

  if q_mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len=}.')
  if kv_mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len=}.')

  q_seq_len_per_shard, mod = divmod(q_seq_len, q_seq_shards)
  if mod != 0:
    raise ValueError(f'{q_seq_shards=} should divide {q_seq_len=}.')

  q_blocks_per_shard, mod = divmod(q_seq_len_per_shard, q_block_size)
  if mod != 0:
    raise ValueError(f'{q_block_size=} should divide {q_seq_len_per_shard=}.')

  kv_seq_len_per_shard, mod = divmod(kv_seq_len, kv_seq_shards)
  if mod != 0:
    raise ValueError(f'{kv_seq_shards=} should divide {kv_seq_len=}.')

  kv_blocks_per_shard, mod = divmod(kv_seq_len_per_shard, kv_block_size)
  if mod != 0:
    raise ValueError(f'{kv_block_size=} should divide {kv_seq_len_per_shard=}.')

  # TODO: checking the validity of the masks is slow for large masks.
  # Disable it for now, reevaluate in the future.

  # The mask object either define q_sequence and mask_function or none of
  # them.
  assert hasattr(mask, 'q_sequence') == hasattr(mask, 'mask_function')

  # If the mask object defines a q_sequence and a mask_function, then make use
  # of these in the kernel rather. This is preferable over loading the mask
  # from memory. When using a mask_function, then mask_next and
  # partial_mask_blocks are left undefined and not used in the kernel.
  if hasattr(mask, 'q_sequence') and hasattr(mask, 'mask_function'):
    q_sequence = mask.q_sequence
    mask_function = mask.mask_function
  else:
    q_sequence = mask_function = None

  # Identify the partial mask blocks and the value of the block mask for each
  # block.
  # Partial mask blocks are uniquified. When partitioning, all partial mask
  # blocks are replicated across shards.

  blocked_shape = (q_blocks_count, kv_blocks_count)
  state_grid = np.zeros(blocked_shape, dtype=np.int32)
  partial_id_grid = np.full(blocked_shape, -1, dtype=np.int32)

  partial_blocks_map = collections.defaultdict(lambda: len(partial_blocks_map))
  unique_chunks = []

  # Partition the dense mask into blocks and categorize them:
  # 0 = Empty, 1 = Partial (mixed 0s and 1s), 2 = Full (all 1s).
  # Partial blocks are deduplicated and stored in unique_chunks to save memory.
  for coords in np.ndindex((q_blocks_count, kv_blocks_count)):
    (q_idx, kv_idx) = coords
    chunk = mask[(
        slice(q_idx * q_block_size, (q_idx + 1) * q_block_size),
        slice(kv_idx * kv_block_size, (kv_idx + 1) * kv_block_size),
    )]
    if chunk.any():
      if chunk.all():
        state_grid[q_idx, kv_idx] = 2
      else:
        state_grid[q_idx, kv_idx] = 1
        chunk_id = partial_blocks_map[_HashableNDArray(chunk)]
        partial_id_grid[q_idx, kv_idx] = chunk_id

        if chunk_id == len(unique_chunks):
          unique_chunks.append(chunk)

  full_mask = (state_grid == 2).all()
  if full_mask:
    return MaskInfo(
        mask_next=None,
        active_rows=None,
        active_cols=None,
        block_mask=None,
        num_active_blocks=None,
        partial_mask_blocks=None,
        q_sequence=q_sequence,
    ), None

  if unique_chunks:
    partial_mask_blocks = np.stack(unique_chunks).astype(
        partial_mask_blocks_dtype
    )
    if is_dkv:
      partial_mask_blocks = partial_mask_blocks.mT
  else:
    partial_mask_blocks = None

  # Work on a fraction of the mask at the time to compute the mask. This is
  # needed to compute the correct data indices, which are relative to the
  # current slice of the mask.
  all_shards_metadata = []
  for q_shard_idx in range(q_seq_shards):
    for kv_shard_idx in range(kv_seq_shards):
      q_slice = slice(
          q_shard_idx * q_blocks_per_shard,
          (q_shard_idx + 1) * q_blocks_per_shard,
      )
      kv_slice = slice(
          kv_shard_idx * kv_blocks_per_shard,
          (kv_shard_idx + 1) * kv_blocks_per_shard,
      )
      metadata = _generate_shard_metadata(
          state_grid[q_slice, kv_slice],
          partial_id_grid[q_slice, kv_slice],
          is_dkv,
          return_dynamic_grid,
      )
      all_shards_metadata.append(metadata)

  (
      active_rows_slices,
      active_cols_slices,
      mask_next_slices,
      block_mask_slices,
      num_active_blocks,
  ) = zip(*all_shards_metadata)

  if return_dynamic_grid:
    # Pad each slice to the largest number of active blocks in any shard.
    max_size = max(num_active_blocks)
    pad_slice = lambda arr: np.pad(
        arr, (0, max_size - arr.shape[0]), mode='constant', constant_values=-1
    )
    active_rows_slices = list(map(pad_slice, active_rows_slices))
    active_cols_slices = list(map(pad_slice, active_cols_slices))
    mask_next_slices = list(map(pad_slice, mask_next_slices))
    block_mask_slices = list(map(pad_slice, block_mask_slices))

    # Concatenate the sequence shards.
    active_rows = np.concatenate(active_rows_slices, axis=0)
    active_cols = np.concatenate(active_cols_slices, axis=0)
    num_active_blocks = np.array(num_active_blocks, dtype=np.int32)

    if downcast_smem_data:
      active_rows = _downcast_to_small_type(active_rows)
      active_cols = _downcast_to_small_type(active_cols)
  else:
    active_rows = active_cols = num_active_blocks = None

  mask_next = np.concatenate(mask_next_slices, axis=0)
  block_mask = np.concatenate(block_mask_slices, axis=0)

  if downcast_smem_data:
    mask_next = _downcast_to_small_type(mask_next)
    block_mask = _downcast_to_small_type(block_mask)

  if partial_mask_blocks is None:
    mask_next = None

  assert (mask_function is not None) == (q_sequence is not None)
  # When the mask can be computed inside the kernel with a mask_function,
  # there is no need to load it from memory. So mask_next and
  # partial_mask_blocks are unused.
  return (
      MaskInfo(
          mask_next=mask_next if mask_function is None else None,
          active_rows=active_rows,
          active_cols=active_cols,
          block_mask=block_mask,
          num_active_blocks=num_active_blocks,
          partial_mask_blocks=partial_mask_blocks
          if mask_function is None
          else None,
          q_sequence=q_sequence,
      ),
      mask_function,
  )


process_mask = functools.partial(_process_mask, is_dkv=False)
process_mask_dkv = functools.partial(_process_mask, is_dkv=True)

process_dynamic_mask = functools.partial(_process_dynamic_mask, is_dkv=False)
process_dynamic_mask_dkv = functools.partial(_process_dynamic_mask, is_dkv=True)


# ---- flattened from py ----

# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Base functionality for Sparse Flash Attention."""

import functools
from typing import Final, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np



MaskInfo = MaskInfo


DEFAULT_MASK_VALUE: Final[float] = -0.7 * float(
    np.finfo(np.dtype("float32")).max
)


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array | jax.sharding.PartitionSpec  # [q_seq_len]
  kv: jax.Array | jax.sharding.PartitionSpec  # [kv_seq_len]


# Return type of SplashAttention function that implements the custom vjp rule.
SplashCustomReturnType = Any
SplashResidualsType = Any


def _attention_reference_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    save_residuals: bool,
    attn_logits_soft_cap: float | None,
) -> SplashCustomReturnType:
  logits = jnp.einsum("sd,td->st", q.astype(jnp.float32), k.astype(jnp.float32))

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap

  if sinks is not None:
    assert sinks.shape == ()  # should already be vmapped

  logits = jnp.where(mask, logits, mask_value)
  m = logits.max(axis=-1)
  sinks = None if sinks is None else sinks.astype(logits.dtype)
  m = m if sinks is None else jnp.maximum(m, sinks)
  s = jnp.exp(logits - m[..., None])
  l = s.sum(axis=-1) + (0 if sinks is None else jnp.exp(sinks - m))
  p = s / l[..., None]

  o = jnp.einsum("st,td->sd", p, v.astype(jnp.float32))

  if save_residuals:
    logsumexp = m + jnp.log(l)
    return o, {"logsumexp": logsumexp, "max_logits": m}
  return o


def _attention_reference_custom_bwd(
    do,
    q,
    k,
    v,
    mask,
    segment_ids,
    sinks,
    o,
    logsumexp,
    mask_value: float = DEFAULT_MASK_VALUE,
    backward_impl: str = "vanilla",
    attn_logits_soft_cap: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None, jax.Array | None]:
  uncapped_logits = jnp.einsum(
      "qc,kc->qk", q, k, preferred_element_type=jnp.float32
  )

  if attn_logits_soft_cap is not None:
    logits = jnp.tanh(uncapped_logits / attn_logits_soft_cap)
    logits = logits * attn_logits_soft_cap
  else:
    logits = uncapped_logits

  if segment_ids is not None:
    mask = jnp.logical_and(
        mask, segment_ids.q[:, None] == segment_ids.kv[None, :]
    )
  logits = jnp.where(mask, logits, mask_value)

  p = jnp.exp(logits - logsumexp[..., None])
  do = do.astype(jnp.float32)  # pytype: disable=attribute-error
  dv = jnp.einsum("pt,pd->td", p, do).astype(v.dtype)
  dp = jnp.einsum("pd,td->pt", do, v.astype(jnp.float32))

  # These two ways of computing ds are mathematically equivalent. The first
  # involves reducing over the head_dim dimension and the second involves
  # reducing over a sequence dimension. They tend to produce slightly different
  # numerics.
  if backward_impl == "flash":
    di = jnp.sum(o.astype(jnp.float32) * do, axis=-1)[..., None]
  else:
    di = jnp.einsum("st,st->s", dp, p)[:, None]
  ds = (dp - di) * p
  if attn_logits_soft_cap is not None:
    normalized = uncapped_logits / attn_logits_soft_cap
    d = jnp.tanh(normalized)
    g = ds * (1 - d)
    ds = g + g * d
  dk = jnp.einsum("sd,st->td", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("st,td->sd", ds, k.astype(jnp.float32)).astype(q.dtype)
  dsinks = None
  if sinks is not None:
    sinks_exp = -jnp.exp(
        sinks[..., None, None].astype(jnp.float32)
        - logsumexp[..., None].astype(jnp.float32)
    )
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))
  return dq, dk, dv, None, None, dsinks


@functools.partial(
    jax.jit,
    static_argnames=[
        "mask_value",
        "save_residuals",
        "attn_logits_soft_cap",
        "is_mqa",
    ],
)
def attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    attn_logits_soft_cap: float | None = None,
):
  """A JIT-compiled reference implementation of attention, handles MQA and MHA."""
  attn_impl = functools.partial(
      _attention_reference_impl,
      mask_value=mask_value,
      save_residuals=save_residuals,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  if is_mqa:
    func = jax.vmap(attn_impl, in_axes=(0, None, None, None, None, 0))
  else:
    # In grouped attention (1 < num_kv_heads && num_kv_heads < num_q_heads).
    # We interleave the KV heads across the Q heads.
    # For example: for 8 Q heads and 4 KV heads:
    # Q head [0, 1] see KV head 0
    # Q head [2, 3] see KV head 1
    # Q head [4, 5] see KV head 2
    # Q head [6, 7] see KV head 3

    kv_heads, q_heads = k.shape[0], q.shape[0]
    assert q_heads % kv_heads == 0

    if kv_heads < q_heads:
      # Repeat K and V heads to match the number of Q heads.
      q_heads_per_kv = q_heads // kv_heads
      k = jnp.repeat(k, repeats=q_heads_per_kv, axis=0)
      v = jnp.repeat(v, repeats=q_heads_per_kv, axis=0)

    func = jax.vmap(attn_impl, in_axes=(0, 0, 0, None, None, 0))

  out = func(q, k, v, mask, segment_ids, sinks)
  return out


@functools.partial(
    jax.jit, static_argnames=["is_mqa", "backward_impl", "attn_logits_soft_cap"]
)
def attention_reference_vjp(
    do,
    q,
    k,
    v,
    mask,
    segment_ids,
    sinks,
    o,
    logsumexp,
    *,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    attn_logits_soft_cap: float | None = None,
):
  """Wrapper for backward reference that handles GQA/MQA broadcasting and reduction."""
  bwd = functools.partial(
      _attention_reference_custom_bwd,
      backward_impl=backward_impl,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )

  num_q_heads = q.shape[0]
  num_kv_heads = 1 if is_mqa else k.shape[0]

  is_grouped = not is_mqa and num_kv_heads < num_q_heads
  assert num_q_heads % num_kv_heads == 0
  head_multiplier = num_q_heads // num_kv_heads
  if is_mqa:
    bwd = jax.vmap(bwd, in_axes=(0, 0, None, None, None, None, 0, 0, 0))
  else:
    bwd = jax.vmap(bwd, in_axes=(0, 0, 0, 0, None, None, 0, 0, 0))
    # Interleave the KV heads to match the corresponding Q heads.
    if is_grouped:
      k = jnp.repeat(k, head_multiplier, axis=0)
      v = jnp.repeat(v, head_multiplier, axis=0)

  dq, dk, dv, _, _, dsinks = bwd(
      do, q, k, v, mask, segment_ids, sinks, o, logsumexp
  )

  if is_mqa:
    dk, dv = dk.sum(axis=0), dv.sum(axis=0)
  elif is_grouped:
    # Perform the sum reduction across the head_multiplier dimension only.
    # So that the output still has KV heads.
    dk = dk.reshape(num_kv_heads, head_multiplier, *dk.shape[1:])
    dv = dv.reshape(num_kv_heads, head_multiplier, *dv.shape[1:])
    dk, dv = dk.sum(axis=1), dv.sum(axis=1)

  return dq, dk, dv, dsinks


# ---- flattened from splash_attention_kernel.py ----

# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention."""

from collections.abc import Callable
import dataclasses
import enum
import functools
import json
import math
from typing import Any, NamedTuple

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np





P = jax.P
MaskInfo = MaskInfo
partial = functools.partial
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)
LOG2E_INV = 1 / LOG2E

# mypy: ignore-errors


def _not(x: jax.Array | bool) -> jax.Array | bool:
  if isinstance(x, jax.Array):
    return jnp.logical_not(x)
  return not x


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  Attributes:
    q: segment ids along the Q sequence
    kv: segment ids along the KV sequence
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    is_mqa: bool, save_residuals: bool, is_segmented: bool, phase: str
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  assert phase in ["dq", "dkv", "fwd"]
  # Saving residuals is supported only for the fwd phase.
  assert not save_residuals or phase == "fwd"
  residuals = "_residuals" if save_residuals else "_no_residuals"
  attention_type = "mqa" if is_mqa else "mha"
  segments = "_segmented" if is_segmented else ""
  return f"splash_{attention_type}_{phase}{segments}{residuals}"


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return vals
  return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  # TODO: Remove these 3 params, they're only kept for backwards compatibility.
  block_q_dq: int | None = None
  block_kv_dq: int | None = None
  use_fused_bwd_kernel: bool = True
  num_stacked_q_heads: int = 1
  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  fwd_cost_estimate: pl.CostEstimate | None = None
  bwd_cost_estimate: pl.CostEstimate | None = None

  residual_checkpoint_name: str | None = None  # whether to checkpoint outputs
  attn_logits_soft_cap: float | None = None
  fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
  use_base2_exp: bool = True
  max_logit_const: float | None = None
  interpret: bool = False
  # The fused bwd kernel accumulates dq at every grid step. To safely avoid
  # read/write conflicts we conservatively avoid *any* in-kernel reductions.
  # This parameter allows to override this behavior and specifies the number of
  # reduction steps. For now, only 3 or all the kv steps are supported.
  dq_reduction_steps: int | None = None
  # An experimental scheduler that sometimes produces better softmax overlap.
  use_experimental_scheduler: bool = False
  # Skip the wasted causal-diagonal QK matmul. On a partial-mask (diagonal) block,
  # split the QK matmul into a qk_diag_grid x qk_diag_grid sub-grid over (kv rows,
  # q cols) and skip every sub-tile that lies entirely above the causal line
  # (kv > q), filling mask_value instead of computing it. Those entries are
  # overwritten to mask_value by _apply_mask_and_soft_cap regardless, so the result
  # is bit-exact; the elementwise/softmax ops still run on the full assembled tile.
  # PRECONDITION (enforced below): pure CausalMask + aligned SQUARE blocks with a
  # single compute tile per block (block_q == block_kv == block_kv_compute, and the
  # backward trio), and sequence length a multiple of the block. Any other config
  # raises (it does not silently corrupt); a non-causal mask raises too.
  qk_diag_skip: bool = False
  # Granularity of the diagonal skip: grid=2 -> quadrants (skip 1/4 of the diagonal
  # block, per-block waste 1/2 -> 1/4); grid=4 -> 4x4 (skip 6/16, waste -> 1/8).
  # Larger grid skips more of the triangle but uses smaller (less MXU-efficient)
  # matmuls; grid=4 is a good default at S=4096/block=2048. Must be a power of 2.
  qk_diag_grid: int = 2

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)

    if self.dq_reduction_steps is not None and self.dq_reduction_steps != 3:
      raise ValueError(
          f"Invalid dq_reduction_steps: {self.dq_reduction_steps}, only 3 or"
          " None are supported."
      )
    if not self.use_fused_bwd_kernel:
      raise ValueError("Only the fused bwd kernel is supported.")

    if self.qk_diag_skip:
      # The skip fills mask_value for sub-tiles where kv > q, relying on the mask to
      # mask EXACTLY those. That holds only for aligned SQUARE blocks (kv-band > q-band
      # <=> fully above the causal line, single compute tile per block) — enforce it or
      # the skip silently corrupts. Causality is checked in _make_splash_attention.
      if not (self.block_q == self.block_kv == self.block_kv_compute):
        raise ValueError(
            "qk_diag_skip requires square forward blocks "
            "(block_q == block_kv == block_kv_compute); got "
            f"{self.block_q}/{self.block_kv}/{self.block_kv_compute}."
        )
      if self.has_backward_blocks and not (
          self.block_q_dkv == self.block_kv_dkv == self.block_kv_dkv_compute
      ):
        raise ValueError(
            "qk_diag_skip requires square backward blocks "
            "(block_q_dkv == block_kv_dkv == block_kv_dkv_compute); got "
            f"{self.block_q_dkv}/{self.block_kv_dkv}/{self.block_kv_dkv_compute}."
        )
      if self.qk_diag_grid < 2 or (self.qk_diag_grid & (self.qk_diag_grid - 1)):
        raise ValueError(
            f"qk_diag_grid must be a power of 2 >= 2; got {self.qk_diag_grid}."
        )

  @property
  def has_backward_blocks(self) -> bool:
    backward_blocks = (
        self.block_q_dkv,
        self.block_kv_dkv,
        self.block_kv_dkv_compute,
    )
    return all(b is not None for b in backward_blocks)

  @classmethod
  def get_default(cls):
    # TODO: Select better parameters based on a heuristic.
    return SplashConfig(
        block_q=128,
        block_kv=128,
        block_kv_compute=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
        fuse_reciprocal=True,
    )


to_i32 = lambda x: x.astype(jnp.int32)


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
    has_partial_mask: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if has_partial_mask:
    if mask_ref is not None:
      mask = mask_ref[:, k_slice] if k_in_lanes else mask_ref[k_slice, :]
      masks.append(mask)
    elif mask_function is not None:
      # Compute the mask using the given q_sequence indices.
      # KV indices are computed on the fly. This works because we only support Q
      # sequence sharding. If we wanted to compute Q indices too, then we would
      # need to keep into account the current shard along Q sequence.

      if k_in_lanes:
        assert q_sequence_ref.shape == (bq, NUM_LANES)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (bq, k_slice.size), 1
        )

        repeats, rem = divmod(k_slice.size, NUM_LANES)
        assert rem == 0
        q_sequence = jnp.tile(
            q_sequence_ref[...], (1, repeats)  # pyrefly: ignore[unsupported-operation]
        )  # [bq, k_slice.size]
      else:
        assert q_sequence_ref.shape == (NUM_SUBLANES, bq)  # pyrefly: ignore[missing-attribute]

        k_sequence = k_offset + jax.lax.broadcasted_iota(
            jnp.int32, (k_slice.size, bq), 0
        )
        q_sequence = q_sequence_ref[:1, :]  # [1, bq]  # pyrefly: ignore[unsupported-operation]
        q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

      assert q_sequence.shape == k_sequence.shape
      computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
      if computed_mask.dtype != jnp.dtype(jnp.bool_):
        raise ValueError(
            "Mask function must return a boolean-valued array, but got:"
            f" {computed_mask.dtype}"
        )
      masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = jnp.tile(
          kv_segment_ids_ref[k_slice, :], (1, repeats)
      )  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    if mask.ndim == 2 and qk.ndim == 3:
      mask = jnp.expand_dims(mask, axis=0)

    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  # If the head_dim_v is not a multiple of the number of lanes, it will be
  # padded to that multiple with zeros.
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

  grid_idx = pl.program_id(1)
  h = pl.program_id(0)

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
    j = active_cols_ref[grid_idx].astype(jnp.int32)
  else:
    should_not_mask = False
    j = grid_idx % kv_steps
    should_initialize = j == 0
    should_write = j == kv_steps - 1

  max_logit_estimate = config.max_logit_const  # potentially None
  if max_logit_value_ref is not None:  # already ensures max_logit_const is None
    assert num_stacked_q_heads == 1
    max_logit_estimate = max_logit_value_ref[0, h]

  if config.use_base2_exp and max_logit_estimate is not None:
    max_logit_estimate *= LOG2E

  @pl.when(should_initialize)
  def init():
    o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    sink = None
    if sinks_ref is not None:
      sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
      if config.use_base2_exp:
        sink *= LOG2E

    if sinks_ref is None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is None and max_logit_estimate is not None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
      l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    elif sinks_ref is not None and max_logit_estimate is None:
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
    else:  # sinks_ref is not None and max_logit_estimate is not None
      exp = jnp.exp2 if config.use_base2_exp else jnp.exp
      m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type]
      l_scratch_ref[...] = exp(
          sink - jnp.full_like(l_scratch_ref, max_logit_estimate)  # pyrefly: ignore[bad-argument-type, unsupported-operation]
      )

  def body(kv_compute_index, _, has_partial_mask=False):
    slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
    m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
    assert m_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)
    assert l_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)

    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
    if config.use_base2_exp:
      q *= LOG2E

    head_dim_qk = q.shape[-1]
    # Collapse the head and sequence dimensions for a larger matmul.
    q_flat = q.reshape((num_stacked_q_heads * bq, head_dim_qk))

    if config.k_layout == HEAD_DIM_MINOR:
      k = k_ref[slice_k, :]
      qk_dims = NT_DIM_NUMBERS
    else:
      k = k_ref[:, slice_k]
      qk_dims = NN_DIM_NUMBERS

    _g = config.qk_diag_grid
    if (
        config.qk_diag_skip
        and has_partial_mask
        and num_stacked_q_heads == 1
        and config.k_layout == HEAD_DIM_MINOR
        and bq % _g == 0
        and bkv_compute % _g == 0
    ):
      # Diagonal skip (forward): qk tile is [q, kv]. On an aligned square diagonal
      # block, sub-tile (q-band qi, kv-band kj) with kj > qi is fully above the causal
      # boundary (kv > q) -> masked to mask_value anyway -> skip its matmul.
      sq = bq // _g
      sk = bkv_compute // _g
      q_parts = [q_flat[i * sq:(i + 1) * sq, :] for i in range(_g)]
      k_parts = [k[j * sk:(j + 1) * sk, :] for j in range(_g)]
      rows = []
      for qi in range(_g):  # q row-band
        cols = []
        for kj in range(_g):  # kv col-band
          if kj > qi:  # fully masked -> skip matmul
            cols.append(jnp.full((sq, sk), mask_value, dtype=float32))
          else:
            cols.append(lax.dot_general(
                q_parts[qi], k_parts[kj], qk_dims, preferred_element_type=float32
            ))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_flat = jnp.concatenate(rows, axis=0)
    else:
      qk_flat = lax.dot_general(
          q_flat, k, qk_dims, preferred_element_type=float32
      )
    qk = qk_flat.reshape((num_stacked_q_heads, bq, bkv_compute))

    apply_mask_and_soft_cap = functools.partial(
        _apply_mask_and_soft_cap,
        qk,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,
        k_offset=j * bkv + kv_compute_index * bkv_compute,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )

    qk = apply_mask_and_soft_cap()

    if max_logit_estimate is None:
      m_curr = qk.max(axis=-1)[..., None]  # pytype: disable=attribute-error
      assert m_curr.shape == (num_stacked_q_heads, bq, 1)
      m_next = jnp.maximum(m_prev, m_curr)
      assert m_next.shape == (num_stacked_q_heads, bq, NUM_LANES)
    else:
      m_next = None

    bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
    if rem != 0:
      raise NotImplementedError(
          f"{bkv_compute=} should be a multiple of {NUM_LANES}"
      )

    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    if max_logit_estimate is None:
      s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))  # pyrefly: ignore[bad-argument-type, unsupported-operation]
    else:
      s_curr = exp(qk - max_logit_estimate)  # pyrefly: ignore[unsupported-operation]
    assert s_curr.shape == (num_stacked_q_heads, bq, bkv_compute)

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
    assert l_curr.shape == (num_stacked_q_heads, bq, NUM_LANES)

    if max_logit_estimate is None:
      alpha = exp(m_prev - m_next)
      l_next = l_curr + alpha * l_prev
      m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
    else:
      alpha = None
      l_scratch_ref[...] = l_curr + l_prev

    s_curr_flat = s_curr.reshape((num_stacked_q_heads * bq, bkv_compute))

    if config.v_layout == HEAD_DIM_MINOR:
      v = v_ref[slice_k, :]
      sv_dims = NN_DIM_NUMBERS
    else:
      v = v_ref[:, slice_k]
      sv_dims = NT_DIM_NUMBERS

    o_curr_flat = lax.dot_general(s_curr_flat, v, sv_dims)
    o_curr = o_curr_flat.reshape((num_stacked_q_heads, bq, head_dim_v))

    if max_logit_estimate is None:
      alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))  # pyrefly: ignore[bad-argument-type]
      alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
      o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
    else:
      o_scratch_ref[...] = o_scratch_ref[...] + o_curr

  assert bkv % bkv_compute == 0
  num_iters = (
      k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
  )

  @pl.when(should_not_mask)
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )

  @pl.when(should_write)
  def end():
    l = l_scratch_ref[...]
    m = m_scratch_ref[...]
    if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
      l_inv = jnp.tile(1.0 / l, (1, 1, head_dim_v_repeats))
      l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
      o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
    else:
      o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
    if logsumexp_ref is not None:
      assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      log = jnp.log2 if config.use_base2_exp else jnp.log
      logsumexp = m + log(l)
      logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
    if l_linear_ref is not None:
      assert l_linear_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      l_linear_ref[...] = l.astype(l_linear_ref.dtype)
    if max_logits_ref is not None:
      assert max_logits_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
      max_logits_ref[...] = m.astype(max_logits_ref.dtype)


def _div(dividend: int, divisor: int):
  if divisor == 1:
    return dividend

  return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | None) -> int:
  if x is None:
    return 0

  if jnp.issubdtype(x.dtype, jnp.floating):
    info = jnp.finfo
  elif jnp.issubdtype(x.dtype, jnp.integer):
    info = jnp.iinfo
  else:
    raise ValueError(f"Unsupported dtype: {x.dtype}")
  return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> SplashCustomReturnType:
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  head_dim_v = v.shape[-1]
  bq, bkv = config.block_q, config.block_kv
  bkv_compute = config.block_kv_compute
  fuse_reciprocal = config.fuse_reciprocal or not save_residuals
  bounds_start, bounds_end = find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  num_stacked_q_heads = config.num_stacked_q_heads

  if num_stacked_q_heads > 1 and (
      sinks is not None or max_logit_value is not None
  ):
    raise ValueError(
        "Stacked heads are not supported with sinks or max_logit_value."
    )

  if is_mqa:
    expected_kv_rank = 2
    num_kv_heads = 1
  else:
    expected_kv_rank = 3
    num_kv_heads = k.shape[0]

  if len(k.shape) != expected_kv_rank:
    raise ValueError(
        f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
        f" {len(k.shape)}-dim one."
    )

  if k.shape[-1] != head_dim_qk:
    raise ValueError(
        f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
        f" {k.shape[-1]}."
    )

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if num_q_heads % num_stacked_q_heads != 0:
    raise ValueError(
        f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}"
    )

  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if q_heads_per_kv_head % num_stacked_q_heads != 0:
    raise ValueError(
        f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  if bkv % bkv_compute:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:  # pyrefly: ignore[unsupported-operation]
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  kv_seq_len = k.shape[-2]
  kv_steps = kv_seq_len // bkv
  dynamic_grid = mask_info.active_rows is not None

  if segment_ids is not None:
    assert isinstance(segment_ids.q, jax.Array)  # for pytype
    assert isinstance(segment_ids.kv, jax.Array)  # for pytype
    if segment_ids.q.shape != (q_seq_len,):
      raise ValueError(
          "Invalid shape for q segment_ids: "
          f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
      )
    if segment_ids.kv.shape != (kv_seq_len,):
      raise ValueError(
          "Invalid shape for kv segment_ids: "
          f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
      )
  if config.max_logit_const is not None and max_logit_value is not None:
    raise ValueError(
        f"Only one of {config.max_logit_const=} and"
        f" {max_logit_value=} can be set."
    )
  if max_logit_value is not None:
    if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
      raise ValueError(
          "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
          f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
      )
    max_logit_value = jnp.broadcast_to(
        jnp.atleast_1d(max_logit_value), (num_q_heads,)
    )

  q_layout = config.q_layout
  k_layout = config.k_layout
  v_layout = config.v_layout

  def unravel(f):
    def index_map(h_block, grid_idx, rows_ref, cols_ref, *_):
      if dynamic_grid:
        i = to_i32(rows_ref[grid_idx])
        j = to_i32(cols_ref[grid_idx])
      else:
        i = grid_idx // kv_steps
        j = grid_idx % kv_steps
      return f(h_block, i, j)

    return index_map

  def create_kv_index_map(layout):
    def index_map(h_block, i, j):
      del i  # Unused.
      first_h_in_block = h_block * num_stacked_q_heads
      prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  q_index_map = unravel(
      lambda h_block, i, j: from_head_minor((h_block, i, 0), q_layout)
  )
  out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
  k_index_map = unravel(create_kv_index_map(k_layout))
  v_index_map = unravel(create_kv_index_map(v_layout))

  def mask_index_map(
      h_block, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_
  ):
    del h_block, rows_ref, cols_ref  # Unused.
    next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
    return next_m, 0, 0

  q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
  kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))

  # Convert the logical shape from head-minor to sequence-minor.
  in_specs = [
      pl.BlockSpec(
          from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
          q_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
              k_layout,
          ),
          k_index_map,
      ),
      pl.BlockSpec(
          from_head_minor(
              (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
          ),
          v_index_map,
      ),
  ]
  if segment_ids is not None:
    in_specs += [
        pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
        pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
    ]
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (q_seq_len, NUM_LANES), (0,)  # pyrefly: ignore[bad-argument-type]
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (NUM_SUBLANES, kv_seq_len), (1,)  # pyrefly: ignore[bad-argument-type]
    )
  else:
    in_specs += [None, None]
    q_segment_ids = kv_segment_ids = None

  if sinks is not None:
    assert sinks.shape == (num_q_heads,), f"{sinks.shape=} != {num_q_heads=}"
    # align sinks to sublanes to allow vmap and shard_map over the kernel
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda h, i, j, *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
    sinks = jnp.broadcast_to(
        sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
    )
  else:
    in_specs += [None]

  if mask_info.partial_mask_blocks is not None:
    in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

  if mask_info.q_sequence is not None:
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
    )
    in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
  else:
    q_sequence = None
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  if max_logit_value is not None:
    # reshape to allow sublane selection for vmap-ping and shard_map-ping
    max_logit_value = jnp.broadcast_to(
        max_logit_value.astype(jnp.float32)[None, :],
        (NUM_SUBLANES, num_q_heads),
    )
    in_specs += [
        pl.BlockSpec(
            (NUM_SUBLANES, num_q_heads),
            lambda *_: (0, 0),
            memory_space=pltpu.SMEM,
        )
    ]
  else:
    in_specs.append(None)  # pyrefly: ignore[bad-argument-type]

  out_shapes = [
      jax.ShapeDtypeStruct((num_q_heads, q_seq_len, head_dim_v), q.dtype),
  ]
  out_specs = [
      pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
  ]
  if save_residuals:
    logsumexp_index_map = unravel(lambda h_block, i, j, *_: (h_block, i, 0))

    out_shapes += [
        # logsumexp
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if fuse_reciprocal
        else None,
        # l_linear
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
        if not fuse_reciprocal
        else None,
        # max_logits
        jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
    ]
    out_specs += [
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if fuse_reciprocal
        else None,
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        )
        if not fuse_reciprocal
        else None,
        pl.BlockSpec(
            (num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map
        ),
    ]
  else:
    out_shapes += [None, None, None]
    out_specs += [None, None, None]

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      is_segmented=segment_ids is not None,
      phase="fwd",
  )
  metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

  def _fwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      partial_mask_blocks: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    matmul_flops = (
        2 * q_seq_len * kv_seq_len * head_dim_qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v
    )

    # This is an upper bound because `mask_sparsity` is actually the mean
    # sparsity of the non-fully masked **blocks**.
    total_flops = num_q_heads * matmul_flops * mask_sparsity

    # Count expensive exp() calls
    transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

    inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))
    return pl.CostEstimate(
        flops=int(total_flops),
        transcendentals=int(transcendentals),
        bytes_accessed=int(input_bytes + output_bytes),
    )

  vmem_inputs = [
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      mask_info.partial_mask_blocks,
  ]
  cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
      *vmem_inputs, out_shapes, fwd_mask_sparsity  # pyrefly: ignore[bad-argument-count, bad-argument-type]
  )

  grid_size_h = num_q_heads // num_stacked_q_heads
  if dynamic_grid:
    num_active_blocks = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (grid_size_h, num_active_blocks)
    is_empty_attention_block = num_active_blocks == 0
  else:
    grid = (grid_size_h, kv_steps * (q_seq_len // bq))
    is_empty_attention_block = False

  with jax.named_scope(kernel_name):
    all_out = pl.pallas_call(
        partial(
            flash_attention_kernel,
            mask_value=mask_value,
            kv_steps=kv_steps,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            num_stacked_q_heads=num_stacked_q_heads,
            # note: fuse_reciprocal can only be False if save_residuals is True
            # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
            fuse_reciprocal=fuse_reciprocal,
            config=config,
            mask_function=mask_function,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=[
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # m_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                ),  # l_scratch
                pltpu.VMEM(
                    (num_stacked_q_heads, bq, head_dim_v), jnp.float32
                ),  # o_scratch
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": (
                    config.use_experimental_scheduler
                )
            },
        ),
        out_shape=out_shapes,
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
        q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
        k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
        v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
        q_segment_ids,
        kv_segment_ids,
        sinks,
        mask_info.partial_mask_blocks,
        q_sequence,
        max_logit_value,
    )
  out, logsumexp, l_linear, max_logits = all_out

  # If there is no compute to do within an attention block, then we want to
  # initialize the output and residuals to default values. Otherwise, we will
  # read uninitialized memory. This is a common case in ring attention.
  def init_if_empty(x: jax.Array, value: float) -> jax.Array:
    if not dynamic_grid:
      return x

    return jnp.where(is_empty_attention_block, value, x)

  out = init_if_empty(out, 0.0)

  if save_residuals:
    assert max_logits is not None
    max_logits = init_if_empty(max_logits[..., 0], mask_value)

    if fuse_reciprocal:
      assert logsumexp is not None
      logsumexp = init_if_empty(logsumexp[..., 0], mask_value)
    else:
      assert l_linear is not None
      log = jnp.log2 if config.use_base2_exp else jnp.log

      l = l_linear[..., 0]
      logsumexp = max_logits + log(l)
      out = (out / l[..., None]).astype(out.dtype)
  else:
    # If we're not saving residuals, then we can't fuse the reciprocal
    # out of the kernel.
    assert fuse_reciprocal

  if config.residual_checkpoint_name is not None:
    out = ad_checkpoint.checkpoint_name(
        out, name=config.residual_checkpoint_name
    )
    if logsumexp is not None:
      logsumexp = ad_checkpoint.checkpoint_name(
          logsumexp, name=config.residual_checkpoint_name
      )
  if save_residuals:
    stats = {"logsumexp": logsumexp, "max_logits": max_logits}
    stats = jax.tree.map(jax.lax.stop_gradient, stats)
    return out, stats
  return out


@partial(
    jax.custom_vjp,
    nondiff_argnames=(
        "save_residuals",
        "mask_value",
        "is_mqa",
        "config",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ),
)
def _splash_attention_custom(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> SplashCustomReturnType:
  # The forward function does not use the dq and dkv MaskInfos, it just forwards
  # them to the backward function as residuals. This is a way to communicate
  # arbitrary Arrays to the backward function. Since the three MaskInfos are
  # constants there is no overhead in passing them to the backward function as
  # residuals. When sharding computation MaskInfos are partitioned so both the
  # forward and the backward kernels need to work on the relevant slice. If we
  # recomputed the backward MaskInfos in the backward function from the numpy
  # mask then we would not work with the MaskInfo slice relevant to the current
  # device.
  del dkv_mask_info

  ret = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=save_residuals,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  if save_residuals:
    out, stats = ret
    if config.use_base2_exp:  # for user, output values in natural base
      stats["logsumexp"] = stats["logsumexp"] / LOG2E
      stats["max_logits"] = stats["max_logits"] / LOG2E
    return out, stats
  else:
    return ret


def _splash_attention_fwd(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
) -> tuple[tuple[jax.Array], SplashResidualsType]:

  # TODO: add some higher order AD check that isn't save_residuals based.
  # if save_residuals:
  #   raise NotImplementedError("Higher-order AD not supported.")

  out, stats = _splash_attention_forward(  # pytype: disable=wrong-arg-types
      fwd_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      config=config,
      save_residuals=True,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      max_logit_value=max_logit_value,
  )
  logsumexp = stats["logsumexp"]  # save in the config base for the bwd pass
  if config.use_base2_exp:  # for user, output values in natural base
    stats["logsumexp"] = stats["logsumexp"] / LOG2E
    stats["max_logits"] = stats["max_logits"] / LOG2E
  residuals = q, k, v, segment_ids, sinks, out, logsumexp, dkv_mask_info
  if save_residuals:
    return (out, stats), residuals  # pyrefly: ignore[bad-return]
  else:
    return out, residuals  # pyrefly: ignore[bad-return]


def _flash_attention_dq_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
):
  del mask_next_ref, active_rows_ref
  float32 = jnp.float32
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  grid_idx = pl.program_id(1)
  if block_mask_ref is not None:
    kv_index = active_cols_ref[grid_idx].astype(jnp.int32)
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index = grid_idx % kv_steps
    should_not_mask = False
    should_initialize = kv_index == 0
    should_write = kv_index == kv_steps - 1

  @pl.when(should_initialize)
  def init():
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)

  def body(has_partial_mask: bool = False):
    q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
    if config.use_base2_exp:
      q *= LOG2E
    # We keep k and v possibly transposed, since they are RHS of dots.
    k = k_ref[...]
    v = v_ref[...]
    logsumexp = jnp.expand_dims(logsumexp_ref[0], -1)
    do = do_ref[...]
    di = jnp.expand_dims(di_ref[0], -1)

    qk_dims = (
        NT_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    qk_uncapped = lax.dot_general(q, k, qk_dims, preferred_element_type=float32)

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=pl.ds(0, bkv),  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv,
        bq=bq,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)  # pyrefly: ignore[unsupported-operation]
    dp_dims = (
        NT_DIM_NUMBERS if config.v_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    dp = lax.dot_general(
        do.astype(v.dtype),
        v,
        dp_dims,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)

    dq_dims = (
        NN_DIM_NUMBERS if config.k_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dq_scratch_ref[...] += lax.dot_general(
        ds.astype(k.dtype),
        k,
        dq_dims,
        preferred_element_type=jnp.float32,
    )

  @pl.when(should_not_mask)
  def _():
    body()

  @pl.when(jnp.logical_not(should_not_mask))
  def _():
    body(has_partial_mask=True)

  @pl.when(should_write)
  def end():
    dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # aliases
    dq_alias,
    dk_alias,
    dv_alias,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    # Scratch
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    mask_value: float,
    q_steps: int,
    bq: int,
    bkv_compute: int,
    bkv: int,
    mask_function: MaskFunctionType | None,
    q_heads_per_kv_head: int,
    config: SplashConfig,
):
  del mask_next_ref, active_cols_ref
  HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
  attn_logits_soft_cap = config.attn_logits_soft_cap
  if attn_logits_soft_cap is not None and config.use_base2_exp:
    attn_logits_soft_cap *= LOG2E

  if active_rows_ref is not None:
    assert bounds_start_ref is not None
    assert bounds_end_ref is not None
    grid_idx = pl.program_id(1)
    kv_index = active_rows_ref[grid_idx].astype(jnp.int32)
    should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
    should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
  else:
    kv_index, q_head, q_index = (
        pl.program_id(0),
        pl.program_id(1),
        pl.program_id(2),
    )
    grid_idx = (kv_index * q_steps) + q_index
    should_initialize = q_index == 0
    should_write = True if q_steps <= 2 else q_index == q_steps - 1
    if q_heads_per_kv_head > 1:
      q_head_index_per_kv_head = lax.rem(q_head, q_heads_per_kv_head)
      should_initialize = jnp.logical_and(
          should_initialize, q_head_index_per_kv_head == 0
      )
      should_write = jnp.logical_and(
          should_write, q_head_index_per_kv_head == q_heads_per_kv_head - 1
      )

  if block_mask_ref is not None:
    should_not_mask = block_mask_ref[grid_idx].astype(jnp.int32) != 1
    should_run = block_mask_ref[grid_idx].astype(jnp.int32) != 0
  else:
    should_not_mask = False
    should_run = True

  # TODO: Update docstring explaining the accumulation logic

  # Consider this situation:
  # Q_heads:   0, 1, 2, 3, 4, 5, 6, 7
  # KV_heads:  0,    1,    2,    3
  # The gradient scratch buffers should be initialized for Q_heads 0, 2, 4, 6
  # (first Q_heads to 'see' a new KV_head).
  # The gradient output buffers should be written for Q_heads 1, 3, 5, 7 (last
  # Q_heads to 'see' the current KV_head).

  @pl.when(should_initialize)
  def init():
    dk_scratch_ref[...] = jnp.zeros_like(dk_scratch_ref)
    dv_scratch_ref[...] = jnp.zeros_like(dv_scratch_ref)

  def body(i, _, has_partial_mask=False):

    slice_k = pl.ds(i * bkv_compute, bkv_compute)
    q = q_ref[...]  # We keep q potentially transposed, since it's always RHS
    if config.use_base2_exp:
      scaled_q = q * LOG2E
    else:
      scaled_q = q

    def _load_kv(ref, layout):
      if layout == HEAD_DIM_MINOR:
        return ref[slice_k, :]
      return ref[:, slice_k].T

    k = _load_kv(k_ref, config.k_layout)
    v = _load_kv(v_ref, config.v_layout)
    logsumexp = logsumexp_ref[:1, :]
    do = do_ref[...]
    di = di_ref[:1, :]

    qk_dims = (
        NT_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NN_DIM_NUMBERS
    )
    _g = config.qk_diag_grid
    if (
        config.qk_diag_skip
        and has_partial_mask
        and bkv_compute % _g == 0
        and bq % _g == 0
    ):
      # Diagonal skip (backward dkv): qk tile is [kv, q]. On an aligned square diagonal
      # block, sub-tile (kv-band ki, q-band qj) with ki > qj is fully above the causal
      # boundary (kv > q) -> overwritten to mask_value anyway -> skip its matmul; compute
      # only ki <= qj sub-tiles; assemble the full tile for the single exp/ds/dv/dk.
      sk = bkv_compute // _g
      sq = bq // _g
      k_parts = [k[i * sk:(i + 1) * sk, :] for i in range(_g)]
      q_parts = [scaled_q[j * sq:(j + 1) * sq, :] for j in range(_g)]
      _mm = lambda kk, qq: lax.dot_general(
          kk, qq, qk_dims, preferred_element_type=jnp.float32
      )
      rows = []
      for ki in range(_g):  # kv row-band
        cols = []
        for qj in range(_g):  # q col-band
          if ki > qj:  # fully masked -> skip matmul
            cols.append(jnp.full((sk, sq), mask_value, dtype=jnp.float32))
          else:
            cols.append(_mm(k_parts[ki], q_parts[qj]))
        rows.append(jnp.concatenate(cols, axis=1))
      qk_uncapped = jnp.concatenate(rows, axis=0)
    else:
      qk_uncapped = lax.dot_general(
          k, scaled_q, qk_dims, preferred_element_type=jnp.float32
      )

    qk = _apply_mask_and_soft_cap(
        qk_uncapped,
        mask_value,
        mask_ref,
        q_sequence_ref,
        q_segment_ids_ref,
        kv_segment_ids_ref,
        attn_logits_soft_cap=attn_logits_soft_cap,
        k_slice=slice_k,  # pyrefly: ignore[bad-argument-type]
        k_offset=kv_index * bkv + i * bkv_compute,
        bq=bq,
        k_in_lanes=False,
        mask_function=mask_function,
        has_partial_mask=has_partial_mask,
    )
    exp = jnp.exp2 if config.use_base2_exp else jnp.exp
    p = exp(qk - logsumexp)
    dv = lax.dot(p.astype(do.dtype), do, preferred_element_type=jnp.float32)
    dv = dv.astype(dv_scratch_ref.dtype) + dv_scratch_ref[slice_k, :]
    dv_scratch_ref[slice_k, :] = dv

    dp = lax.dot_general(
        v,
        do,
        NT_DIM_NUMBERS,
        preferred_element_type=jnp.float32,
    )
    ds = (dp - di) * p
    if attn_logits_soft_cap is not None:
      normalized = qk_uncapped / attn_logits_soft_cap
      d = jnp.tanh(normalized)
      ds = ds * (1 - d * d)
    dk_dims = (
        NN_DIM_NUMBERS if config.q_layout == HEAD_DIM_MINOR else NT_DIM_NUMBERS
    )
    dk = lax.dot_general(
        ds.astype(do.dtype), q, dk_dims, preferred_element_type=jnp.float32
    )
    dk = dk.astype(dk_scratch_ref.dtype) + dk_scratch_ref[slice_k, :]
    dk_scratch_ref[slice_k, :] = dk
    if dq_scratch_ref is not None or dq_ref is not None:
      dq = lax.dot_general(
          ds.T.astype(k.dtype),
          k,
          NN_DIM_NUMBERS,
          preferred_element_type=jnp.float32,
      )
      if dq_scratch_ref is not None:
        # Compute block size != memory block size
        dq_scratch_ref[...] += dq
      else:
        # Compute block size == memory block size
        if dq_alias is not None:
          dq_ref[...] = dq_alias[...] + dq.astype(dq_ref.dtype)
        else:
          dq_ref[...] = dq.astype(dq_ref.dtype)

  if dq_scratch_ref is not None:
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)
  elif dq_alias is not None:
    dq_ref[...] = dq_alias[...]
  else:
    dq_ref[...] = jnp.zeros_like(dq_ref)

  num_iters = (
      k_ref.shape[0 if config.k_layout is HEAD_DIM_MINOR else 1] // bkv_compute
  )

  @pl.when(jnp.logical_and(should_not_mask, should_run))
  def _():
    lax.fori_loop(0, num_iters, body, None, unroll=True)

  @pl.when(jnp.logical_and(_not(should_not_mask), should_run))
  def _():
    lax.fori_loop(
        0, num_iters, partial(body, has_partial_mask=True), None, unroll=True
    )

  if dq_scratch_ref is not None:
    if dq_alias is not None:
      dq_ref[...] = dq_alias[...] + dq_scratch_ref[...].astype(dq_ref.dtype)
    else:
      dq_ref[...] = dq_scratch_ref[...].astype(dq_ref.dtype)

  if dk_alias is None:
    assert dv_alias is None

    @pl.when(should_write)
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

  else:
    q_head = pl.program_id(0)
    first_q_head_in_kv_group = lax.rem(q_head, q_heads_per_kv_head) == 0

    @pl.when(jnp.logical_and(should_write, first_q_head_in_kv_group))
    def _():
      dk_ref[...] = dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_scratch_ref[...].astype(dv_ref.dtype)

    @pl.when(jnp.logical_and(should_write, _not(first_q_head_in_kv_group)))
    def _():
      dk_ref[...] = dk_alias[...] + dk_scratch_ref[...].astype(dk_ref.dtype)
      dv_ref[...] = dv_alias[...] + dv_scratch_ref[...].astype(dv_ref.dtype)


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
    dkv_mask_sparsity: float,
):
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = find_bounds(mask_info.active_rows)  # pyrefly: ignore[bad-argument-type]
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
        "leading dimensions."
    )

  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]  # pyrefly: ignore[unsupported-operation]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])  # pyrefly: ignore[unsupported-operation]
      return next_m, 0, 0

  q_index_map = unravel(
      lambda h, i, j: from_head_minor((h, i, 0), config.q_layout)
  )
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(
      from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map
  )

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    return (*prefix, j, 0)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))

    q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
    kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,)
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv, (kv_seq_len, NUM_LANES), (0,)
    )
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)  # pyrefly: ignore[bad-argument-type]

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(
        mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,)
    )
  else:
    q_sequence = None
    in_specs.append(None)

  dq_reduction_steps = config.dq_reduction_steps
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  if dq_reduction_steps == 3:
    dq_index_map = unravel(lambda h, i, j: (j % 3, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *q.shape), q.dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: (j, h, i, 0))
    dq_spec = pl.BlockSpec((None, None, bq, head_dim_qk), dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *q.shape), q_dtype)

  in_specs += [dq_alias_spec]

  if bkv == bkv_compute:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM((bq, head_dim_qk), jnp.float32)

  if dynamic_grid and q_heads_per_kv_head != 1:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = k.dtype
    dv_type = v.dtype

  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(k.shape, dk_type),
      jax.ShapeDtypeStruct(v.shape, dv_type),
  ]
  out_specs = [dq_spec, dk_spec, dv_spec]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=config.use_experimental_scheduler,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  num_args = sum(1 for x in args if x is not None)
  input_output_aliases = {}
  if dq_reduction_steps == 3:
    if dynamic_grid and q_heads_per_kv_head != 1:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif dynamic_grid and q_heads_per_kv_head != 1:
    input_output_aliases = {num_args: 1, num_args + 1: 2}

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM((bkv, head_dim_qk), jnp.float32),
      pltpu.VMEM((bkv, head_dim_v), jnp.float32),
  ]

  def _bwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      logsumexp: jax.Array,
      do: jax.Array,
      di: jax.Array,
      partial_mask_blocks: jax.Array | None,
      q_sequence: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]

    total_matmul_flops_per_head = (
        2 * q_seq_len * kv_seq_len * head_dim_qk  # qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dv
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dp
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dq
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dk
    )

    estimated_flops = int(
        total_matmul_flops_per_head * num_q_heads * mask_sparsity_factor
    )

    exp_flops = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    if config.attn_logits_soft_cap is None:
      tanh_flops = 0
    else:
      tanh_flops = (
          2 * num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
      )
    estimated_transcendentals = int(exp_flops + tanh_flops)

    inputs_ = [
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        partial_mask_blocks,
        q_sequence,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))

    estimated_bytes = input_bytes + output_bytes

    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=estimated_bytes,
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,  # pyrefly: ignore[bad-argument-type]
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  with jax.named_scope(kernel_name):
    dq_unreduced, dk, dv = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * len(grid)
        ),
        name=kernel_name,
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv)
  dq = dq_unreduced.sum(axis=0)
  dq = dq.astype(q.dtype)
  dk = dk.astype(k.dtype)
  dv = dv.astype(v.dtype)
  return dq, dk, dv


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
    res: SplashResidualsType,
    grads: jax.Array | tuple[jax.Array, dict[str, jax.Array]],
) -> tuple[
    MaskInfo | None,  # fwd_mask_info
    MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segment_ids
    jax.Array | None,  # segment_ids
    jax.Array | None,  # max_logit_estimate
]:
  # If `save_residuals` is True, `_splash_attention_fwd` returns `(out, stats)`,
  # so we unpack the gradients, otherwise it returns `out` and `grads` is just
  # `do`.
  if save_residuals:
    do, _ = grads
  else:
    do = grads
  del save_residuals, fwd_mask_sparsity
  if not config.has_backward_blocks:
    raise ValueError("Need to specify backward blocks.")
  bq_dkv, bkv_dkv_memory, bkv_dkv_compute = (
      config.block_q_dkv,
      config.block_kv_dkv,
      config.block_kv_dkv_compute,
  )
  q, k, v, segment_ids, sinks, o, logsumexp, dkv_mask_info = res

  # di: [num_heads, q_seq_len]
  di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))  # pytype: disable=attribute-error
  dq, dk, dv = _splash_attention_bwd_dkv(
      q,
      k,
      v,
      segment_ids,
      logsumexp,
      do,
      di,
      bq=bq_dkv,  # pyrefly: ignore[bad-argument-type]
      bkv=bkv_dkv_memory,  # pyrefly: ignore[bad-argument-type]
      bkv_compute=bkv_dkv_compute,  # pyrefly: ignore[bad-argument-type]
      is_mqa=is_mqa,
      mask_info=dkv_mask_info,  # pyrefly: ignore[bad-argument-type]
      mask_value=mask_value,
      mask_function=mask_function,
      config=config,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )
  dsinks = None
  if sinks is not None:
    logsumexp_ = (logsumexp / LOG2E) if config.use_base2_exp else logsumexp
    sinks_exp = -jnp.exp(
        sinks[..., None, None].astype(jnp.float32)
        - logsumexp_[..., None].astype(jnp.float32)
    )
    dsinks = jnp.sum(sinks_exp.astype(o.dtype) * o * do, axis=(-1, -2))  # pyrefly: ignore[bad-argument-type]
  # Match the signature of the fwd function.
  assert dq is not None
  return (
      None,  # fwd_mask_info
      None,  # dvk_mak_info
      dq,  # q
      dk,  # k
      dv,  # v
      None,  # segment_ids
      dsinks,  # sinks
      None,  # max_logit_estimate
  )


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "dkv_mask_sparsity",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig | None,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    dkv_mask_sparsity: float,
) -> SplashCustomReturnType:
  return _splash_attention_custom(
      fwd_mask_info,
      dkv_mask_info,
      q,
      k,
      v,
      segment_ids,
      sinks,
      mask_value=mask_value,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      config=config,
      max_logit_value=max_logit_value,
      mask_function=mask_function,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:

  def __init__(
      self,
      fwd_mask_info: MaskInfo,
      dkv_mask_info: MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dkv_mask_info,
        *args,
        **dict(self.kwargs, **kwargs),
    )

  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.block_mask is not None:
      block_mask_shape = self.fwd_mask_info.block_mask.shape
      try:
        sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError(
            "The sharding must divide the mask blocks evenly between devices"
        ) from exc

    if len(sharding.spec) != 1:
      raise ValueError("Only q sequence sharding is supported.")

    _resolve_spec = lambda x: sharding.spec if x is not None else None
    mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
        mask_next=_resolve_spec(self.fwd_mask_info.mask_next),  # pyrefly: ignore[bad-argument-type]
        active_rows=_resolve_spec(self.fwd_mask_info.active_rows),  # pyrefly: ignore[bad-argument-type]
        active_cols=_resolve_spec(self.fwd_mask_info.active_cols),  # pyrefly: ignore[bad-argument-type]
        num_active_blocks=_resolve_spec(self.fwd_mask_info.num_active_blocks),  # pyrefly: ignore[bad-argument-type]
        block_mask=_resolve_spec(self.fwd_mask_info.block_mask),  # pyrefly: ignore[bad-argument-type]
        partial_mask_blocks=jax.sharding.PartitionSpec()  # replicated  # pyrefly: ignore[bad-argument-type]
        if self.fwd_mask_info.partial_mask_blocks is not None
        else None,
        q_sequence=_resolve_spec(self.fwd_mask_info.q_sequence),  # pyrefly: ignore[bad-argument-type]
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return ((self.fwd_mask_info, self.dkv_mask_info), self.kwargs)

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dkv_mask_info = (
        MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    )
    return SplashAttentionKernel(
        MaskInfo(*fwd_mask_info), dkv_mask_info, **kwargs
    )


def _make_splash_attention(
    mask: np.ndarray | Mask,
    *,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
):
  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if isinstance(mask, np.ndarray):
    mask = NumpyMask(mask)

  if config is None:
    config = SplashConfig.get_default()

  if config.qk_diag_skip and not isinstance(mask, CausalMask):
    # The skip assumes kv > q is ALWAYS masked — a pure-causal property. Any mask
    # that admits a valid kv > q entry (bidirectional, local/sliding window, custom)
    # would be silently corrupted, so fail loud. (Square-block preconditions are
    # enforced in SplashConfig.__post_init__.)
    raise ValueError(
        "qk_diag_skip=True requires a pure CausalMask (the skip fills mask_value "
        "for all kv > q sub-tiles, assuming the mask masks exactly those); got "
        f"{type(mask).__name__}. Disable qk_diag_skip for non-causal masks."
    )

  process_fn = partial(
      process_mask,
      downcast_smem_data=downcast_smem_data,
      partial_mask_blocks_dtype=partial_mask_blocks_dtype,
      q_seq_shards=q_seq_shards,
  )

  fwd_mask_info, mask_function_fwd = process_fn(
      mask,
      (config.block_q, config.block_kv),
  )
  fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != 0))
  fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

  dkv_mask_info = None
  if config.has_backward_blocks:
    bq_dkv, bkv_dkv = config.block_q_dkv, config.block_kv_dkv
    dkv_mask_info, mask_function_dkv = process_fn(
        mask,
        (bq_dkv, bkv_dkv),
        is_dkv=True,
        return_dynamic_grid=config.dq_reduction_steps == 3,
    )

    assert (mask_function_fwd is None) == (mask_function_dkv is None)

    dkv_mask_sparsity = float(np.mean(dkv_mask_info.block_mask != 0))
    dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)
  else:
    dkv_mask_sparsity = 1.0

  return SplashAttentionKernel(
      fwd_mask_info,
      dkv_mask_info,
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=mask_function_fwd,
      fwd_mask_sparsity=fwd_mask_sparsity,
      dkv_mask_sparsity=dkv_mask_sparsity,
  )


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
):
  if (mesh is not None) != (mask_spec is not None):
    raise ValueError(
        "Either both or neither of mesh and mask_spec must be specified."
    )

  if mask_spec is not None and len(mask_spec) != 1:
    raise ValueError("Only shard over the query sequence dimension.")

  if len(mask.shape) != 2:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

  if config is None:
    config = SplashConfig.get_default()

  # This is the only mode that supports the dynamic grid.
  config = dataclasses.replace(config, dq_reduction_steps=3)

  def process_mask_shard(mask):
    process_mask_fn = functools.partial(
        _process_dynamic_mask,
        downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,
    )

    fwd_mask_info = process_mask_fn(
        mask, (config.block_q, config.block_kv), is_dkv=False
    )

    dkv_mask_info = None
    if config.has_backward_blocks:
      dkv_mask_info = process_mask_fn(
          mask, (config.block_q_dkv, config.block_kv_dkv), is_dkv=True
      )

    return fwd_mask_info, dkv_mask_info

  kwargs = dict(
      config=config,
      is_mqa=is_mqa,
      save_residuals=save_residuals,
      mask_value=mask_value,
      mask_function=None,
      fwd_mask_sparsity=1.0,
      dkv_mask_sparsity=1.0,
  )

  # If the input mask is replicated we don't need to call shard_map.
  if mask_spec is None:
    fwd_mask_info, dkv_mask_info = process_mask_shard(mask)
    kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
    return kernel

  mask_info_specs = MaskInfo(  # pytype: disable=wrong-arg-types
      mask_next=mask_spec,  # pyrefly: ignore[bad-argument-type]
      active_rows=None,
      active_cols=None,
      num_active_blocks=None,
      block_mask=mask_spec,  # pyrefly: ignore[bad-argument-type]
      partial_mask_blocks=mask_spec,  # pyrefly: ignore[bad-argument-type]
      q_sequence=None,
  )
  out_specs = (
      mask_info_specs,
      mask_info_specs if config.has_backward_blocks else None,
  )

  @partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=mask_spec,
      out_specs=out_specs,
      check_vma=False,
  )
  def process_all_shards(mask):
    return process_mask_shard(mask)

  fwd_mask_info, dkv_mask_info = process_all_shards(mask)
  kernel = SplashAttentionKernel(fwd_mask_info, dkv_mask_info, **kwargs)
  kernel_spec = SplashAttentionKernel(*out_specs, **kwargs)

  return (kernel, kernel_spec)


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, q_seq_shards=1)

make_splash_mqa_single_device = partial(make_splash_mqa, q_seq_shards=1)

make_dynamic_splash_mqa = partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = partial(_make_dynamic_splash_attention, is_mqa=False)



# Corpus protocol: the default callable builds a causal, single-device Splash
# kernel.  Keeping mask construction outside the jitted call matches Tokamax.
def build_kernel(
    sequence: int = 256,
    *,
    interpret: bool = False,
    block_q: int = 128,
    block_kv: int = 128,
    block_kv_compute: int | None = None,
):
  mask = CausalMask(shape=(sequence, sequence))
  config = dataclasses.replace(
      SplashConfig.get_default(),
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute or block_kv,
      interpret=interpret,
  )
  return make_splash_mha_single_device(mask, config=config)


kernel = build_kernel


def _main() -> None:
  import argparse
  import time

  parser = argparse.ArgumentParser()
  parser.add_argument("--sequence", type=int, default=256)
  parser.add_argument("--heads", type=int, default=1)
  parser.add_argument("--head-dim", type=int, default=128)
  parser.add_argument("--interpret", action="store_true")
  args = parser.parse_args()

  keys = jax.random.split(jax.random.key(42), 3)
  shape = (args.heads, args.sequence, args.head_dim)
  q, k, v = (
      jax.random.normal(key, shape, dtype=jnp.bfloat16) for key in keys
  )
  attention = build_kernel(args.sequence, interpret=args.interpret)
  compiled = jax.jit(attention)
  start = time.perf_counter()
  output = compiled(q, k, v)
  output.block_until_ready()
  elapsed_ms = (time.perf_counter() - start) * 1e3
  print(json.dumps({
      "implementation": IMPLEMENTATION,
      "contract": SOURCE["contract"],
      "shape": list(output.shape),
      "compile_and_run_ms": elapsed_ms,
  }))


if __name__ == "__main__":
  _main()
