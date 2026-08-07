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

"""Standalone vLLM tpu-inference layout-transpose kernels.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/mla/v2/transpose.py
  also inlines: tpu_inference/kernels/ragged_paged_attention/v3/util.py  (only: get_dtype_bitwidth, get_dtype_packing)
  transformation: three imports were resolved and no kernel body was touched.
    `get_dtype_packing` and the helper it calls were inlined from the ragged
    paged attention util module; the vLLM logger became the stdlib one; and
    `sympy.divisors`, used only for host-side tile selection, was replaced by
    the trial-division equivalent shared with tools/flatten_mla.py, because
    sympy is not in the pinned dependency set.

Entry points: ``xpose_full``, ``xpose_pipeline``, ``pin_vmem_custom_call``
(three separate Pallas launches), and ``kernel`` (alias of ``xpose_pipeline``,
the one MLA v2 actually calls).  Each returns a **list**, not an array.

Contract ``layout_transpose``.  All three move data without computing on it, so
the reference is `jnp.transpose` -- which is also what upstream's own
`tests/kernels/transpose_test.py` compares against, and is exact rather than
approximate:

    xpose_full(x, transpose_axes=axes)[0]      == jnp.transpose(x, axes)
    xpose_pipeline(x, transpose_axes=axes)[0]  == jnp.transpose(x, axes)
    pin_vmem_custom_call(x)[0]                 == x

``xpose_full`` maps the whole array into VMEM at once, so it is limited by VMEM;
``xpose_pipeline`` tiles the parallel and pipeline axes and double-buffers, and
is the one to use when the array does not fit.  Its tile sizes are requests, not
commands: `prev_closest_valid_divisor` lowers `n_tile` to the largest divisor of
the axis that is also a multiple of the dtype's sublane packing, warns when it
has to, and **raises** when no such divisor exists rather than quietly using a
non-divisor tile that would leave rows unprocessed.

``pin_vmem_custom_call`` computes nothing at all -- `identity_fn_generator`
copies each input buffer to the matching output. The point is the side effect:
the result is a VMEM-resident buffer, so a later consumer does not pay to fetch
it. That makes it the one kernel here a correctness test cannot really check
beyond bitwise identity, and the ledger says so.

Native shape: none declared. Upstream's tests parameterise 2-D through 4-D,
mostly float8_e4m3fn, including a 128x2048x256 case chosen to be too large for
`xpose_full`.
"""

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/mla/v2/transpose.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "layout_transpose",
    "family": "layout_transpose",
    "launch_points": 3,
    "native_shape": None,
}

import logging

from collections.abc import Sequence

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# ---- inlined from tpu_inference/kernels/ragged_paged_attention/v3/util.py: get_dtype_bitwidth, get_dtype_packing ----

from jax._src import dtypes


def get_dtype_bitwidth(dtype):
    return dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


# ``sympy.divisors`` upstream; see SYMPY_DIVISORS_REPLACEMENT in
# tools/flatten_mla.py for why it is inlined rather than imported.
def divisors(n: int) -> list[int]:
    """Positive divisors of ``n``, ascending."""
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]
@jax.jit(static_argnames=[
    'transpose_axes',
])
def xpose_full(input, *, transpose_axes):

    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)

    input_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
    output_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)]
    transposed_shape = tuple(input.shape[i] for i in transpose_axes)

    output_shape = [
        jax.ShapeDtypeStruct(shape=transposed_shape, dtype=input.dtype)
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_full_shape_{shape_str}_xpose_{transpose_str}"
    return pl.pallas_call(xpose_kernel,
                          in_specs=input_specs,
                          out_specs=output_specs,
                          out_shape=output_shape,
                          name=scope_name)(input)


def prev_closest_valid_divisor(number: int,
                               divider: int,
                               multiple_of: int = 1) -> int:
    """
    Finds the largest divisor of 'number' that is <= 'divider' and divisible
    by 'multiple_of'.

    Raises ValueError if no divisor of 'number' satisfies both constraints.
    The exception is if min(number, divider) < multiple_of and number <= divider: return
      'number' itself.  
    This is because Pallas accepts a sublane tile equal to the full array
    dimension.
    """
    if divider < 1:
        return 1

    bound = min(number, divider)
    if bound < multiple_of:
        if divider < number:
            raise ValueError(
                f"divider={divider} < number={number} and both are < "
                f"multiple_of={multiple_of}: no valid tile size exists.")
        # number <= divider and number < multiple_of: tile equals full dim.
        return number

    all_divisors = divisors(number)
    valid = [d for d in all_divisors if d <= divider and d % multiple_of == 0]
    if valid:
        return valid[-1]

    raise ValueError(
        f"No divisor of {number} is both <= {divider} and divisible by "
        f"{multiple_of}. A non-divisor tile would produce incorrect results.")


def get_reshape_dimension(shape, reshape_axes, dtype=jnp.float32):
    input_shape_struct = jax.ShapeDtypeStruct(shape, dtype)

    def _reshape(inp):
        return inp.reshape(*reshape_axes)

    return jax.eval_shape(_reshape, input_shape_struct).shape


def identity_fn_generator(num_scalars: int = 0):
    """Method to copy input content into outputs."""

    def identity(*arg):
        n = len(arg)
        d = n // 2  # first half of args are inputs; second half are outputs
        for i in range(d):
            # Copy over kernel scalars directly into output
            if i < num_scalars:
                if arg[i].ndim == 0:
                    arg[i + d].set(arg[i].get())
                else:  # ndim == 1
                    for j in range(arg[i].shape[0]):
                        arg[i + d][j] = arg[i][j]
            # Write the input VMEM contents into the output VMEM buffer.
            else:
                arg[i + d][...] = arg[i][...]

    return identity


@jax.jit(static_argnames=['num_scalars'])
def pin_vmem_custom_call(input_tensor: jax.Array, num_scalars: int = 0):
    """Prefetches buffers to VMEM."""
    return jax.named_scope("prefetch")(pl.pallas_call(
        identity_fn_generator(num_scalars),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(input_tensor.shape, input_tensor.dtype),
        ],
        name="prefetch",
    ))(input_tensor)


@jax.jit(static_argnames=[
    'transpose_axes', 'n_tile', 'm_tile', 'parallel_axis', 'pipeline_axis',
    'vmem_limit_bytes'
])
def xpose_pipeline(input: jax.Array,
                   *,
                   transpose_axes: Sequence[int],
                   n_tile: int = 128,
                   m_tile: int = 128,
                   parallel_axis: int = 0,
                   pipeline_axis: int = 1,
                   vmem_limit_bytes: int | None = None):
    """
    Double buffer transpose custom call implementation.
    n_tile is used to tile the parallel dimension while m_tile is used to tile the pipeline dimension.
    Args:
      input: input array to be transposed
      tranpose_axes: transpose ordering
      n_tile: tile amount for the parallelizable axis
      m_tile: tile amount for the pipelined axis
      parallel_axis: index of the parallel axis
      pipeline_axis: index of the pipeline axis
      vmem_limit_bytes: the vmem limit for the pallas kernel. With the default
        (None), the kernel is subject to XLA's global scoped-vmem limit
        (--xla_tpu_scoped_vmem_limit_kib, 32 MiB by default), which large
        tile shapes can exceed at compile time.
    """

    def xpose_kernel(input_ref, output_ref):
        output_ref[...] = input_ref[...].transpose(*transpose_axes)

    n_tile = n_tile if n_tile <= input.shape[parallel_axis] else input.shape[
        parallel_axis]
    m_tile = m_tile if m_tile <= input.shape[pipeline_axis] else input.shape[
        pipeline_axis]
    # Find the best tile that (a) divides the axis inclusively
    # and (b) satisfies Pallas's sublane alignment
    # requirement: block dims must be divisible by
    # get_dtype_packing(dtype) * 8. If no such tiling exists,
    # then throw a ValueError
    sublane_multiple = get_dtype_packing(input.dtype) * 8
    n_tile_new = prev_closest_valid_divisor(input.shape[parallel_axis],
                                            n_tile,
                                            multiple_of=sublane_multiple)
    if input.shape[parallel_axis] % n_tile_new != 0:
        raise ValueError(
            f"No divisor of parallel axis size {input.shape[parallel_axis]} "
            f"is both <= {n_tile} and divisible by {sublane_multiple} "
            f"(dtype={input.dtype}). Consider increasing n_tile and/or padding your input to be "
            f"suble-aligned (i.e. a multiple of {sublane_multiple}).")
    m_tile_new = prev_closest_valid_divisor(input.shape[pipeline_axis],
                                            m_tile,
                                            multiple_of=sublane_multiple)
    if input.shape[pipeline_axis] % m_tile_new != 0:
        raise ValueError(
            f"No divisor of pipeline axis size {input.shape[pipeline_axis]} "
            f"is both <= {m_tile} and divisible by {sublane_multiple} "
            f"(dtype={input.dtype}). Consider increasing n_tile and/or padding your input to be "
            f"suble-aligned (i.e. a multiple of {sublane_multiple}).")
    if n_tile_new != n_tile:
        logger.warning(
            f"Adjusting n_tile={n_tile} to new valid tiling={n_tile_new} "
            f"which is <= n_tile={n_tile} and sublane-aligned (i.e a multiple of "
            f"{sublane_multiple}).")
    if m_tile_new != m_tile:
        logger.warning(
            f"Adjusting m_tile={m_tile} to new valid tiling={m_tile_new} "
            f"which is <= m_tile={m_tile} and sublane-aligned (i.e a multiple of "
            f"{sublane_multiple}).")
    n_tile, m_tile = n_tile_new, m_tile_new
    grid = (input.shape[parallel_axis] // n_tile,
            input.shape[pipeline_axis] // m_tile)

    # Define the input and ouptut shapes and block shapes.
    full_block_shape = list(input.shape)
    full_block_shape[parallel_axis] = n_tile
    full_block_shape[pipeline_axis] = m_tile
    full_block_shape = tuple(full_block_shape)
    transposed_block_shape = tuple(full_block_shape[i] for i in transpose_axes)
    transposed_input_shape = tuple(input.shape[i] for i in transpose_axes)
    output_shape = transposed_input_shape

    # The transposition settings will influence the input and ouptut index maps.
    def get_grid_index(i: int, j: int, input_grid: bool):
        grid_idx = [
            0,
        ] * input.ndim
        if input_grid:
            grid_idx[parallel_axis] = i
            grid_idx[pipeline_axis] = j
        else:
            grid_idx[pipeline_axis] = i
            grid_idx[parallel_axis] = j
        return grid_idx

    out_index_map = lambda i, j: get_grid_index(  # noqa: E731
        i, j, input_grid=False)

    input_specs = [
        pl.BlockSpec(
            index_map=lambda i, j: get_grid_index(i, j, input_grid=True),
            block_shape=full_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    output_specs = [
        pl.BlockSpec(
            index_map=out_index_map,
            block_shape=transposed_block_shape,
            memory_space=pltpu.VMEM,
        )
    ]
    shape_str = "x".join([str(i) for i in input.shape])
    transpose_str = "x".join([str(i) for i in transpose_axes])
    scope_name = f"xpose_pipeline_shape_{shape_str}_xpose_{transpose_str}_n_tile_{n_tile}_m_tile_{m_tile}_pa_{parallel_axis}_pi_{pipeline_axis}"
    return pl.pallas_call(xpose_kernel,
                          grid=grid,
                          compiler_params=pltpu.CompilerParams(
                              dimension_semantics=("parallel", "arbitrary"),
                              vmem_limit_bytes=vmem_limit_bytes),
                          in_specs=input_specs,
                          out_specs=output_specs,
                          out_shape=[
                              jax.ShapeDtypeStruct(shape=output_shape,
                                                   dtype=input.dtype)
                          ],
                          name=scope_name)(input)

# The launch MLA v2 calls, and the only one of the three that works on arrays
# too large for VMEM.
kernel = xpose_pipeline
