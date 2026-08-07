"""Standalone vLLM tpu-inference unit-lower-triangular inverse kernels.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tpu_inference/kernels/gdn/triangle_solver.py
  transformation: copied as a standalone implementation with source metadata
    added; nothing else changed. The file imports nothing outside jax.

**Two Pallas launch points**, two algorithms for one contract:

- ``newton_schulz_inverse_pallas`` -- Newton-Schulz iteration;
- ``decompose_triangular_matrix_inverse_pallas`` -- blockwise decomposition.

``kernel`` is bound to the blockwise one, which is what the gated delta net
chunked scan calls.

Contract ``unit_lower_triangular_inverse``: given a **unit** lower triangular ``A`` (ones on the
diagonal), return ``A^-1``.

Unlike everything else in this family the contract is a plain mathematical
identity, so the corpus checks it against ``jnp.linalg.inv`` -- an independent
oracle -- as well as against upstream's own references, which live in this same
file (``newton_schulz_inverse_ref`` and ``local_forward_substitution``, both
pure JAX). Everywhere else in gated_delta_net the only available reference is
the one upstream wrote, so an error shared between kernel and reference would
go unnoticed; here it would not.

These are helpers for the chunked scan rather than a model op in their own
right: the chunked algorithm needs the inverse of a triangular matrix built
from the per-chunk decay terms.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tpu_inference/kernels/gdn/triangle_solver.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "unit_lower_triangular_inverse",
    "launch_points": 2,
    "entry_points": (
        "newton_schulz_inverse_pallas",
        "decompose_triangular_matrix_inverse_pallas",
    ),
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

import enum
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# Implementation of inverse of triangle based on Newton-Schulz iteration.
def newton_schulz_inverse_ref(A, n=None):
    """Inverse of unit lower triangular matrix using Newton-Schulz iteration.

  Args:
    A: Tensor with last two dimensions representing a square lower triangular
      matrix with unit diagonal.
    n: Number of iterations to run.

  Newton Schulz iteration:
  https://en.wikipedia.org/wiki/Matrix_sign_function#Newton%E2%80%93Schulz_iteration
  S_{k+1} = S_k @ (2 * I - A @ S_k)

  Let L = A - I
  Starting with S_0 = I, this is equivalent mathematically to
  S_k = (I - L) @ (I + L^2) @ (I + L^4)....(I + (L^(2^k))), k > 0

  If L is strictly lower (or upper) triangular, L ^ n == 0.
  So this series converges after log(n) steps.

  We don't directly compute S_k as above to reduce precision loss.
  We run the last step in higher precision to improve the overall estimate.
  Initial steps are kept in lower precision for speed.

  Returns:
    Inverse of A.
  """
    if n is None:
        n = A.shape[-1]
    eye = jnp.broadcast_to(jnp.eye(n, dtype=A.dtype), A.shape)
    S = 2 * eye - A
    k = 1
    while k < n:
        precision = jax.lax.Precision.HIGHEST
        k *= 2
        I_plus_error = 2 * eye - jnp.matmul(A, S, precision=precision)
        S = jnp.matmul(S, I_plus_error, precision=precision)
    return S


# Pallas implementation of Newton-Schulz iteration for unit lower triangular
# matrices.
def newton_schulz_inverse_pallas_kernel(A_ref, x_ref):
    x_ref[...] = newton_schulz_inverse_ref(A_ref[...])


def newton_schulz_inverse_pallas(A, *, block_size=64):
    """Newton-Schulz iteration for unit lower triangular matrices on Pallas."""

    A_shape = A.shape
    # Squash all the leading dimensions
    A = A.reshape(-1, *A.shape[-2:])
    N = A.shape[0]
    grid_size = pl.cdiv(N, block_size)
    x = pl.pallas_call(
        newton_schulz_inverse_pallas_kernel,
        out_shape=jax.ShapeDtypeStruct(A.shape, A.dtype),
        grid=(grid_size, ),
        in_specs=[
            pl.BlockSpec((block_size, A.shape[-2], A.shape[-1]), lambda idx:
                         (idx, 0, 0)),
        ],
        out_specs=pl.BlockSpec((block_size, A.shape[-2], A.shape[-1]),
                               lambda idx: (idx, 0, 0)),
        name="newton_schulz_inverse_kernel",
    )(A)
    return x.reshape(A_shape)


def local_forward_substitution(A, b):
    """Solves A X = B for unit lower triangular matrix A using forward substitution.

    Args:
      A: A tensor of shape (B, N, N) representing a batch of unit lower triangular
        matrices.
      b: A tensor of shape (B, N, K) representing the right-hand side.

    Returns:
      A tensor of shape (B, N, K) representing the solution X.
    """
    B, N, K = b.shape
    x_list = []
    for i in range(N):
        b_i = b[:, i, :]
        if i == 0:
            x_i = b_i
        else:
            stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
            all_prev_A = A[:, i, :i]  # (B, i)
            prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x,
                               axis=1)  # (B, K)
            x_i = b_i - prev_sum  # (B, K) for the row i
        x_list.append(x_i)
    x = jnp.stack(x_list, axis=1)  # (B, N, K)
    return x


def decompose_triangular_matrix_inverse_pallas_kernel(A_ref,
                                                      x_ref,
                                                      *,
                                                      block_size=16):
    A = A_ref[...]
    # Matrix dimension
    B, N, _ = A.shape
    num_blocks = N // block_size

    # same as lower_triangle_solver_pallas_kernel but 2d block wise
    # AX = I, solve for X block wise. X = I - sum(AX_prev)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))
        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = x_ref[:, :start, :]
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_substitution(local_A, target_b)
        x_ref[..., start:end, :] = x_block


def decompose_triangular_matrix_inverse_pallas(A,
                                               *,
                                               n_block_size=64,
                                               block_size=16):
    """Inverts unit lower triangular matrices using a block-wise approach in Pallas.

    This function solves A X = I for X, where A is a unit lower triangular matrix.
    It uses a block-wise Gaussian elimination approach to improve performance.

    Args:
      A: A tensor of shape (batch_size, chunks, heads, head_dim, head_dim) where
        the last two dimensions represent unit lower triangular matrices.
      n_block_size: The block size for Pallas grid execution.
      block_size: The block size for the block-wise inversion algorithm.

    Returns:
      A tensor of the same shape as A, representing the inverse of A.
    """

    # Squash all the leading dimensions
    A_reshaped = A.reshape(-1, *A.shape[-2:])
    A_shape = A_reshaped.shape
    x_shape = A_shape

    N = A_reshaped.shape[0]
    grid_size = pl.cdiv(N, n_block_size)

    head_dim = A_shape[-1]
    kernel = functools.partial(
        decompose_triangular_matrix_inverse_pallas_kernel,
        block_size=block_size)
    x = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x_shape, A.dtype),
        grid=(grid_size, ),
        in_specs=[
            pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx:
                         (idx, 0, 0)),
        ],
        out_specs=pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx:
                               (idx, 0, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=67108864),
        name=
        f"decompose_triangular_matrix_inverse_pallas_kernel_{n_block_size}_{block_size}",
    )(A_reshaped)

    return x.reshape(A.shape)


class TriangleSolverImpl(str, enum.Enum):
    GAUSSIAN = "gaussian"
    NEWTON_SCHULZ = "newton_schulz"

    #TODO: Choose based on Chunk size and vmem constraints. Newton-schulz is unsatable, it needs S to be nilpotent to converge and also with small values to avoid NaNs
    def __call__(self, A):
        if self == TriangleSolverImpl.GAUSSIAN:
            return decompose_triangular_matrix_inverse_pallas(A,
                                                              n_block_size=min(
                                                                  64,
                                                                  A.shape[-1]))
        elif self == TriangleSolverImpl.NEWTON_SCHULZ:
            return newton_schulz_inverse_pallas(A)
        else:
            print(f"Unknown solver: {self.value} Using default solver."
                  f" {TriangleSolverImpl.GAUSSIAN.value}")
            return decompose_triangular_matrix_inverse_pallas(A,
                                                              n_block_size=min(
                                                                  64,
                                                                  A.shape[-1]))

kernel = decompose_triangular_matrix_inverse_pallas
