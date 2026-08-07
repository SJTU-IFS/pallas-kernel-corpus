"""Standalone PallasBench matmul kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/matmul.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_matmul`` (also exported as ``kernel``).

Contract ``matmul``, in the ``dense_matmul`` family.  The reference is
upstream's own ``jax_matmul`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("matmul")`` there
reproduces the dtypes and value ranges upstream's benchmark harness uses, so a
comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[1024, 1024], [1024, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/matmul.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "matmul",
    "family": "dense_matmul",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[1024, 1024], [1024, 1024]],
    "validation_shape": [[1024, 1024], [1024, 1024]],
    "validation_reason": None,
}

"""Level 1: Tiled matrix multiplication via Pallas.

Demonstrates: 2D grid, BlockSpec with K-dimension accumulation,
multi-block tiling pattern from the Pallas quickstart.
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]


def pallas_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = y.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, y)


pallas_kernel = pallas_matmul
task_name = "matmul"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul"
level = 1


kernel = pallas_matmul
