"""Standalone PallasBench matmul_relu kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/matmul_relu.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_matmul_relu`` (also exported as ``kernel``).

Contract ``matmul_relu``, in the ``matmul_activation`` family.  The reference
is upstream's own ``jax_matmul_relu`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("matmul_relu")`` there reproduces the dtypes and
value ranges upstream's benchmark harness uses, so a comparison here is against
upstream's definition of the task rather than a corpus reading of it.

Native shape: [[1024, 1024], [1024, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/matmul_relu.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "matmul_relu",
    "family": "matmul_activation",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[1024, 1024], [1024, 1024]],
    "validation_shape": [[1024, 1024], [1024, 1024]],
    "validation_reason": None,
}

"""Level 2: Fused MatMul + ReLU via Pallas.

Demonstrates: operator fusion — single pallas_call replaces matmul + relu,
avoiding a round-trip through HBM between the two ops.
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_relu_kernel(x_ref, w_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...] @ w_ref[...], 0)


def pallas_matmul_relu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_relu
task_name = "matmul_relu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2


kernel = pallas_matmul_relu
