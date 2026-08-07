"""Standalone PallasBench matmul_silu kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/matmul_silu.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_matmul_silu`` (also exported as ``kernel``).

Contract ``matmul_silu``, in the ``matmul_activation`` family.  The reference
is upstream's own ``jax_matmul_silu`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("matmul_silu")`` there reproduces the dtypes and
value ranges upstream's benchmark harness uses, so a comparison here is against
upstream's definition of the task rather than a corpus reading of it.

Native shape: [[1024, 1024], [1024, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/matmul_silu.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "matmul_silu",
    "family": "matmul_activation",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[1024, 1024], [1024, 1024]],
    "validation_shape": [[1024, 1024], [1024, 1024]],
    "validation_reason": None,
}

"""Level 2: Fused MatMul + SiLU via Pallas.

Provenance: openxla/tokamax gated_linear_unit uses SiLU gate path
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _matmul_silu_kernel(x_ref, w_ref, o_ref):
    z = x_ref[...] @ w_ref[...]
    o_ref[...] = z / (1.0 + jnp.exp(-z))


def pallas_matmul_silu(x: jax.Array, w: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)
    BLOCK_N = min(n, 128)
    grid = (m // BLOCK_M, n // BLOCK_N)

    return pl.pallas_call(
        _matmul_silu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i, j: (i, 0)),
            pl.BlockSpec((k, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, w)


pallas_kernel = pallas_matmul_silu
task_name = "matmul_silu"
input_shapes = [(1024, 1024), (1024, 1024)]
category = "matmul_activation"
level = 2


kernel = pallas_matmul_silu
