"""Standalone PallasBench rmsnorm_residual kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/rmsnorm_residual.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_rmsnorm_residual`` (also exported as ``kernel``).

Contract ``rmsnorm_residual``, in the ``norm_residual`` family.  The reference
is upstream's own ``jax_rmsnorm_residual`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("rmsnorm_residual")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[2048, 1024], [2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/rmsnorm_residual.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "rmsnorm_residual",
    "family": "norm_residual",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[2048, 1024], [2048, 1024]],
    "validation_shape": [[2048, 1024], [2048, 1024]],
    "validation_reason": None,
}

"""Level 2: Fused RMSNorm + Residual Add via Pallas.

Demonstrates: two-input fusion, norm + elementwise add in one kernel.
Inspired by pallas-forge's 3.44x speedup over XLA for this pattern.
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _rmsnorm_residual_kernel(x_ref, residual_ref, o_ref):
    x = x_ref[...]
    residual = residual_ref[...]
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(ms + 1e-5)
    o_ref[...] = normed + residual


def pallas_rmsnorm_residual(
    x: jax.Array, residual: jax.Array
) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _rmsnorm_residual_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x, residual)


pallas_kernel = pallas_rmsnorm_residual
task_name = "rmsnorm_residual"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "norm_residual"
level = 2


kernel = pallas_rmsnorm_residual
