"""Standalone PallasBench mse_loss kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/mse_loss.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_mse_loss`` (also exported as ``kernel``).

Contract ``mse_loss``, in the ``similarity_loss`` family.  The reference is
upstream's own ``jax_mse_loss`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("mse_loss")`` there
reproduces the dtypes and value ranges upstream's benchmark harness uses, so a
comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[2048, 1024], [2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/mse_loss.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "mse_loss",
    "family": "similarity_loss",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[2048, 1024], [2048, 1024]],
    "validation_shape": [[2048, 1024], [2048, 1024]],
    "validation_reason": None,
}

"""Level 1: Mean squared error loss via Pallas.

Provenance: standard regression loss, (pred - target)^2 reduced per row
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _mse_kernel(pred_ref, target_ref, o_ref):
    diff = pred_ref[...] - target_ref[...]
    o_ref[...] = jnp.mean(diff * diff, axis=-1)


def pallas_mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    n_rows = pred.shape[0]
    n_cols = pred.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _mse_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), pred.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(pred, target)


pallas_kernel = pallas_mse_loss
task_name = "mse_loss"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss"
level = 1


kernel = pallas_mse_loss
