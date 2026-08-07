"""Standalone PallasBench cosine_sim kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/cosine_sim.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_cosine_sim`` (also exported as ``kernel``).

Contract ``cosine_sim``, in the ``similarity_loss`` family.  The reference is
upstream's own ``jax_cosine_sim`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("cosine_sim")``
there reproduces the dtypes and value ranges upstream's benchmark harness uses,
so a comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[2048, 1024], [2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/cosine_sim.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "cosine_sim",
    "family": "similarity_loss",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[2048, 1024], [2048, 1024]],
    "validation_shape": [[2048, 1024], [2048, 1024]],
    "validation_reason": None,
}

"""Level 1: Row-wise cosine similarity via Pallas.

Provenance: standard similarity metric for embeddings and retrieval
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _cosine_sim_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    dot = jnp.sum(x * y, axis=-1)
    norm_x = jnp.sqrt(jnp.sum(x * x, axis=-1))
    norm_y = jnp.sqrt(jnp.sum(y * y, axis=-1))
    o_ref[...] = dot / (norm_x * norm_y + 1e-8)


def pallas_cosine_sim(x: jax.Array, y: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _cosine_sim_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(x, y)


pallas_kernel = pallas_cosine_sim
task_name = "cosine_sim"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss"
level = 1


kernel = pallas_cosine_sim
