"""Standalone PallasBench layernorm kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/layernorm.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_layernorm`` (also exported as ``kernel``).

Contract ``layernorm``, in the ``layer_norm`` family.  The reference is
upstream's own ``jax_layernorm`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("layernorm")`` there
reproduces the dtypes and value ranges upstream's benchmark harness uses, so a
comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/layernorm.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "layernorm",
    "family": "layer_norm",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[2048, 1024]],
    "validation_shape": [[2048, 1024]],
    "validation_reason": None,
}

"""Level 1: Layer normalization via Pallas.

Demonstrates: mean/variance reduction, epsilon stability, row-parallel tiling.
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _layernorm_kernel(x_ref, o_ref):
    x = x_ref[...]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    o_ref[...] = (x - mean) / jnp.sqrt(var + 1e-5)


def pallas_layernorm(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _layernorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_layernorm
task_name = "layernorm"
input_shapes = [(2048, 1024)]
category = "normalization"
level = 1


kernel = pallas_layernorm
