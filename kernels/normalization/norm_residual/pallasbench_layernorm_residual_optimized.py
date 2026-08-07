"""Standalone PallasBench layernorm_residual kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/layernorm_residual.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_layernorm_residual`` (also exported as ``kernel``).

Contract ``layernorm_residual``, in the ``norm_residual`` family.  The
reference is upstream's own ``jax_layernorm_residual`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("layernorm_residual")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[2048, 1024], [2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/layernorm_residual.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "layernorm_residual",
    "family": "norm_residual",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[2048, 1024], [2048, 1024]],
    "validation_shape": [[2048, 1024], [2048, 1024]],
    "validation_reason": None,
}

"""Level 2: Fused LayerNorm + Residual Add via Pallas.

Provenance: standard transformer pre-norm pattern, MaxText attention blocks
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _layernorm_residual_kernel(x_ref, residual_ref, o_ref):
    x = x_ref[...]
    residual = residual_ref[...]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + 1e-5)
    o_ref[...] = normed + residual


def pallas_layernorm_residual(x: jax.Array, residual: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _layernorm_residual_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x, residual)


pallas_kernel = pallas_layernorm_residual
task_name = "layernorm_residual"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "norm_residual"
level = 2


kernel = pallas_layernorm_residual
