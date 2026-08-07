"""Standalone PallasBench outer_product kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/outer_product.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_outer_product`` (also exported as ``kernel``).

Contract ``outer_product``, in the ``dense_matmul`` family.  The reference is
upstream's own ``jax_outer_product`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("outer_product")`` there reproduces the dtypes
and value ranges upstream's benchmark harness uses, so a comparison here is
against upstream's definition of the task rather than a corpus reading of it.

Native shape: [[1024], [1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/outer_product.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "outer_product",
    "family": "dense_matmul",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[1024], [1024]],
    "validation_shape": [[1024], [1024]],
    "validation_reason": None,
}

"""Level 1: Outer product via Pallas.

Provenance: jnp.outer, rank-1 update pattern used in Evoformer/AlphaFold
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _outer_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = x[:, None] * y[None, :]


def pallas_outer_product(x: jax.Array, y: jax.Array) -> jax.Array:
    m = x.shape[0]
    n = y.shape[0]

    return pl.pallas_call(
        _outer_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m,), lambda i: (0,)),
            pl.BlockSpec((n,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((m, n), lambda i: (0, 0)),
    )(x, y)


pallas_kernel = pallas_outer_product
task_name = "outer_product"
input_shapes = [(1024,), (1024,)]
category = "matmul"
level = 1


kernel = pallas_outer_product
