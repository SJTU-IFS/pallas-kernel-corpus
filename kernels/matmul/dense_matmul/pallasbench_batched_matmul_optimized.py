"""Standalone PallasBench batched_matmul kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/batched_matmul.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_batched_matmul`` (also exported as ``kernel``).

Contract ``batched_matmul``, in the ``dense_matmul`` family.  The reference is
upstream's own ``jax_batched_matmul`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("batched_matmul")`` there reproduces the dtypes
and value ranges upstream's benchmark harness uses, so a comparison here is
against upstream's definition of the task rather than a corpus reading of it.

Native shape: [[8, 256, 256], [8, 256, 256]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/batched_matmul.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "batched_matmul",
    "family": "dense_matmul",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[8, 256, 256], [8, 256, 256]],
    "validation_shape": [[8, 256, 256], [8, 256, 256]],
    "validation_reason": None,
}

"""Level 1: Batched matrix multiplication via Pallas.

Provenance: jnp.matmul with batch dims, used in multi-head attention
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _batched_matmul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] @ y_ref[...]


def pallas_batched_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    batch, m, k = x.shape
    _, _, n = y.shape

    return pl.pallas_call(
        _batched_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((batch, m, n), x.dtype),
        grid=(batch,),
        in_specs=[
            pl.BlockSpec((1, m, k), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, k, n), lambda b: (b, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, m, n), lambda b: (b, 0, 0)),
    )(x, y)


pallas_kernel = pallas_batched_matmul
task_name = "batched_matmul"
input_shapes = [(8, 256, 256), (8, 256, 256)]
category = "matmul"
level = 1


kernel = pallas_batched_matmul
