"""Standalone PallasBench rmsnorm kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/rmsnorm.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_rmsnorm`` (also exported as ``kernel``).

Contract ``rmsnorm``, in the ``rms_norm`` family.  The reference is upstream's
own ``jax_rmsnorm`` from `pallasbench/baselines/jax_baseline.py`, carried in
this directory's `baseline.py`; ``create_inputs("rmsnorm")`` there reproduces
the dtypes and value ranges upstream's benchmark harness uses, so a comparison
here is against upstream's definition of the task rather than a corpus reading
of it.

Native shape: [[2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/rmsnorm.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "rmsnorm",
    "family": "rms_norm",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[2048, 1024]],
    "validation_shape": [[2048, 1024]],
    "validation_reason": None,
}

"""Level 1: RMS normalization via Pallas.

Demonstrates: squared-mean reduction, rsqrt pattern.
Inspired by pallas-forge's RMSNorm kernel (3.44x over XLA).
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _rmsnorm_kernel(x_ref, o_ref):
    x = x_ref[...]
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    o_ref[...] = x * jax.lax.rsqrt(ms + 1e-5)


def pallas_rmsnorm(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_rmsnorm
task_name = "rmsnorm"
input_shapes = [(2048, 1024)]
category = "normalization"
level = 1


kernel = pallas_rmsnorm
