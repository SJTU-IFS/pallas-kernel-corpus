"""Standalone PallasBench one_hot kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/one_hot.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_one_hot`` (also exported as ``kernel``).

Contract ``one_hot``, in the ``one_hot`` family.  The reference is upstream's
own ``jax_one_hot`` from `pallasbench/baselines/jax_baseline.py`, carried in
this directory's `baseline.py`; ``create_inputs("one_hot")`` there reproduces
the dtypes and value ranges upstream's benchmark harness uses, so a comparison
here is against upstream's definition of the task rather than a corpus reading
of it.

Native shape: [[512]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/one_hot.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "one_hot",
    "family": "one_hot",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[512]],
    "validation_shape": [[512]],
    "validation_reason": None,
}

"""Level 1: One-hot encoding via Pallas.

Provenance: jax.nn.one_hot, used in cross-entropy label preparation
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _one_hot_kernel(indices_ref, o_ref):
    idx = indices_ref[...]
    num_classes = o_ref.shape[-1]
    o_ref[...] = (jnp.arange(num_classes) == idx[..., None]).astype(jnp.float32)


def pallas_one_hot(indices: jax.Array) -> jax.Array:
    seq_len = indices.shape[0]
    num_classes = 1024

    return pl.pallas_call(
        _one_hot_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, num_classes), jnp.float32),
        grid=(1,),
        in_specs=[pl.BlockSpec((seq_len,), lambda i: (0,))],
        out_specs=pl.BlockSpec((seq_len, num_classes), lambda i: (0, 0)),
    )(indices)


pallas_kernel = pallas_one_hot
task_name = "one_hot"
input_shapes = [(512,)]
category = "index"
level = 1


kernel = pallas_one_hot
