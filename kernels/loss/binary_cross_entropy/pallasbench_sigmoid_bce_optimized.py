"""Standalone PallasBench sigmoid_bce kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/sigmoid_bce.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_sigmoid_bce`` (also exported as ``kernel``).

Contract ``sigmoid_bce``, in the ``binary_cross_entropy`` family.  The
reference is upstream's own ``jax_sigmoid_bce`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("sigmoid_bce")`` there reproduces the dtypes and
value ranges upstream's benchmark harness uses, so a comparison here is against
upstream's definition of the task rather than a corpus reading of it.

Native shape: [[2048, 1024], [2048, 1024]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/sigmoid_bce.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "sigmoid_bce",
    "family": "binary_cross_entropy",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[2048, 1024], [2048, 1024]],
    "validation_shape": [[2048, 1024], [2048, 1024]],
    "validation_reason": None,
}

"""Level 2: Fused Sigmoid + Binary Cross-Entropy via Pallas.

Provenance: standard binary classification loss fusion
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _sigmoid_bce_kernel(logits_ref, targets_ref, o_ref):
    logits = logits_ref[...]
    targets = targets_ref[...]
    max_val = jnp.maximum(-logits, 0.0)
    loss = max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
    o_ref[...] = loss - targets * logits + targets * loss


def pallas_sigmoid_bce(logits: jax.Array, targets: jax.Array) -> jax.Array:
    n = logits.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _sigmoid_bce_kernel,
        out_shape=jax.ShapeDtypeStruct(logits.shape, logits.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
            pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
        ],
        out_specs=pl.BlockSpec((block_size, *logits.shape[1:]), lambda i: (i, *([0] * (logits.ndim - 1)))),
    )(logits, targets)


pallas_kernel = pallas_sigmoid_bce
task_name = "sigmoid_bce"
input_shapes = [(2048, 1024), (2048, 1024)]
category = "loss_fusion"
level = 2


kernel = pallas_sigmoid_bce
