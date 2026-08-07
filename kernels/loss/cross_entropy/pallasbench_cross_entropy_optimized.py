"""Standalone PallasBench cross_entropy kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/cross_entropy.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_cross_entropy`` (also exported as ``kernel``).

Contract ``cross_entropy``, in the ``cross_entropy`` family.  The reference is
upstream's own ``jax_cross_entropy`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("cross_entropy")`` there reproduces the dtypes
and value ranges upstream's benchmark harness uses, so a comparison here is
against upstream's definition of the task rather than a corpus reading of it.

Native shape: [[1024, 512], [1024, 512]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/cross_entropy.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "cross_entropy",
    "family": "cross_entropy",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[1024, 512], [1024, 512]],
    "validation_shape": [[1024, 512], [1024, 512]],
    "validation_reason": None,
}

"""Level 1: Row-wise cross-entropy loss via Pallas.

Provenance: openxla/tokamax linear_softmax_cross_entropy_loss pattern
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _cross_entropy_kernel(logits_ref, labels_ref, o_ref):
    logits = logits_ref[...]
    labels = labels_ref[...]
    row_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - row_max
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    o_ref[...] = -jnp.sum(labels * log_probs, axis=-1)


def pallas_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    n_rows = logits.shape[0]
    n_cols = logits.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _cross_entropy_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), logits.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(logits, labels)


pallas_kernel = pallas_cross_entropy
task_name = "cross_entropy"
input_shapes = [(1024, 512), (1024, 512)]
category = "loss"
level = 1


kernel = pallas_cross_entropy
