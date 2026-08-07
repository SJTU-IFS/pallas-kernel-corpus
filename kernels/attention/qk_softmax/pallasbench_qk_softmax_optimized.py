"""Standalone PallasBench qk_softmax kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/qk_softmax.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_qk_softmax`` (also exported as ``kernel``).

Contract ``qk_softmax``, in the ``qk_softmax`` family.  The reference is
upstream's own ``jax_qk_softmax`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("qk_softmax")``
there reproduces the dtypes and value ranges upstream's benchmark harness uses,
so a comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[256, 64], [256, 64]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/qk_softmax.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "qk_softmax",
    "family": "qk_softmax",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[256, 64], [256, 64]],
    "validation_shape": [[256, 64], [256, 64]],
    "validation_reason": None,
}

"""Level 2: Fused QK^T + Softmax via Pallas.

Provenance: jax-ml/jax flash_attention.py attention score computation
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _qk_softmax_kernel(q_ref, k_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(jnp.float32(d_k))
    row_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - row_max)
    o_ref[...] = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)


def pallas_qk_softmax(q: jax.Array, k: jax.Array) -> jax.Array:
    seq_len, d_model = q.shape
    BLOCK_Q = min(seq_len, 128)
    grid_size = seq_len // BLOCK_Q

    return pl.pallas_call(
        _qk_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, seq_len), q.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_Q, seq_len), lambda i: (i, 0)),
    )(q, k)


pallas_kernel = pallas_qk_softmax
task_name = "qk_softmax"
input_shapes = [(256, 64), (256, 64)]
category = "attention_component"
level = 2


kernel = pallas_qk_softmax
