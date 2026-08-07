"""Standalone PallasBench multi_head_attention kernel (level 3).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level3/multi_head_attention.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_multi_head_attention`` (also exported as ``kernel``).

Contract ``multi_head_attention``, in the ``multi_head_attention`` family.  The
reference is upstream's own ``jax_multi_head_attention`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("multi_head_attention")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[8, 256, 64], [8, 256, 64], [8, 256, 64]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level3/multi_head_attention.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "multi_head_attention",
    "family": "multi_head_attention",
    "level": 3,
    "launch_points": 1,
    "native_shape": [[8, 256, 64], [8, 256, 64], [8, 256, 64]],
    "validation_shape": [[8, 256, 64], [8, 256, 64], [8, 256, 64]],
    "validation_reason": None,
}

"""Level 3: Multi-Head Attention via Pallas.

Provenance: jax-ml/jax pallas/ops/tpu/flash_attention.py
             AI-Hypercomputer/maxtext splash attention training kernel
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _mha_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(jnp.float32(d_k))
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    o_ref[...] = weights @ v


def pallas_multi_head_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    n_heads, seq_len, d_head = q.shape

    return pl.pallas_call(
        _mha_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(n_heads,),
        in_specs=[
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
            pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, seq_len, d_head), lambda h: (h, 0, 0)),
    )(q, k, v)


pallas_kernel = pallas_multi_head_attention
task_name = "multi_head_attention"
input_shapes = [(8, 256, 64), (8, 256, 64), (8, 256, 64)]
category = "attention"
level = 3


kernel = pallas_multi_head_attention
