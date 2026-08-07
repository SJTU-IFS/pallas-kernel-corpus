"""Standalone PallasBench transformer_block kernel (level 3).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level3/transformer_block.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_transformer_block`` (also exported as ``kernel``).

Contract ``transformer_block``, in the ``transformer_block`` family.  The
reference is upstream's own ``jax_transformer_block`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("transformer_block")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[128, 256], [256, 64], [256, 64], [256, 64], [64, 256], [256, 256]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level3/transformer_block.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "transformer_block",
    "family": "transformer_block",
    "level": 3,
    "launch_points": 1,
    "native_shape": [[128, 256], [256, 64], [256, 64], [256, 64], [64, 256], [256, 256]],
    "validation_shape": [[128, 256], [256, 64], [256, 64], [256, 64], [64, 256], [256, 256]],
    "validation_reason": None,
}

"""Level 3: Simplified Transformer Block (Attention + MLP) via Pallas.

Provenance: AI-Hypercomputer/maxtext Llama/Gemma model architecture
             Standard pre-norm transformer block pattern
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _transformer_block_kernel(x_ref, wq_ref, wk_ref, wv_ref, wo_ref, w_ff_ref, o_ref):
    x = x_ref[...]
    seq_len, d_model = x.shape
    d_head = wq_ref.shape[-1]

    # Pre-norm (RMSNorm)
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(ms + 1e-5)

    # Self-attention
    q = x_norm @ wq_ref[...]
    k = x_norm @ wk_ref[...]
    v = x_norm @ wv_ref[...]
    scores = q @ k.T / jnp.sqrt(jnp.float32(d_head))
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    attn_out = weights @ v
    attn_proj = attn_out @ wo_ref[...]

    # Residual + MLP
    h = x + attn_proj
    ms2 = jnp.mean(h ** 2, axis=-1, keepdims=True)
    h_norm = h * jax.lax.rsqrt(ms2 + 1e-5)
    ff = jnp.maximum(h_norm @ w_ff_ref[...], 0)

    o_ref[...] = h + ff


def pallas_transformer_block(
    x: jax.Array, wq: jax.Array, wk: jax.Array,
    wv: jax.Array, wo: jax.Array, w_ff: jax.Array
) -> jax.Array:
    seq_len, d_model = x.shape

    return pl.pallas_call(
        _transformer_block_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(x.shape, lambda i: (0, 0)),
            pl.BlockSpec(wq.shape, lambda i: (0, 0)),
            pl.BlockSpec(wk.shape, lambda i: (0, 0)),
            pl.BlockSpec(wv.shape, lambda i: (0, 0)),
            pl.BlockSpec(wo.shape, lambda i: (0, 0)),
            pl.BlockSpec(w_ff.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(x.shape, lambda i: (0, 0)),
    )(x, wq, wk, wv, wo, w_ff)


pallas_kernel = pallas_transformer_block
task_name = "transformer_block"
input_shapes = [(128, 256), (256, 64), (256, 64), (256, 64), (64, 256), (256, 256)]
category = "full_model"
level = 3


kernel = pallas_transformer_block
