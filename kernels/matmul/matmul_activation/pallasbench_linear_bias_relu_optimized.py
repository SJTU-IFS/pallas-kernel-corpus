"""Standalone PallasBench linear_bias_relu kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/linear_bias_relu.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_linear_bias_relu`` (also exported as ``kernel``).

Contract ``linear_bias_relu``, in the ``matmul_activation`` family.  The
reference is upstream's own ``jax_linear_bias_relu`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("linear_bias_relu")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[1024, 1024], [1024, 2048], [2048]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/linear_bias_relu.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "linear_bias_relu",
    "family": "matmul_activation",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[1024, 1024], [1024, 2048], [2048]],
    "validation_shape": [[1024, 1024], [1024, 2048], [2048]],
    "validation_reason": None,
}

"""Level 2: Fused Linear + Bias + ReLU via Pallas.

Provenance: keras-team FusedDense pattern, standard MLP first layer
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _linear_bias_relu_kernel(x_ref, w_ref, b_ref, o_ref):
    z = x_ref[...] @ w_ref[...] + b_ref[...]
    o_ref[...] = jnp.maximum(z, 0)


def pallas_linear_bias_relu(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    m, k = x.shape
    _, n = w.shape
    BLOCK_M = min(m, 128)

    return pl.pallas_call(
        _linear_bias_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((n,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, n), lambda i: (i, 0)),
    )(x, w, b)


pallas_kernel = pallas_linear_bias_relu
task_name = "linear_bias_relu"
input_shapes = [(1024, 1024), (1024, 2048), (2048,)]
category = "matmul_activation"
level = 2


kernel = pallas_linear_bias_relu
