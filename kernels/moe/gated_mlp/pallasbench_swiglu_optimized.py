"""Standalone PallasBench swiglu kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/swiglu.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_swiglu`` (also exported as ``kernel``).

Contract ``swiglu``, in the ``gated_mlp`` family.  The reference is upstream's
own ``jax_swiglu`` from `pallasbench/baselines/jax_baseline.py`, carried in
this directory's `baseline.py`; ``create_inputs("swiglu")`` there reproduces
the dtypes and value ranges upstream's benchmark harness uses, so a comparison
here is against upstream's definition of the task rather than a corpus reading
of it.

Native shape: [[512, 1024], [1024, 2048], [1024, 2048]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/swiglu.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "swiglu",
    "family": "gated_mlp",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[512, 1024], [1024, 2048], [1024, 2048]],
    "validation_shape": [[512, 1024], [1024, 2048], [1024, 2048]],
    "validation_reason": None,
}

"""Level 2: Fused SwiGLU activation via Pallas.

SwiGLU: gate = silu(x @ W_gate), up = x @ W_up, output = gate * up.
Demonstrates: multi-input fusion, gated activation, silu transcendental.
Inspired by pallas-forge's SwiGLU kernel.
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _swiglu_kernel(x_ref, w_gate_ref, w_up_ref, o_ref):
    x = x_ref[...]
    gate = x @ w_gate_ref[...]
    gate = gate / (1.0 + jnp.exp(-gate))  # silu
    up = x @ w_up_ref[...]
    o_ref[...] = gate * up


def pallas_swiglu(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array
) -> jax.Array:
    m, k = x.shape
    _, n = w_gate.shape
    BLOCK_M = min(m, 128)

    return pl.pallas_call(
        _swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // BLOCK_M,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, k), lambda i: (i, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
            pl.BlockSpec((k, n), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, n), lambda i: (i, 0)),
    )(x, w_gate, w_up)


pallas_kernel = pallas_swiglu
task_name = "swiglu"
input_shapes = [(512, 1024), (1024, 2048), (1024, 2048)]
category = "mlp_fusion"
level = 2


kernel = pallas_swiglu
