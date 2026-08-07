"""Standalone PallasBench softmax kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/softmax.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_softmax`` (also exported as ``kernel``).

Contract ``softmax``, in the ``softmax`` family.  The reference is upstream's
own ``jax_softmax`` from `pallasbench/baselines/jax_baseline.py`, carried in
this directory's `baseline.py`; ``create_inputs("softmax")`` there reproduces
the dtypes and value ranges upstream's benchmark harness uses, so a comparison
here is against upstream's definition of the task rather than a corpus reading
of it.

Native shape: [[2048, 2048]].
Validated at [[1024, 2048]] instead, because this v6e's scoped-VMEM limit is 32
MiB (E1001 CompileTimeScopedVmemOom), and at the native 2048x2048 f32 this
kernel needs 37.02 MiB once its max/sum/exp intermediates are counted. Largest
power-of-two reduction of the leading dimension that compiles. The kernel
itself is untouched -- only the shape it is called with differs, and the native
shape stays recorded above and in ``SOURCE``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/softmax.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "softmax",
    "family": "softmax",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[2048, 2048]],
    "validation_shape": [[1024, 2048]],
    "validation_reason": "this v6e's scoped-VMEM limit is 32 MiB (E1001 CompileTimeScopedVmemOom), and at the native 2048x2048 f32 this kernel needs 37.02 MiB once its max/sum/exp intermediates are counted. Largest power-of-two reduction of the leading dimension that compiles.",
}

"""Level 1: Row-wise softmax via Pallas.

Demonstrates: reductions within a block, numerical stability (max subtraction),
multi-pass pattern (max -> subtract -> exp -> sum -> divide).
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    row_max = jnp.max(x, axis=-1, keepdims=True)
    x_safe = x - row_max
    exp_x = jnp.exp(x_safe)
    sum_exp = jnp.sum(exp_x, axis=-1, keepdims=True)
    o_ref[...] = exp_x / sum_exp


def pallas_softmax(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    n_cols = x.shape[1]
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0)),
    )(x)


pallas_kernel = pallas_softmax
task_name = "softmax"
input_shapes = [(2048, 2048)]
category = "softmax"
level = 1


kernel = pallas_softmax
