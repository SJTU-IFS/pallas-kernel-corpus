"""Standalone PallasBench relu kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/relu.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_relu`` (also exported as ``kernel``).

Contract ``relu``, in the ``activation`` family.  The reference is upstream's
own ``jax_relu`` from `pallasbench/baselines/jax_baseline.py`, carried in this
directory's `baseline.py`; ``create_inputs("relu")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[4096, 4096]].
Validated at [[1024, 4096]] instead, because this v6e has 127.94 MiB of VMEM,
and the kernel maps whole arrays into it at once: at the native 4096x4096 f32
the input and output windows are 64 MiB each, and the compiler reports using
128.00 MiB of 127.94 MiB. Largest power-of-two reduction of the leading
dimension that compiles. The kernel itself is untouched -- only the shape it is
called with differs, and the native shape stays recorded above and in
``SOURCE``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/relu.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "relu",
    "family": "activation",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[4096, 4096]],
    "validation_shape": [[1024, 4096]],
    "validation_reason": 'this v6e has 127.94 MiB of VMEM, and the kernel maps whole arrays into it at once: at the native 4096x4096 f32 the input and output windows are 64 MiB each, and the compiler reports using 128.00 MiB of 127.94 MiB. Largest power-of-two reduction of the leading dimension that compiles.',
}

"""Level 1: Elementwise ReLU via Pallas.

Demonstrates: basic pallas_call, grid, BlockSpec, program_id.
"""



import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _relu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jnp.maximum(x, 0)


def pallas_relu(x: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1))))],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x)


pallas_kernel = pallas_relu
task_name = "relu"
input_shapes = [(4096, 4096)]
category = "activation"
level = 1


kernel = pallas_relu
