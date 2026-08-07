"""Standalone PallasBench add kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/add_op.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_add`` (also exported as ``kernel``).

Contract ``add``, in the ``elementwise_binary`` family.  The reference is
upstream's own ``jax_add`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("add")`` there
reproduces the dtypes and value ranges upstream's benchmark harness uses, so a
comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[4096, 4096], [4096, 4096]].
Validated at [[512, 4096], [512, 4096]] instead, because this v6e has 127.94
MiB of VMEM, and the kernel maps whole arrays into it at once: at the native
4096x4096 f32 the two inputs and the output are 64 MiB each, and the compiler
reports using 192.00 MiB of 127.94 MiB. Largest power-of-two reduction of the
leading dimension that compiles. The kernel itself is untouched -- only the
shape it is called with differs, and the native shape stays recorded above and
in ``SOURCE``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/add_op.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "add",
    "family": "elementwise_binary",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[4096, 4096], [4096, 4096]],
    "validation_shape": [[512, 4096], [512, 4096]],
    "validation_reason": 'this v6e has 127.94 MiB of VMEM, and the kernel maps whole arrays into it at once: at the native 4096x4096 f32 the two inputs and the output are 64 MiB each, and the compiler reports using 192.00 MiB of 127.94 MiB. Largest power-of-two reduction of the leading dimension that compiles.',
}

"""Level 1: Elementwise addition via Pallas.

Provenance: jnp.add, fundamental binary op in residual connections
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def pallas_add(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    MAX_BLOCK = 65536
    block_size = min(n, MAX_BLOCK)
    grid_size = n // block_size

    return pl.pallas_call(
        _add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
            pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
        ],
        out_specs=pl.BlockSpec((block_size, *x.shape[1:]), lambda i: (i, *([0] * (x.ndim - 1)))),
    )(x, y)


pallas_kernel = pallas_add
task_name = "add"
input_shapes = [(4096, 4096), (4096, 4096)]
category = "elementwise"
level = 1


kernel = pallas_add
