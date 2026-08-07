"""Standalone PallasBench reduce_max kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/reduce_max.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_reduce_max`` (also exported as ``kernel``).

Contract ``reduce_max``, in the ``reduction`` family.  The reference is
upstream's own ``jax_reduce_max`` from `pallasbench/baselines/jax_baseline.py`,
carried in this directory's `baseline.py`; ``create_inputs("reduce_max")``
there reproduces the dtypes and value ranges upstream's benchmark harness uses,
so a comparison here is against upstream's definition of the task rather than a
corpus reading of it.

Native shape: [[4096, 2048]].
Validated at [[2048, 2048]] instead, because this v6e's scoped-VMEM limit is 32
MiB (E1001 CompileTimeScopedVmemOom), and at the native 4096x2048 f32 this
kernel needs 33.71 MiB of scoped allocation. Largest power-of-two reduction of
the leading dimension that compiles. The kernel itself is untouched -- only the
shape it is called with differs, and the native shape stays recorded above and
in ``SOURCE``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/reduce_max.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "reduce_max",
    "family": "reduction",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[4096, 2048]],
    "validation_shape": [[2048, 2048]],
    "validation_reason": "this v6e's scoped-VMEM limit is 32 MiB (E1001 CompileTimeScopedVmemOom), and at the native 4096x2048 f32 this kernel needs 33.71 MiB of scoped allocation. Largest power-of-two reduction of the leading dimension that compiles.",
}

"""Level 1: Row-wise max reduction via Pallas.

Provenance: jnp.max reduction, used in softmax numerics and argmax patterns
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _reduce_max_kernel(x_ref, o_ref):
    o_ref[...] = jnp.max(x_ref[...], axis=-1)


def pallas_reduce_max(x: jax.Array) -> jax.Array:
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    MAX_BLOCK = 65536
    block_rows = min(n_rows, MAX_BLOCK)
    grid_size = n_rows // block_rows

    return pl.pallas_call(
        _reduce_max_kernel,
        out_shape=jax.ShapeDtypeStruct((n_rows,), x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((block_rows, n_cols), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((block_rows,), lambda i: (i,)),
    )(x)


pallas_kernel = pallas_reduce_max
task_name = "reduce_max"
input_shapes = [(4096, 2048)]
category = "reduce"
level = 1


kernel = pallas_reduce_max
