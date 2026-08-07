"""Standalone PallasBench triangle_update kernel (level 3).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level3/triangle_update.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_triangle_update`` (also exported as ``kernel``).

Contract ``triangle_update``, in the ``genomics`` family.  The reference is
upstream's own ``jax_triangle_update`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("triangle_update")`` there reproduces the dtypes
and value ranges upstream's benchmark harness uses, so a comparison here is
against upstream's definition of the task rather than a corpus reading of it.

Native shape: [[64, 64, 32], [64, 64]].
Validated at [[32, 32, 32], [32, 32]] instead, because this v6e's scoped-VMEM
limit is 32 MiB (E1001 CompileTimeScopedVmemOom), and the kernel materialises a
(c, n, n, n) outer product -- 33.5 MiB at the native n=64 -- for a 39.08 MiB
scoped allocation. n is the leading dimension of both inputs, so halving it is
coherent. The kernel itself is untouched -- only the shape it is called with
differs, and the native shape stays recorded above and in ``SOURCE``.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level3/triangle_update.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "triangle_update",
    "family": "genomics",
    "level": 3,
    "launch_points": 1,
    "native_shape": [[64, 64, 32], [64, 64]],
    "validation_shape": [[32, 32, 32], [32, 32]],
    "validation_reason": "this v6e's scoped-VMEM limit is 32 MiB (E1001 CompileTimeScopedVmemOom), and the kernel materialises a (c, n, n, n) outer product -- 33.5 MiB at the native n=64 -- for a 39.08 MiB scoped allocation. n is the leading dimension of both inputs, so halving it is coherent.",
}

"""Level 3: Triangle Multiplicative Update via Pallas.

Implements the core triangular multiplicative update from AlphaFold2/3's
Evoformer / Pairformer: for each pair (i,j), aggregate information from
all intermediate positions k via element-wise product of edges (i,k)
and (k,j), enabling triplet reasoning for 3D structure consistency.

This is the computational heart of protein structure prediction and
is the most expensive operation in the Pairformer stack.

Provenance: google-deepmind/alphafold3 Pairformer triangle multiplication
             "Triangle Multiplication Is All You Need" (arXiv:2510.18870)
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _triangle_update_kernel(pair_ref, mask_ref, o_ref):
    pair = pair_ref[...]
    mask = mask_ref[...]
    n, _, c = pair.shape

    left_proj = pair * mask[:, :, None]
    right_proj = pair * mask[:, :, None]

    # Triangle update: out[i,j] = sum_k left[i,k] * right[k,j]
    # This is effectively a batched matmul over the channel dimension
    left_t = left_proj.transpose(2, 0, 1)
    right_t = right_proj.transpose(2, 0, 1)
    update = jnp.sum(left_t[:, :, :, None] * right_t[:, None, :, :], axis=2)
    update = update.transpose(1, 2, 0)

    o_ref[...] = pair + update


def pallas_triangle_update(pair: jax.Array, mask: jax.Array) -> jax.Array:
    n, _, c = pair.shape

    return pl.pallas_call(
        _triangle_update_kernel,
        out_shape=jax.ShapeDtypeStruct(pair.shape, pair.dtype),
        grid=(1,),
        in_specs=[
            pl.BlockSpec(pair.shape, lambda i: (0, 0, 0)),
            pl.BlockSpec(mask.shape, lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec(pair.shape, lambda i: (0, 0, 0)),
    )(pair, mask)


pallas_kernel = pallas_triangle_update
task_name = "triangle_update"
input_shapes = [(64, 64, 32), (64, 64)]
category = "genomics"
level = 3


kernel = pallas_triangle_update
