"""Standalone PallasBench pairwise_distance kernel (level 2).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level2/pairwise_distance.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_pairwise_distance`` (also exported as ``kernel``).

Contract ``pairwise_distance``, in the ``genomics`` family.  The reference is
upstream's own ``jax_pairwise_distance`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("pairwise_distance")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[256, 32]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level2/pairwise_distance.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "pairwise_distance",
    "family": "genomics",
    "level": 2,
    "launch_points": 1,
    "native_shape": [[256, 32]],
    "validation_shape": [[256, 32]],
    "validation_reason": None,
}

"""Level 2: Pairwise Euclidean distance matrix via Pallas.

Computes the N x N distance matrix from N points in d dimensions.
Core operation in structural biology (AlphaFold distance maps),
molecular dynamics (neighbor lists), and genomics (phylogenetics).

Provenance: google-deepmind/alphafold3 pair representation distance maps
             JAX-MD (arxiv:1912.04232) pairwise distance computation
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _pairwise_dist_kernel(x_ref, o_ref):
    x = x_ref[...]
    n = x.shape[0]
    diff = x[:, None, :] - x[None, :, :]
    o_ref[...] = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)


def pallas_pairwise_distance(x: jax.Array) -> jax.Array:
    n, d = x.shape

    return pl.pallas_call(
        _pairwise_dist_kernel,
        out_shape=jax.ShapeDtypeStruct((n, n), x.dtype),
        grid=(1,),
        in_specs=[pl.BlockSpec((n, d), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((n, n), lambda i: (0, 0)),
    )(x)


pallas_kernel = pallas_pairwise_distance
task_name = "pairwise_distance"
input_shapes = [(256, 32)]
category = "genomics"
level = 2


kernel = pallas_pairwise_distance
