"""Standalone PallasBench nucleotide_onehot kernel (level 1).

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level1/nucleotide_onehot.py
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``pallas_nucleotide_onehot`` (also exported as ``kernel``).

Contract ``nucleotide_onehot``, in the ``genomics`` family.  The reference is
upstream's own ``jax_nucleotide_onehot`` from
`pallasbench/baselines/jax_baseline.py`, carried in this directory's
`baseline.py`; ``create_inputs("nucleotide_onehot")`` there reproduces the
dtypes and value ranges upstream's benchmark harness uses, so a comparison here
is against upstream's definition of the task rather than a corpus reading of
it.

Native shape: [[4096]].
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level1/nucleotide_onehot.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "nucleotide_onehot",
    "family": "genomics",
    "level": 1,
    "launch_points": 1,
    "native_shape": [[4096]],
    "validation_shape": [[4096]],
    "validation_reason": None,
}

"""Level 1: Nucleotide one-hot encoding via Pallas.

Encodes integer-encoded DNA sequences (A=0, C=1, G=2, T=3) into
4-channel one-hot representation used by genomics models (Enformer, etc).

Provenance: google-deepmind/deepmind-research Enformer
             DNA sequence input encoding (one-hot 4-channel)
"""




import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _nucleotide_onehot_kernel(seq_ref, o_ref):
    seq = seq_ref[...]
    num_classes = 4
    o_ref[...] = (jnp.arange(num_classes) == seq[..., None]).astype(jnp.float32)


def pallas_nucleotide_onehot(seq: jax.Array) -> jax.Array:
    seq_len = seq.shape[0]

    return pl.pallas_call(
        _nucleotide_onehot_kernel,
        out_shape=jax.ShapeDtypeStruct((seq_len, 4), jnp.float32),
        grid=(1,),
        in_specs=[pl.BlockSpec((seq_len,), lambda i: (0,))],
        out_specs=pl.BlockSpec((seq_len, 4), lambda i: (0, 0)),
    )(seq)


pallas_kernel = pallas_nucleotide_onehot
task_name = "nucleotide_onehot"
input_shapes = [(4096,)]
category = "genomics"
level = 1


kernel = pallas_nucleotide_onehot
