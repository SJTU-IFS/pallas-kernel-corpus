"""JAX references for the collective kernels in this directory.

**Neither of these has ever been executed.** Both kernels require hardware this
corpus did not have -- `all_gather_matmul` needs exactly 8 devices, the
hierarchical reduce-scatter needs a multi-chip topology -- so these references
are written from the operations' definitions and carried unrun. Every other
`baseline.py` in this corpus states a checked relationship; this one states an
intended one, and the distinction is the whole reason for this paragraph.

They are nonetheless plain identities rather than readings of anyone's
convention, which is what makes writing them defensible at all. An
all-gather-matmul must equal a matmul on the gathered operand; a reduce-scatter
must equal a sum across devices followed by a slice. Both have exact JAX
spellings, given below, and both are meant to be called inside the same
`shard_map` the kernel runs in.

Source:
  the definitions of `jax.lax.all_gather` and `jax.lax.psum_scatter`;
  no upstream reference file exists for either kernel.
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("all_gather_matmul", "hierarchical_reduce_scatter"),
    "validated": False,
    "requires_devices": 8,
}

import jax
import jax.numpy as jnp
from jax import lax


def all_gather_matmul(x_shard: jax.Array, y: jax.Array, *, axis_name: str,
                      rhs_transpose: bool = False) -> jax.Array:
    """What the fused kernel must equal: gather every shard, then multiply.

    Call inside a `shard_map` over `axis_name`, with `x_shard` the local rows
    of `x`. The kernel overlaps the gather with the multiply; this does them in
    sequence, which is the point -- same answer, no overlap.
    """
    x = lax.all_gather(x_shard, axis_name, axis=0, tiled=True)
    return jnp.dot(x, y.T if rhs_transpose else y)


def hierarchical_reduce_scatter(local_x: jax.Array, *, axis_name: str) -> jax.Array:
    """What the SparseCore kernel must equal: sum across devices, keep your slice.

    `local_x` is this device's partial sum over the full token range; the result
    is the fully reduced values for the slice this device owns. `psum_scatter`
    is exactly that operation, without the Die-to-Die / Chip-to-Chip pipelining
    the kernel exists to exploit.
    """
    return lax.psum_scatter(local_x, axis_name, scatter_dimension=0, tiled=True)
