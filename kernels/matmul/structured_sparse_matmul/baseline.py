"""JAX reference for the structured sparse matmul in this directory.

There is nothing to carry from upstream here, and that is the honest position
rather than an omission. A structured-sparse matmul is defined by what it must
*equal*: the dense product of the same matrices with the pruned entries set to
the default value. `tests/kernels/spmm_v1_test.py` says exactly that --
`expected = jnp.dot(lhs, rhs, preferred_element_type=out_dtype)` after
`jnp.where(mask, x, default_val)` -- so the reference is `jnp.dot`, a plain
identity, not a reading of anyone's convention.

The interesting machinery goes the other way. Turning a dense matrix into the
`(nonzeros, metadata)` pair the kernel takes is *not* obvious, and this corpus
does not re-derive it: `Sparsifier` and `gen_sparse_mask` are upstream's and
stay in the kernel file, where the kernel that consumes them can be read beside
them.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tests/kernels/spmm_v1_test.py  (upstream's choice of reference)
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tests/kernels/spmm_v1_test.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("structured_sparse_matmul",),
}

import jax
import jax.numpy as jnp


def dense_matmul(lhs: jax.Array, rhs: jax.Array, *, rhs_transpose: bool = False,
                 out_dtype=None) -> jax.Array:
    """What the sparse kernel must equal, on already-densified operands.

    `lhs` and `rhs` are the matrices *after* pruning -- that is, with the
    entries the mask dropped already replaced by the kernel's `default_value`.
    Densifying is the caller's job precisely because the kernel never sees a
    dense matrix.
    """
    if rhs_transpose:
        rhs = rhs.T
    return jnp.dot(lhs, rhs, preferred_element_type=out_dtype)


def densify(x: jax.Array, mask: jax.Array, default_value) -> jax.Array:
    """Apply a sparsity mask the way upstream's test does before comparing."""
    return jnp.where(mask, x, default_value)
