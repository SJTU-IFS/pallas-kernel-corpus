"""JAX reference for the layout-transpose kernels in this directory.

There is nothing to derive here, and that is worth stating rather than leaving
implicit.  Elsewhere in this corpus a reference is carried from upstream because
writing one risks encoding a *different task* than the kernel implements -- a
packed cache layout, a fused activation order, a reduction convention.  A
transpose has no such freedom: `jnp.transpose(x, axes)` is the definition, it is
exact rather than approximate, and it is what upstream's own
`tests/kernels/transpose_test.py` compares against.

The identity kernel is the same story with even less to say.

Source:
  repository: https://github.com/vllm-project/tpu-inference
  commit: 8b9c90928c94c7230d1bc891534a301510a6a30d
  path: tests/kernels/transpose_test.py  (upstream's own choice of reference)
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/vllm-project/tpu-inference",
    "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
    "path": "tests/kernels/transpose_test.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("layout_transpose",),
}

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def transposed(x: jax.Array, transpose_axes: Sequence[int]) -> jax.Array:
    """What both `xpose_full` and `xpose_pipeline` must return."""
    return jnp.transpose(x, transpose_axes)


def pinned(x: jax.Array) -> jax.Array:
    """What `pin_vmem_custom_call` must return: its input, unchanged.

    The kernel's purpose is a side effect -- leaving the buffer resident in
    VMEM -- which no comparison of values can observe.  What a test can pin
    down is that the values are untouched, bitwise, and that a Pallas kernel
    ran at all rather than the call folding away.
    """
    return x
