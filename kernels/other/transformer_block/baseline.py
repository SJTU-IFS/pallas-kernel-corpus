"""JAX references for the PallasBench transformer_block kernels in this directory.

Every function here is upstream's own, copied from
``pallasbench/baselines/jax_baseline.py`` at the commit below -- a module whose
docstring reads "pure JAX reference implementations ... Each function uses only
jax.numpy / jax.lax / jax.nn -- no Pallas".  ``generate_inputs`` is likewise
upstream's, from ``pallasbench/utils.py``: it is what PallasBench's own
benchmark harness feeds these kernels, including the integer dtypes and the
positive-domain ranges that ``log`` and ``rsqrt`` need.

Carrying upstream's reference rather than writing one matters here for the same
reason it did in the attention families: the risk is not that ``jnp.maximum(x,
0)`` is hard to write, it is that a corpus-written reference quietly encodes a
*different task* than the kernel implements.

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  paths:
    pallasbench/baselines/jax_baseline.py  (jax_* references)
    pallasbench/utils.py                   (generate_inputs)

Tasks in this directory: ['transformer_block']
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "backend": "jax",
    "target": "portable",
    "family": "transformer_block",
    "contracts": ['transformer_block'],
}

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


#: Per task: the shapes correctness runs at, PallasBench's own ``native_shapes``
#: (identical unless ``reason`` says why they had to differ), and the dtypes and
#: value ranges upstream attaches to the tasks that need them.  ``None`` for a
#: dtype or range means float32 standard normal, which is what PallasBench uses
#: everywhere else.
TASK_INPUTS = {'transformer_block': {'dtypes': None,
                       'input_shapes': [[128, 256],
                                        [256, 64],
                                        [256, 64],
                                        [256, 64],
                                        [64, 256],
                                        [256, 256]],
                       'native_shapes': [[128, 256],
                                         [256, 64],
                                         [256, 64],
                                         [256, 64],
                                         [64, 256],
                                         [256, 256]],
                       'ranges': None,
                       'reason': None}}


def create_inputs(task: str, seed: int = 0) -> list[jax.Array]:
    """Upstream's inputs for ``task``, at the shape correctness runs at."""
    spec = TASK_INPUTS[task]
    return generate_inputs(
        [tuple(s) for s in spec["input_shapes"]], seed=seed,
        dtypes=spec["dtypes"], ranges=spec["ranges"],
    )


def runs_below_native_shape(task: str) -> str | None:
    """Why ``task`` is validated below its native shape, or None if it is not."""
    return TASK_INPUTS[task]["reason"]

def generate_inputs(
    shapes: Sequence[tuple[int, ...]],
    dtype: str = "float32",
    seed: int = 0,
    dtypes: Sequence[str] | None = None,
    ranges: Sequence[tuple[float, float] | None] | None = None,
) -> list[jax.Array]:
    key = jax.random.PRNGKey(seed)
    inputs = []
    dtype_list = list(dtypes) if dtypes is not None else [dtype] * len(shapes)
    range_list = list(ranges) if ranges is not None else [None] * len(shapes)
    if len(dtype_list) != len(shapes):
        raise ValueError("Number of dtypes must match number of shapes")
    if len(range_list) != len(shapes):
        raise ValueError("Number of ranges must match number of shapes")
    for shape in shapes:
        key, subkey = jax.random.split(key)
        current_dtype = dtype_list[len(inputs)]
        range_spec = range_list[len(inputs)]
        jnp_dtype = jnp.dtype(current_dtype)
        if np.issubdtype(jnp_dtype, np.bool_):
            sample = jax.random.bernoulli(subkey, 0.5, shape).astype(jnp_dtype)
        elif np.issubdtype(jnp_dtype, np.integer):
            if range_spec is not None:
                low, high = range_spec
                sample = jax.random.randint(
                    subkey,
                    shape,
                    int(low),
                    max(int(high), int(low) + 1),
                    dtype=jnp_dtype,
                )
            else:
                upper = max(shape[-1] if shape else 1, 2)
                sample = jax.random.randint(subkey, shape, 0, upper, dtype=jnp_dtype)
        else:
            if range_spec is not None:
                low, high = range_spec
                sample = jax.random.uniform(
                    subkey,
                    shape,
                    dtype=jnp_dtype,
                    minval=low,
                    maxval=high,
                )
            else:
                sample = jax.random.normal(subkey, shape, dtype=jnp_dtype)
        inputs.append(sample)
    return inputs


@jax.jit
def jax_transformer_block(
    x: jax.Array, wq: jax.Array, wk: jax.Array,
    wv: jax.Array, wo: jax.Array, w_ff: jax.Array
) -> jax.Array:
    d_head = wq.shape[-1]
    # Pre-norm
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(ms + 1e-5)
    # Attention
    q = x_norm @ wq
    k = x_norm @ wk
    v = x_norm @ wv
    scores = q @ k.T / jnp.sqrt(jnp.float32(d_head))
    weights = jax.nn.softmax(scores, axis=-1)
    attn_out = weights @ v
    attn_proj = attn_out @ wo
    # Residual + MLP
    h = x + attn_proj
    ms2 = jnp.mean(h ** 2, axis=-1, keepdims=True)
    h_norm = h * jax.lax.rsqrt(ms2 + 1e-5)
    ff = jnp.maximum(h_norm @ w_ff, 0)
    return h + ff


#: The corpus protocol: one named reference per task.
REFERENCES = {
    "transformer_block": jax_transformer_block,
}
