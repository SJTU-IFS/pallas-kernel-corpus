# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText's own JAX references for the ragged attention in this directory.

`reference_mqa`, `reference_mha` and `reference_gqa` are upstream's, lifted
verbatim out of the kernel file (they are defined beside the kernel there) and
placed here so this directory follows the corpus's one-baseline-per-family
shape.  Nothing is re-derived: these are the functions MaxText's own tests
compare the kernel against.

Each returns a **triple** -- output, per-head max logit, and softmax
denominator -- not just the attention output, because the kernel is designed to
be composed across sequence splits and the caller needs the running statistics
to do that.

Source:
  repository: https://github.com/AI-Hypercomputer/maxtext
  commit: ca420634a9e9e73feaacc8001f605163d2d80ea1
  path: src/maxtext/kernels/attention/ragged_attention.py  (the reference_* functions)
"""

from __future__ import annotations

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/AI-Hypercomputer/maxtext",
    "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
    "path": "src/maxtext/kernels/attention/ragged_attention.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("ragged_attention",),
}

import functools

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp


# ---- inlined from src/maxtext/common/common_types.py: DEFAULT_MASK_VALUE ----
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@functools.partial(jax.jit, static_argnames=["mask_value"])
def reference_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Multi query attention reference.

  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, head_dim] jax.Array.
    v: A [batch_size, seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """
  logits = jnp.einsum("bhd,btd->bht", q.astype(jnp.float32), k.astype(jnp.float32))
  mask = jnp.arange(k.shape[1])[None] < lengths[:, None]

  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None]
  logits_max = logits.max(axis=-1)

  unnormalized = jnp.exp(logits - logits_max[..., None])
  denominator = unnormalized.sum(axis=-1)
  o = jnp.einsum("bht,btd->bhd", unnormalized.astype(v.dtype), v) / denominator[..., None]
  return o, logits_max[..., None], denominator[..., None]


@jax.jit
def reference_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Multi head attention reference.

  Args:
    q: A [batch_size, 1, num_heads, head_dim] jax.Array.
    k: A [batch_size, seq_len, num_heads, head_dim] jax.Array.
    v: A [batch_size, seq_len, num_heads, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """
  q = jnp.swapaxes(q, 1, 2)
  k = jnp.swapaxes(k, 1, 2)
  v = jnp.swapaxes(v, 1, 2)
  return jax.vmap(functools.partial(reference_mqa, mask_value=mask_value), in_axes=(1, 1, 1, None), out_axes=2)(
      q, k, v, lengths
  )


@functools.partial(jax.jit, static_argnames=["mask_value"])
def reference_gqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Vanilla attention GQA implementation for reference.

  Args:
    q: A [batch_size, num_q_heads, head_dim] jax.Array.
    k: A [batch_size, num_kv_heads, max_seq_len, head_dim] jax.Array.
    v: A [batch_size, num_kv_heads, max_seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.

  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """
  batch_size, num_heads_q, head_dim = q.shape
  _, num_heads_kv, seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads_q % num_heads_kv == 0

  q = q.reshape(batch_size, num_heads_kv, num_heads_q // num_heads_kv, head_dim)

  logits = jnp.einsum("bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32))
  mask = jnp.arange(seq_len)[None] < lengths[:, None]
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]
  logits_max = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - logits_max[..., None])
  denominator = unnormalized.sum(axis=-1)
  o = jnp.einsum("bhgt,bhtd->bhgd", unnormalized.astype(v.dtype), v) / denominator[..., None]
  logits_max = logits_max.reshape(batch_size, 1, num_heads_q, 1)
  denominator = denominator.reshape(batch_size, 1, num_heads_q, 1)
  o = o.reshape(batch_size, 1, num_heads_q, head_dim)
  return o, logits_max, denominator
