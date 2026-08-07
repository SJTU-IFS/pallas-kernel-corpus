# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tokamax's own JAX reference for the cross-entropy kernels in this directory.

This is `tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py` at the commit below, unmodified apart from
this header and the same `jaxtyping` substitution the kernel file needed -- the
module upstream's own kernel tests compare against.

It defines both directions.  The backward's signature is worth reading before
using it: it takes `lse` from the forward, because the kernel it checks does not
store logits and recomputes them from `lse` instead.

Source:
  repository: https://github.com/openxla/tokamax
  commit: 927e3f94e8ffe0430cf38bd1423112bb2f69ec66
  path: tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py
"""

SOURCE = {
    "kind": "reference",
    "repository": "https://github.com/openxla/tokamax",
    "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
    "path": "tokamax/_src/ops/linear_softmax_cross_entropy_loss/reference.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("linear_softmax_cross_entropy",),
}

from functools import partial
from typing import Literal
import jax
import jax.numpy as jnp
# ---- corpus substitution for `jaxtyping` ------------------------------------
# Upstream annotates every signature with jaxtyping shape types, and this module
# has no `from __future__ import annotations`, so those expressions are
# evaluated when each `def` is executed.  jaxtyping is not in this corpus's
# pinned dependency set, so these stand-ins take its place: subscriptable and
# unionable, and inert.  They check nothing -- but neither does jaxtyping
# without its runtime checker, which upstream does not install here.
class _ShapeAnnotation:
  """A subscriptable placeholder standing in for a jaxtyping shape type."""

  def __init__(self, name: str):
    self._name = name

  def __getitem__(self, item):
    return self

  def __or__(self, other):
    return self

  def __ror__(self, other):
    return self

  def __repr__(self):
    return self._name


Array = _ShapeAnnotation("Array")
Integer = _ShapeAnnotation("Integer")
Real = _ShapeAnnotation("Real")
Scalar = _ShapeAnnotation("Scalar")


@partial(jax.jit, static_argnames=["reduction"])
def linear_softmax_cross_entropy_loss_fwd_reference(
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> tuple[Real[Array, "*"], Real[Array, "B"]]:
  """The reference Jax implementation of the linear softmax cross-entropy loss."""
  logits = x @ w
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  loss = -labels_one_hot * log_probs
  lse = jax.nn.logsumexp(logits, axis=-1)
  loss_per_sample = jnp.sum(loss, axis=-1, dtype=jnp.float32)

  if reduction == "sum":
    return jnp.sum(loss_per_sample, dtype=jnp.float32), lse
  elif reduction == "mean":
    return jnp.mean(loss_per_sample, dtype=jnp.float32), lse
  elif reduction == "none":
    return loss_per_sample, lse
  else:
    raise ValueError(f"Unsupported reduction method: {reduction}")


@partial(jax.jit, static_argnames=["reduction"])
def linear_softmax_cross_entropy_loss_bwd_reference(
    dout: Real[Array, "*"],
    lse: Real[Array, "B"],
    x: Real[Array, "B H"],
    labels: Integer[Array, "B"],
    w: Real[Array, "H V"],
    *,
    reduction: Literal["sum", "mean", "none"] = "sum",
) -> tuple[Real[Array, "B H"], Real[Array, "H V"]]:
  """The reference Jax implementation of the linear softmax cross-entropy loss backward kernel."""
  labels_one_hot = jax.nn.one_hot(labels, num_classes=w.shape[1], dtype=x.dtype)
  s = -labels_one_hot + jnp.exp(x @ w - lse[:, None])
  num_tokens = x.shape[0]

  if reduction == "none":
    s_scaled = s * dout[:, None]
    x_grad = s_scaled @ w.T
    w_grad = x.T @ s_scaled
  elif reduction == "mean":
    scale = dout / num_tokens
    x_grad = (s @ w.T) * scale
    w_grad = (x.T @ s) * scale
  elif reduction == "sum":
    x_grad = (s @ w.T) * dout
    w_grad = (x.T @ s) * dout
  else:
    raise ValueError(f"Unsupported reduction method: {reduction}")

  return x_grad, w_grad
