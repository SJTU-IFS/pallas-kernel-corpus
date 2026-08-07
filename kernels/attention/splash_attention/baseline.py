"""Reference notes for the Splash Attention kernels in this directory.

Splash Attention is **block-sparse** attention: a ``mask_lib`` Mask object is
compiled into block metadata ahead of the call, and the kernel visits only the
blocks the mask marks live.  Masks are built outside the jitted call, matching
upstream.

Contract ``splash_attention_mha``::

    q     [num_q_heads, seq_len, head_dim]
    k     [num_kv_heads, seq_len, head_dim]     num_q_heads % num_kv_heads == 0
    v     [num_kv_heads, seq_len, head_dim]
    mask  a mask_lib.MultiHeadMask with one entry per query head
    ->    [num_q_heads, seq_len, head_dim]

There is no batch dimension: the kernel is per-device and callers ``vmap`` over
batch, which is what JAXBench's ``workload`` does.  Q is *not* pre-scaled here,
unlike the Tokamax Splash contract (``splash_mha_hsd``) already in the corpus at
``kernels/attention/flash_attention/tokamax_optimized.py`` -- that one takes
pre-scaled Q and independent ``Dqk``/``Dv``.  The two are not interchangeable,
which is why they live in different directories despite sharing a family.

## This family has backward passes

These are the corpus's **first migrated backward kernels**.  Each file carries
three Pallas launch points:

    _splash_attention_forward     the forward pass
    _splash_attention_bwd_dq      backward with respect to Q
    _splash_attention_bwd_dkv     backward with respect to K and V

The kernel builders return a ``custom_vjp``, so ``jax.grad`` through the
forward entry point exercises both backward launches.  That is how the corpus
validates them -- see ``tests/test_splash_attention_tpu.py``.

## Why this file defines no reference

Both migrated files ship ``attention_reference`` and ``make_attention_reference``
upstream, and both are pure JAX (verified: no ``pallas`` or ``pl.`` reference in
either body).  They take the same mask object the kernel takes, so they are
exact-contract references including for the backward pass, where
``backward_impl="vanilla"`` gives a plain autodiff gradient to compare against.

Writing a corpus reference would mean re-deriving the block-sparse masking
semantics by hand, with the same risk of a subtly wrong reference producing
false failures that applied in the ragged-paged-attention and MLA families.
The upstream references are used instead, and because the two implementations
come from different repositories, each is also checked against the other's.

## Reading the gradient tolerances

Gradient comparisons report a large ``max_abs`` (order 1-3) alongside a cosine
of 0.99998.  That is not a failure: the gradients themselves have large
magnitudes at these shapes, so absolute error scales with them.  Cosine
similarity is the meaningful statistic here, and the relative error is ~1e-5.

## What is not migrated

Of the family's 13 audited TPU launch points, 7 are now migrated (3 here from
JAXBench, 3 here from MaxText, 1 forward in the flash_attention directory from
Tokamax).  The rest:

* **JAXBench ``4p_Sparse_Attention``** (3 launches) is the *same kernel* as
  ``2p_GQA_Attention``: the two files differ only in ``create_inputs``,
  ``get_flops`` and ``workload``, with 28 of 31 top-level definitions
  AST-identical.  Migrating it would double-count, the same reasoning applied
  to ``3p_MLA_Attention``.
* **MaxText ``tokamax_splash_attention/``** (2 launches) is MaxText's vendored
  copy of Tokamax's kernel, and it has diverged from both its own splash kernel
  and from Tokamax's -- only 35% of shared definitions are AST-identical to
  Tokamax's.  It needs three repo-local modules (base, mask, mask_info) and is
  audited but not migrated.
* **Tokamax's ``_splash_attention_bwd_dkv``** is audited but not migrated; only
  its forward pass is in the corpus.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)


SOURCE = {
    "kind": "reference-notes",
    "backend": "jax",
    "target": "portable",
    "contracts": ("splash_attention_mha",),
}


# JAXBench 2p_GQA_Attention CONFIG: Llama-3.1-405B GQA.
NATIVE_CONFIG = {
    "batch": 4,
    "seq_len": 4096,
    "num_query_heads": 128,
    "num_kv_heads": 8,
    "head_dim": 128,
}


def causal_multi_head_mask(seq_len: int, num_q_heads: int):
    """The mask JAXBench's workload builds: causal, one entry per query head."""
    return mask_lib.MultiHeadMask(
        [mask_lib.CausalMask(shape=(seq_len, seq_len))] * num_q_heads
    )


def create_inputs(
    *,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    seq_len: int = 512,
    head_dim: int = 128,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-device ``[H, S, D]`` inputs; callers vmap over batch."""
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    keys = jax.random.split(jax.random.key(seed), 3)
    return (
        jax.random.normal(keys[0], (num_q_heads, seq_len, head_dim), dtype),
        jax.random.normal(keys[1], (num_kv_heads, seq_len, head_dim), dtype),
        jax.random.normal(keys[2], (num_kv_heads, seq_len, head_dim), dtype),
    )


def logical_flops(
    *, batch: int, num_q_heads: int, seq_len: int, head_dim: int, causal: bool = True
) -> int:
    """Full-rectangle QK + PV FLOPs, halved when the mask is causal.

    Splash only visits live blocks, so the causal count is the closer estimate
    of scheduled work; both are reported by the profiler.
    """
    rectangle = 4 * batch * num_q_heads * seq_len * seq_len * head_dim
    return rectangle // 2 if causal else rectangle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the Splash contract and a sample input specification."
    )
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    start = time.perf_counter()
    q, k, v = create_inputs(
        num_q_heads=args.q_heads, num_kv_heads=args.kv_heads,
        seq_len=args.seq_len, head_dim=args.head_dim,
    )
    mask = causal_multi_head_mask(args.seq_len, args.q_heads)
    jax.block_until_ready((q, k, v))
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline-notes",
                "contract": "splash_attention_mha",
                "shapes": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)},
                "mask": type(mask).__name__,
                "launch_points_per_file": ["forward", "backward_dq", "backward_dkv"],
                "note": (
                    "no runnable reference here; the migrated files ship "
                    "pure-JAX attention_reference upstream -- see the module "
                    "docstring"
                ),
                "build_ms": elapsed_ms,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
