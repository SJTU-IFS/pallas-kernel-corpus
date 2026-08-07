"""Standalone sglang-jax biased topk MoE router kernel.

Source:
  repository: https://github.com/sgl-project/sglang-jax
  commit: a7353325e8c00d287294c2cd679a77173f1a4594
  paths:
    python/sgl_jax/srt/kernels/biased_topk/
      tuned_block_sizes.py
      v1/kernel.py
  transformation: the repo-local modules were flattened in dependency order and
    the repo-local imports removed. The kernel bodies are unmodified.

Entry points: ``topk_pallas``, ``biased_topk_pallas``

plain and bias-corrected MoE router top-k.  ``biased_topk_pallas``
selects on ``logits + correction_bias`` but returns the *pre-bias*
weights, which is what a router needs for the combine step.

All entry points return batch-major ``(weights[batch, topk], ids[batch, topk])``.
Inside the kernels the Pallas grid works transposed -- the batch dimension is
the lane dimension -- and the wrappers transpose back before returning, which is
why their internal variables are named ``weights_t``/``ids_t``.

``num_experts`` must be a multiple of 128; the kernels reject anything else.
``block_tokens="auto"`` picks the largest safe 128-aligned divisor of the batch.
"""

from __future__ import annotations

SOURCE = {
    "repository": "https://github.com/sgl-project/sglang-jax",
    "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
    "path": "python/sgl_jax/srt/kernels/biased_topk",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "router_biased_topk",
}


# ---- flattened from tuned_block_sizes.py ----

"""Tuned token block sizes for the biased top-k Pallas kernel."""

import jax


def _device_name() -> str:
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("not a TPU device")
    if kind.endswith(" lite"):
        return kind[: -len(" lite")] + "e"
    if kind == "TPU7x":
        return "TPU v7"
    return kind


# device_name -> {(T, E, k): block_tokens}
# TPU v7 E=384/k=8 entries are populated from the real-device tuner.
TUNED_BT: dict[str, dict[tuple[int, int, int], int]] = {
    "TPU v6e": {
        (64, 384, 8): 64,
        (128, 384, 8): 128,
        (256, 384, 8): 256,
        (512, 384, 8): 256,
        (1024, 384, 8): 256,
        (2048, 384, 8): 1024,
        (4096, 384, 8): 1024,
        (8192, 384, 8): 1024,
        (16384, 384, 8): 1024,
        (32768, 384, 8): 1024,
    },
    "TPU v7": {
        (64, 384, 8): 64,
        (128, 384, 8): 128,
        (256, 384, 8): 256,
        (512, 384, 8): 512,
        (1024, 384, 8): 512,
        (2048, 384, 8): 512,
        (4096, 384, 8): 1024,
        (8192, 384, 8): 1024,
        (16384, 384, 8): 1024,
        (32768, 384, 8): 1024,
    },
}


def get_tuned_bt(tokens: int, experts: int, topk: int) -> int | None:
    """Return a measured block size for this exact routing shape."""
    try:
        device = _device_name()
    except Exception:  # noqa: BLE001
        return None
    return TUNED_BT.get(device, {}).get((tokens, experts, topk))


# ---- flattened from v1/kernel.py ----

"""Sort-free top-k routing for TPU.

The plain path returns the same values and ids as ``jax.lax.top_k`` for finite
f32 router scores. The biased path selects with
``router_logits + correction_bias`` while returning pre-bias weights. Tokens
occupy the TPU lane dimension in the VMEM compute layout so expert reductions
avoid cross-lane permutation.
"""



import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp



NEG_INF = -jnp.inf
SAFE_AUTO_BT = 2048


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _safe_auto_block_tokens(batch_size: int) -> int | None:
    if batch_size <= SAFE_AUTO_BT:
        return batch_size
    for candidate in range(SAFE_AUTO_BT, 0, -128):
        if batch_size % candidate == 0:
            return candidate
    return None


def _select_topk(
    logits,  # [E, BT] f32, values returned to the caller
    scores,  # [E, BT] f32, values used for selection
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    block_tokens = logits.shape[1]

    expert_iota = jax.lax.broadcasted_iota(
        jnp.int32,
        (num_experts, block_tokens),
        0,
    )
    row_iota = jax.lax.broadcasted_iota(jnp.int32, (topk, block_tokens), 0)
    ids_init = jnp.full((topk, block_tokens), -1, dtype=jnp.int32)
    weights_init = jnp.zeros((topk, block_tokens), dtype=jnp.float32)

    def select_one(k, carry):
        current_scores, ids, weights = carry
        max_score = jnp.max(current_scores, axis=0, keepdims=True)
        selected_id = jnp.min(
            jnp.where(current_scores == max_score, expert_iota, num_experts),
            axis=0,
            keepdims=True,
        )
        selected = expert_iota == selected_id
        selected_weight = jnp.sum(
            jnp.where(selected, logits, 0.0),
            axis=0,
            keepdims=True,
        )
        write_row = row_iota == k
        ids = jnp.where(write_row, selected_id.astype(jnp.int32), ids)
        weights = jnp.where(write_row, selected_weight.astype(jnp.float32), weights)
        current_scores = jnp.where(selected, NEG_INF, current_scores)
        return current_scores, ids, weights

    _, ids, weights = jax.lax.fori_loop(
        0,
        topk,
        select_one,
        (scores, ids_init, weights_init),
        unroll=True,
    )
    weights_ref[...] = weights
    ids_ref[...] = ids


def _biased_topk_kernel(
    logits_ref,  # [BT, E] f32, pre-bias
    bias_ref,  # [E] f32
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    logits = logits_ref[...].astype(jnp.float32).T  # [E, BT]
    scores = logits + bias_ref[...].astype(jnp.float32)[:, None]
    _select_topk(
        logits,
        scores,
        weights_ref,
        ids_ref,
        topk=topk,
        num_experts=num_experts,
    )


def _topk_kernel(
    logits_ref,  # [BT, E] f32
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    logits = logits_ref[...].astype(jnp.float32).T  # [E, BT]
    _select_topk(
        logits,
        logits,
        weights_ref,
        ids_ref,
        topk=topk,
        num_experts=num_experts,
    )


def _resolve_block_tokens(
    router_logits: jax.Array,
    *,
    topk: int,
    block_tokens: int | str,
) -> tuple[int, int, int]:
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be rank 2, got shape={router_logits.shape}")
    batch_size, num_experts = router_logits.shape
    if num_experts % 128 != 0:
        raise ValueError(f"num_experts must be divisible by 128, got {num_experts}")
    if not 1 <= topk <= num_experts:
        raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
    if block_tokens == "auto":
        block_tokens = get_tuned_bt(batch_size, num_experts, topk)
        if block_tokens is None:
            block_tokens = _safe_auto_block_tokens(batch_size)
        if block_tokens is None:
            raise ValueError(
                f"no VMEM-safe block_tokens for batch_size={batch_size}; "
                "fall back to jax.lax.top_k"
            )
    block_tokens = int(block_tokens)
    if not 1 <= block_tokens <= batch_size:
        raise ValueError(f"block_tokens must be in [1, {batch_size}], got {block_tokens}")
    if batch_size % block_tokens != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by block_tokens={block_tokens}"
        )
    return batch_size, num_experts, block_tokens


def biased_topk_pallas(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Select biased top-k expert ids and return their pre-bias weights."""
    batch_size, num_experts, block_tokens = _resolve_block_tokens(
        router_logits,
        topk=topk,
        block_tokens=block_tokens,
    )
    if correction_bias.shape != (num_experts,):
        raise ValueError(
            "correction_bias must have shape "
            f"({num_experts},), got shape={correction_bias.shape}"
        )
    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _biased_topk_kernel,
        topk=topk,
        num_experts=num_experts,
    )
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(batch_size // block_tokens,),
        in_specs=[
            pl.BlockSpec((block_tokens, num_experts), lambda i: (i, 0)),
            pl.BlockSpec((num_experts,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, batch_size), jnp.float32),
            jax.ShapeDtypeStruct((topk, batch_size), jnp.int32),
        ],
        interpret=interpret,
        name="biased-topk",
    )(
        router_logits.astype(jnp.float32),
        correction_bias.astype(jnp.float32),
    )
    return weights_t.T, ids_t.T


def topk_pallas(
    router_logits: jax.Array,
    *,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the top-k values and ids of a finite f32 routing matrix."""
    batch_size, num_experts, block_tokens = _resolve_block_tokens(
        router_logits,
        topk=topk,
        block_tokens=block_tokens,
    )
    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _topk_kernel,
        topk=topk,
        num_experts=num_experts,
    )
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(batch_size // block_tokens,),
        in_specs=[
            pl.BlockSpec((block_tokens, num_experts), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, batch_size), jnp.float32),
            jax.ShapeDtypeStruct((topk, batch_size), jnp.int32),
        ],
        interpret=interpret,
        name="topk",
    )(router_logits.astype(jnp.float32))
    return weights_t.T, ids_t.T



kernel = biased_topk_pallas
