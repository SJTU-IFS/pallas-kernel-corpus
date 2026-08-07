"""JAX references for the flash-attention implementations in this directory.

The source repositories use the name "flash attention" for more than one
interface.  Keep the reference contracts separate instead of pretending that
the 2-D educational kernel and the batched causal MHA kernel are interchangeable.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp


SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("causal_bhsd", "dense_2d", "splash_mha_hsd"),
}


def causal_bhsd(
    query: jax.Array, key: jax.Array, value: jax.Array
) -> jax.Array:
    """Scaled causal attention for tensors shaped ``[B, H, S, D]``."""
    _, _, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    logits = jnp.einsum("bhqd,bhkd->bhqk", query, key)
    logits = logits * (head_dim**-0.5)
    causal_mask = jnp.arange(key_length)[None, :] <= jnp.arange(
        query_length
    )[:, None]
    logits = jnp.where(causal_mask, logits, -jnp.inf)
    probabilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", probabilities, value)


def dense_2d(
    query: jax.Array, key: jax.Array, value: jax.Array
) -> jax.Array:
    """Scaled non-causal attention for tensors shaped ``[S, D]``."""
    logits = query @ key.T / math.sqrt(query.shape[-1])
    return jax.nn.softmax(logits, axis=-1) @ value


def splash_mha_hsd(
    query: jax.Array, key: jax.Array, value: jax.Array
) -> jax.Array:
    """Tokamax Splash contract for pre-scaled tensors shaped ``[H, S, D]``."""
    sequence = query.shape[-2]
    logits = jnp.einsum(
        "hsd,htd->hst",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    )
    causal_mask = jnp.arange(sequence)[None, :] <= jnp.arange(sequence)[:, None]
    logits = jnp.where(causal_mask, logits, -jnp.inf)
    probabilities = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum(
        "hst,htd->hsd", probabilities, value.astype(jnp.float32)
    )
    return output.astype(value.dtype)


def splash_mha_hsd_blockwise(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    block_q: int = 128,
) -> jax.Array:
    """Memory-bounded pure-JAX Splash reference for ``[H,S,D]`` tensors.

    Query blocking avoids materializing the complete ``[H,S,S]`` logits tensor.
    Q is pre-scaled, matching the Tokamax Splash public contract.
    """
    _, sequence, _ = query.shape
    if sequence % block_q:
        raise ValueError(f"{sequence=} must be divisible by {block_q=}")
    output = jnp.zeros(
        (query.shape[0], sequence, value.shape[-1]), dtype=value.dtype
    )

    def body(block_index: int, output: jax.Array) -> jax.Array:
        query_start = block_index * block_q
        query_block = jax.lax.dynamic_slice_in_dim(
            query, query_start, block_q, axis=1
        )
        logits = jnp.einsum(
            "hqd,hkd->hqk",
            query_block.astype(jnp.float32),
            key.astype(jnp.float32),
        )
        query_positions = query_start + jnp.arange(block_q)
        causal_mask = (
            jnp.arange(sequence)[None, :] <= query_positions[:, None]
        )
        logits = jnp.where(causal_mask, logits, -jnp.inf)
        probabilities = jax.nn.softmax(logits, axis=-1)
        output_block = jnp.einsum(
            "hqk,hkd->hqd",
            probabilities,
            value.astype(jnp.float32),
        ).astype(value.dtype)
        return jax.lax.dynamic_update_slice_in_dim(
            output, output_block, query_start, axis=1
        )

    return jax.lax.fori_loop(0, sequence // block_q, body, output)


def splash_mha_bhsd_blockwise(
    query: jax.Array, key: jax.Array, value: jax.Array
) -> jax.Array:
    """Sequential-batch wrapper that bounds native Tokamax reference memory."""
    return jax.lax.map(
        lambda inputs: splash_mha_hsd_blockwise(*inputs),
        (query, key, value),
    )


# The common name used by the JAXBench workload.
workload = causal_bhsd
kernel = causal_bhsd


def create_inputs(
    *,
    contract: str = "causal_bhsd",
    batch: int = 1,
    heads: int = 1,
    sequence: int = 128,
    head_dim: int = 128,
    dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.key(42), 3)
    if contract == "causal_bhsd":
        shape = (batch, heads, sequence, head_dim)
    elif contract == "dense_2d":
        shape = (sequence, head_dim)
    elif contract == "splash_mha_hsd":
        shape = (heads, sequence, head_dim)
    else:
        raise ValueError(f"unknown contract: {contract}")
    return tuple(jax.random.normal(k, shape, dtype=dtype) for k in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        choices=("causal_bhsd", "dense_2d", "splash_mha_hsd"),
        default="causal_bhsd",
    )
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    inputs = create_inputs(
        contract=args.contract,
        sequence=args.sequence,
        head_dim=args.head_dim,
    )
    implementation = {
        "causal_bhsd": causal_bhsd,
        "dense_2d": dense_2d,
        "splash_mha_hsd": splash_mha_hsd,
    }[args.contract]
    compiled = jax.jit(implementation)
    start = time.perf_counter()
    output = compiled(*inputs)
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": args.contract,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
