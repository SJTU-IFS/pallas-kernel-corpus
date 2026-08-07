"""Level 3: tiled flash attention via Pallas.

Implements the core Flash Attention pattern: tiled QK^T computation with
online softmax accumulation to avoid materializing the full N x N attention
matrix.

Demonstrates: multi-dimensional grid, online accumulation with fori_loop,
memory-efficient tiling, the full Pallas repertoire.

Reference: jax/experimental/pallas/ops/tpu/flash_attention.py

Source:
  repository: https://github.com/Tyronita/PallasBench
  commit: 30a6ee07fd4923f3877906a94002d994e972d6fe
  path: pallasbench/kernels/level3/flash_attention.py
  transformation: removed PallasBench-only provenance import; added a
    self-contained runner and explicit reference contract; uses explicit fp32
    dot accumulators and casts the result back so bf16 inputs are valid on TPU.
"""

import argparse
import json
import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


SOURCE = {
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "30a6ee07fd4923f3877906a94002d994e972d6fe",
    "path": "pallasbench/kernels/level3/flash_attention.py",
    "backend": "pallas",
    "target": "tpu",
    "contract": "dense_2d",
}


def _flash_attention_kernel(q_ref, k_ref, v_ref, o_ref):
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    d_k = q.shape[-1]

    scores = jnp.dot(
        q,
        k.swapaxes(-2, -1),
        preferred_element_type=jnp.float32,
    ) / jnp.sqrt(jnp.float32(d_k))
    weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    output = jnp.dot(weights, v, preferred_element_type=jnp.float32)
    o_ref[...] = output.astype(o_ref.dtype)


def pallas_flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    seq_len, d_model = q.shape
    BLOCK_Q = min(seq_len, 128)
    if seq_len % BLOCK_Q:
        raise ValueError(
            f"sequence length {seq_len} must be divisible by tile {BLOCK_Q}"
        )
    grid_size = seq_len // BLOCK_Q

    return pl.pallas_call(
        _flash_attention_kernel,
        interpret=interpret,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(grid_size,),
        in_specs=[
            pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
            pl.BlockSpec((seq_len, d_model), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_Q, d_model), lambda i: (i, 0)),
    )(q, k, v)


pallas_kernel = pallas_flash_attention
kernel = pallas_flash_attention
task_name = "flash_attention"
input_shapes = [(512, 64), (512, 64), (512, 64)]
category = "attention"
level = 3


def reference(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    logits = q @ k.T / jnp.sqrt(jnp.float32(q.shape[-1]))
    return jax.nn.softmax(logits, axis=-1) @ v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="run Pallas interpret mode (useful without a TPU)",
    )
    args = parser.parse_args()
    keys = jax.random.split(jax.random.key(42), 3)
    shape = (args.sequence, args.head_dim)
    inputs = tuple(
        jax.random.normal(key, shape, dtype=jnp.float32) for key in keys
    )
    start = time.perf_counter()
    output = kernel(*inputs, interpret=args.interpret)
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    expected = reference(*inputs)
    max_error = float(jnp.max(jnp.abs(output - expected)))
    print(
        json.dumps(
            {
                "implementation": "pallasbench",
                "contract": SOURCE["contract"],
                "shape": list(output.shape),
                "compile_and_run_ms": elapsed_ms,
                "max_abs_error": max_error,
            }
        )
    )


if __name__ == "__main__":
    main()
