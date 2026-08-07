"""JAX references for the quantized-matmul implementations in this directory.

Both migrated files expose two entry points, so this baseline exports two
matching contracts.  They are not interchangeable: the weight scale has a
different rank and a different meaning in each.

``quantized_matmul_per_channel`` -- one scale per output channel::

    x        [n_batch, n_in]   unquantized activations
    w_q      [n_out, n_in]     quantized weights
    w_scale  [n_out]           per-output-channel scale
    ->       [n_batch, n_out]  (x_q @ w_q.T) * w_scale * x_scale

``quantized_matmul_blockwise`` -- sub-channel quantization along the
contraction axis: ``block_size`` is a scalar over ``n_in`` and ``w_scale`` has
shape ``[n_in // block_size, 1, n_out]``, so the dequantisation varies along
the contraction and the matmul cannot be issued as one dot.

Activation quantization, when requested, is symmetric and per token::

    x_scale = max(|x|, axis=-1) / dtype_max      # dtype_max of x_q_dtype
    x_q     = round-free cast of x / x_scale

Passing ``x_q_dtype=None`` (or the same dtype as ``x``) skips activation
quantization entirely and contracts the unquantized activations against the
quantized weights.  Weight zero points are not supported by either upstream
kernel, so this reference is symmetric-only too.

Neither reference uses Pallas.  Note in particular that sglang-jax's
``xla_quantized_matmul_local``, which looks like an XLA reference, dispatches
*into* the block-wise Pallas kernel and is therefore not a valid baseline.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp


SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("quantized_matmul_per_channel", "quantized_matmul_blockwise"),
}


def dtype_max(dtype: jnp.dtype) -> float:
    """Largest representable magnitude, matching the kernels' helper."""
    info = (
        jnp.finfo(dtype)
        if jnp.issubdtype(dtype, jnp.floating)
        else jnp.iinfo(dtype)
    )
    return float(info.max)


def quantize_per_token(
    x: jax.Array, x_q_dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """Symmetric per-token activation quantization.

    Mirrors the kernels' ``quantize_array``: the scale is derived from the row
    abs-max, and the reciprocal is taken with a NaN guard so an all-zero row
    does not produce inf.
    """
    x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = x_abs_max / dtype_max(x_q_dtype)
    scale_inv = jnp.nan_to_num(1.0 / scale, posinf=dtype_max(x_q_dtype))
    return (x * scale_inv).astype(x_q_dtype), scale.astype(jnp.float32)


def quantized_matmul_per_channel(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    *,
    x_q_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pure-JAX reference for ``quantized_matmul_per_channel``."""
    if x_q_dtype is None or jnp.dtype(x_q_dtype) == x.dtype:
        accumulator = jax.lax.dot_general(
            x.astype(jnp.float32),
            w_q.astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        activation_scale = None
    else:
        x_q, activation_scale = quantize_per_token(x, x_q_dtype)
        accumulator = jax.lax.dot_general(
            x_q.astype(jnp.float32),
            w_q.astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    out = accumulator * w_scale.astype(jnp.float32)
    if activation_scale is not None:
        out = out * activation_scale
    return out.astype(x.dtype)


def quantized_matmul_blockwise(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    block_size: int,
    *,
    x_q_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pure-JAX reference for ``quantized_matmul_blockwise``.

    Sub-channel quantization along the *contraction* axis only: ``block_size``
    is a scalar over ``n_in``, and ``w_scale`` has shape
    ``[n_in // block_size, 1, n_out]``.  Because the scale varies along the
    contraction axis, the contraction must be split into per-block partial
    products that are each scaled before summation -- which is exactly why the
    kernel accumulates block by block rather than issuing one matmul.
    """
    n_out, n_in = w_q.shape
    if n_in % block_size:
        raise ValueError(f"{n_in=} is not divisible by {block_size=}")
    expected = (n_in // block_size, 1, n_out)
    if w_scale.shape != expected:
        raise ValueError(f"{w_scale.shape=} should be {expected}")

    if x_q_dtype is None or jnp.dtype(x_q_dtype) == x.dtype:
        lhs, activation_scale = x.astype(jnp.float32), None
    else:
        x_q, activation_scale = quantize_per_token(x, x_q_dtype)
        lhs = x_q.astype(jnp.float32)

    out = jnp.zeros((x.shape[0], n_out), jnp.float32)
    for in_block in range(n_in // block_size):
        columns = slice(in_block * block_size, (in_block + 1) * block_size)
        partial = jax.lax.dot_general(
            lhs[:, columns],
            w_q[:, columns].astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out + partial * w_scale[in_block].astype(jnp.float32)

    if activation_scale is not None:
        out = out * activation_scale
    return out.astype(x.dtype)


def quantized_matmul_per_channel_xla(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    *,
    x_q_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Pure-XLA speed denominator for ``quantized_matmul_per_channel``.

    ``quantized_matmul_per_channel`` above contracts in fp32 so that it is an
    exact correctness reference; that makes it a poor speed target, because
    nobody serving a quantized model would upcast to fp32.  This is what a user
    would actually write without a custom kernel: dequantize the weights into
    the activation dtype and issue one ordinary matmul, accumulating in fp32.

    It is therefore the fair "no custom kernel" baseline, while the fp32
    version stays the correctness reference.  Contains no Pallas call.
    """
    if x_q_dtype is not None and jnp.dtype(x_q_dtype) != x.dtype:
        x_q, activation_scale = quantize_per_token(x, x_q_dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out * w_scale.astype(jnp.float32) * activation_scale
        return out.astype(x.dtype)

    # w_scale is per output channel, i.e. per row of w_q [n_out, n_in].
    weights = w_q.astype(x.dtype) * w_scale.astype(x.dtype)[:, None]
    return jax.lax.dot_general(
        x,
        weights,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)


kernel = quantized_matmul_per_channel
workload = quantized_matmul_per_channel


def create_inputs(
    *,
    n_batch: int = 1024,
    n_in: int = 4096,
    n_out: int = 4096,
    dtype: jnp.dtype = jnp.bfloat16,
    w_q_dtype: jnp.dtype = jnp.int8,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-channel inputs: activations, int8 weights, and one scale per row."""
    keys = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(keys[0], (n_batch, n_in), dtype=dtype)
    limit = int(dtype_max(w_q_dtype))
    w_q = jax.random.randint(
        keys[1], (n_out, n_in), -limit, limit + 1, dtype=jnp.int32
    ).astype(w_q_dtype)
    w_scale = (
        jax.random.uniform(keys[2], (n_out,), dtype=jnp.float32) * 0.01 + 0.001
    )
    return x, w_q, w_scale


def create_blockwise_inputs(
    *,
    n_batch: int = 1024,
    n_in: int = 4096,
    n_out: int = 4096,
    block_size: int = 128,
    dtype: jnp.dtype = jnp.bfloat16,
    w_q_dtype: jnp.dtype = jnp.int8,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Block-wise inputs: the scale is [n_in // block_size, 1, n_out]."""
    keys = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(keys[0], (n_batch, n_in), dtype=dtype)
    limit = int(dtype_max(w_q_dtype))
    w_q = jax.random.randint(
        keys[1], (n_out, n_in), -limit, limit + 1, dtype=jnp.int32
    ).astype(w_q_dtype)
    w_scale = (
        jax.random.uniform(
            keys[2], (n_in // block_size, 1, n_out), dtype=jnp.float32
        )
        * 0.01
        + 0.001
    )
    return x, w_q, w_scale, block_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        choices=("per_channel", "blockwise"),
        default="per_channel",
    )
    parser.add_argument("--n-batch", type=int, default=1024)
    parser.add_argument("--n-in", type=int, default=4096)
    parser.add_argument("--n-out", type=int, default=4096)
    parser.add_argument(
        "--quantize-activation",
        action="store_true",
        help="dynamically quantize activations to int8 per token",
    )
    args = parser.parse_args()

    x_q_dtype = jnp.int8 if args.quantize_activation else None
    if args.contract == "per_channel":
        inputs = create_inputs(
            n_batch=args.n_batch, n_in=args.n_in, n_out=args.n_out
        )
        compiled = jax.jit(
            quantized_matmul_per_channel, static_argnames="x_q_dtype"
        )
    else:
        x, w_q, w_scale, block_size = create_blockwise_inputs(
            n_batch=args.n_batch, n_in=args.n_in, n_out=args.n_out
        )
        inputs = (x, w_q, w_scale, block_size)
        compiled = jax.jit(
            quantized_matmul_blockwise, static_argnames=("block_size", "x_q_dtype")
        )

    start = time.perf_counter()
    output = compiled(*inputs, x_q_dtype=x_q_dtype)
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": f"quantized_matmul_{args.contract}",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "quantize_activation": args.quantize_activation,
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
