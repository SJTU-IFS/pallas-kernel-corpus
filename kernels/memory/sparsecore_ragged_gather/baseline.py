"""JAX references for the SparseCore ragged gather kernels in this directory.

These are the corpus's first **SparseCore** kernels. They launch through
``pl.kernel`` over a ``plsc.VectorSubcoreMesh`` and run on the v6e's
SparseCores rather than its TensorCore -- ``pltpu.get_tpu_info().sparse_core``
reports 2 cores, 16 subcores and 8 lanes here.

Unusually for this corpus, the references are *simple*: a ragged gather is
``x[indices]``, and each upstream kernel says so itself by falling back to
exactly that when no SparseCore is present::

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
      return x[indices]          # upstream's own fallback

So these references are not reverse-engineered; they are the semantics upstream
documents in code, and all of them were confirmed bit-exact on TPU.

## Contracts

``ragged_gather``::

    x        [num_rows, hidden]
    indices  [out_rows]              int32
    start,   end                     int32[1], the live range of `indices`
    ->       [padded_out_rows, padded_hidden]

    out[i] == x[indices[i]] for i in [start, end)

    The output is padded up to the SparseCore block size (num_lanes x
    num_cores x num_subcores) and the hidden dimension up to the kernel's
    column tile, so only the first ``end - start`` rows and the first
    ``x.shape[-1]`` columns carry data. Rows outside the live range are not
    meaningful.

``ragged_gather_weighted`` -- MaxText only, with ``has_weights=True``::

    out[i] == weights[i] * x[indices[i]]

``ragged_gather_reduce`` -- the MoE combine step::

    x                [num_rows, hidden]
    indices          [out_rows]
    topk_weights     [out_rows]
    valid_rows_mask  [out_rows]  bool
    reduce_group_size                 int
    ->               [out_rows // reduce_group_size, hidden]

    Gather, scale each row by its weight, then sum consecutive groups of
    ``reduce_group_size`` rows. Rows masked out contribute nothing.

## The gather_reduce kernel declines to run on small inputs

``ragged_gather_reduce_pallas`` has a **second** fallback the plain gathers do
not, and it is silent::

    if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
      return _fallback_implementation(...)      # plain XLA

On a v6e (128 MiB VMEM) that means fp32 ``x`` must exceed **38.4 MiB** before
the SparseCore path is used at all; upstream's judgement is that XLA already
wins when ``x`` fits comfortably in VMEM. It additionally asserts
``num_cores // num_column_partitions <= num_lanes``, which needs
``hidden >= 2048`` on this device.

This matters for measurement, not just for use: below the threshold the
"kernel" and the baseline are literally the same code, so a benchmark that
ignores the guard reports a dead heat between XLA and itself. The profiles for
this contract therefore use ``num_rows=8192, hidden=2048`` while the plain
gathers use ``num_rows=4096, hidden=1024`` -- the two contracts have different
declared shapes for this reason and not for tuning. ``tools/profile_kernel.py``
now refuses to profile any non-reference implementation that lowers to zero
``tpu_custom_call``s, and every ``result.json`` records ``pallas_launches``.

## What is not migrated

``ragged_scatter`` is **not** in the corpus. Tokamax's
``ragged_scatter_pallas`` runs on this device, but under the obvious calling
convention it returns *gather* results, not scatter ones. Measured with a
cyclic-shift permutation (chosen because, unlike a reversal, it distinguishes
the two)::

    out == x[indices]          -> True
    out[indices] == x          -> False

Its ``_preprocess_indices`` derives separate ``src_indices`` and
``dst_indices``, so the intended calling convention is probably not the one
tried here rather than the kernel being mislabelled. Until that is resolved it
would be wrong to count it as validated, so it is left out -- the same decision
made for Tokamax's SparseCore top-k in the topk_routing family.

The remaining audited launches in this family (MaxText's two
``ragged_gather_reduce`` variants and ``gather_reduce_pallas``, and
tpu-inference's four ``sparse_core`` kernels, which need a ``core_map_helper``
module) are audited but not migrated.
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
    "contracts": (
        "ragged_gather",
        "ragged_gather_weighted",
        "ragged_gather_reduce",
    ),
}


def ragged_gather(
    x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """Pure-JAX reference for ``ragged_gather``; live range only.

    Returns exactly ``end - start`` rows, unpadded. Compare against the
    kernel's ``out[start:end, : x.shape[-1]]``.
    """
    del start, end  # The live range selects rows of the result, not of `x`.
    return x[indices]


def ragged_gather_weighted(
    x: jax.Array, indices: jax.Array, weights: jax.Array
) -> jax.Array:
    """Pure-JAX reference for MaxText's weighted gather."""
    return x[indices] * weights[:, None]


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Pure-JAX reference for the MoE combine step."""
    out_rows = indices.shape[0]
    if out_rows % reduce_group_size:
        raise ValueError(f"{out_rows=} is not divisible by {reduce_group_size=}")
    gathered = x[indices] * topk_weights[:, None]
    gathered = jnp.where(valid_rows_mask[:, None], gathered, 0)
    return gathered.reshape(
        out_rows // reduce_group_size, reduce_group_size, x.shape[-1]
    ).sum(axis=1)


kernel = ragged_gather
workload = ragged_gather


def create_inputs(
    *,
    num_rows: int = 4096,
    hidden: int = 1024,
    out_rows: int = 2048,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 3,
) -> dict[str, jax.Array]:
    """Inputs for every contract in this directory."""
    keys = jax.random.split(jax.random.key(seed), 3)
    return dict(
        x=jax.random.normal(keys[0], (num_rows, hidden), dtype),
        indices=jax.random.randint(keys[1], (out_rows,), 0, num_rows, jnp.int32),
        weights=jax.random.uniform(keys[2], (out_rows,), jnp.float32),
        valid_rows_mask=jnp.ones((out_rows,), jnp.bool_),
        start=jnp.array([0], jnp.int32),
        end=jnp.array([out_rows], jnp.int32),
    )


def bytes_moved(
    *, out_rows: int, hidden: int, itemsize: int, indices_itemsize: int = 4
) -> int:
    """A gather reads one row per index and writes one; indices are read once.

    There are no FLOPs in a plain gather, so achieved bandwidth is the metric.
    """
    return out_rows * hidden * itemsize * 2 + out_rows * indices_itemsize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        choices=("gather", "weighted", "gather_reduce"),
        default="gather",
    )
    parser.add_argument("--num-rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--out-rows", type=int, default=2048)
    parser.add_argument("--reduce-group-size", type=int, default=4)
    args = parser.parse_args()

    built = create_inputs(
        num_rows=args.num_rows, hidden=args.hidden, out_rows=args.out_rows
    )
    if args.contract == "gather":
        compiled = jax.jit(ragged_gather)
        call = lambda: compiled(  # noqa: E731
            built["x"], built["indices"], built["start"], built["end"]
        )
    elif args.contract == "weighted":
        compiled = jax.jit(ragged_gather_weighted)
        call = lambda: compiled(  # noqa: E731
            built["x"], built["indices"], built["weights"]
        )
    else:
        compiled = jax.jit(ragged_gather_reduce, static_argnums=4)
        call = lambda: compiled(  # noqa: E731
            built["x"], built["indices"], built["weights"],
            built["valid_rows_mask"], args.reduce_group_size,
        )

    start = time.perf_counter()
    output = call()
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": "baseline",
                "contract": f"ragged_{args.contract}",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "bytes_moved": bytes_moved(
                    out_rows=args.out_rows, hidden=args.hidden, itemsize=4
                ),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
