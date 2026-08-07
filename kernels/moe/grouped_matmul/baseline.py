"""JAX references for the grouped-matmul implementations in this directory.

All three migrated implementations share one semantic contract, so unlike the
flash-attention family this directory needs only a single reference interface.

``grouped_matmul_batched_dense`` -- PRIMARY BASELINE
    Pure XLA.  When every group has the same size (the native MoE workload,
    where routing is balanced), the ragged problem collapses to a batched dense
    matmul: reshape lhs to [num_groups, rows_per_group, k] and contract against
    rhs.  Exact, and it performs exactly the logical FLOP count with no
    algorithmic handicap, so it is the fair speedup denominator.

``grouped_matmul_loop``
    Pure XLA, any group sizes.  A masked accumulation over groups: correct for
    arbitrary routing but it performs num_groups times the necessary work, so
    it states the contract in code rather than being a speed target.

``grouped_matmul_ragged_dot`` -- NOT A BASELINE
    ``jax.lax.ragged_dot``.  Despite living in ``jax.lax``, on TPU this lowers
    to two Mosaic ``tpu_custom_call``s -- one building the same group metadata
    the Pallas kernels build, one doing the grouped matmul.  It is the Megablox
    kernel shipped inside JAX, so it is a fourth *implementation* to compare
    against, never the JAX denominator.  Verify with::

        jax.jit(grouped_matmul_ragged_dot).lower(*inputs).compile().as_text()

    and look for ``custom_call_target="tpu_custom_call"``.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np


#: The backward-dW contract, distinct from the forward one above: ``tgmm``
#: contracts over the m axis instead of k, and returns one matrix per group.
#: See ``grouped_matmul_transpose`` for the transpose trap between the two
#: upstream calling conventions.
TRANSPOSE_CONTRACT = "grouped_matmul_2d_transpose"

SOURCE = {
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("grouped_matmul_2d",),
}

CONTRACT = """\
grouped_matmul_2d:
  lhs         [m, k]
  rhs         [num_groups, k, n]
  group_sizes [num_groups], int32
  out         [m, n], accumulated in preferred_element_type

  Rows of lhs are partitioned into contiguous groups.  Group i owns rows
  [offsets[i], offsets[i] + group_sizes[i]) where offsets = exclusive cumsum of
  group_sizes, and those rows are multiplied by rhs[i].  Groups of size zero
  contribute nothing.

  REQUIRED: sum(group_sizes) == m.  Rows not covered by any group are outside
  this contract and the migrated implementations genuinely disagree there --
  see UNCOVERED_ROWS below.  The native MoE workload always covers every row,
  because every routed token belongs to exactly one expert.
"""

UNCOVERED_ROWS = """\
Measured on TPU v6e with m=1024, num_groups=8, group_sizes=[128]*7+[0], so rows
896..1023 belong to no group.  num_current_groups == num_total_groups there,
which leaves each implementation's `_zero_uninitialized_memory` post-pass
disabled, and the lineages then behave differently:

  jaxbench_optimized        UNINITIALISED.  The kernel never writes these rows,
                            so they hold whatever was already in the output
                            buffer: exact zeros on the first call in a fresh
                            process, stale values once the allocator has reused
                            a dirty buffer.  Verified by running the same jitted
                            call before and after other same-shaped work.
  tpu_inference_optimized   a deterministic partial product from the last
  sglang_jax_optimized      m-tile; identical cold and warm, and bit-identical
  jax.lax.ragged_dot        to each other.
  grouped_matmul_loop       exact zeros, by construction.

So this region is not merely "two conventions": for one lineage it is not a
value at all.  Never read it, never compare it, and keep correctness and
performance measurements on inputs with sum(group_sizes) == m.  Comparing
across lineages here produced cosine 0.60 -- a spurious FAIL against code that
is entirely correct on its actual domain.
"""


def check_uniform_groups(
    group_sizes: jax.Array, *, rows: int, num_groups: int
) -> None:
    """Raise unless every group holds exactly ``rows // num_groups`` rows."""
    sizes = np.asarray(group_sizes)
    if not (sizes == rows // num_groups).all():
        raise ValueError(
            "grouped_matmul_batched_dense requires equal group sizes; got "
            f"{sizes.tolist()}. Use grouped_matmul_loop for uneven routing."
        )


def grouped_matmul_batched_dense(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Pure-XLA baseline for ``grouped_matmul_2d`` with equal-sized groups.

    Requires ``group_sizes`` to be uniform, which is the native MoE workload:
    every expert receives ``rows // num_groups`` routed tokens.  Under that
    condition the ragged structure is a pure reshape, so this computes exactly
    the same FLOPs as the Pallas kernels with no ragged bookkeeping.

    ``group_sizes`` does not enter the computation.  It is verified when it is
    concrete; under ``jax.jit`` it arrives as a tracer and cannot be, so the
    caller is then responsible for passing uniform sizes.  Call this once
    outside jit, or use ``check_uniform_groups``, before trusting a jitted run.
    """
    num_groups, k, n = rhs.shape
    rows = lhs.shape[0]
    if rows % num_groups:
        raise ValueError(f"{rows=} must be divisible by {num_groups=}")
    if not isinstance(group_sizes, jax.core.Tracer):
        check_uniform_groups(group_sizes, rows=rows, num_groups=num_groups)
    blocked = lhs.reshape(num_groups, rows // num_groups, k)
    out = jnp.einsum(
        "gmk,gkn->gmn",
        blocked.astype(preferred_element_type),
        rhs.astype(preferred_element_type),
        preferred_element_type=preferred_element_type,
    )
    return out.reshape(rows, n)


def grouped_matmul_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jax.Array:
    """``jax.lax.ragged_dot``: a Mosaic kernel, NOT a JAX baseline.

    See the module docstring.  On TPU this compiles to ``tpu_custom_call``.
    """
    return jax.lax.ragged_dot(
        lhs, rhs, group_sizes, preferred_element_type=preferred_element_type
    )


def grouped_matmul_loop(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Explicit per-group reference for ``grouped_matmul_2d``.

    Written as a masked accumulation rather than as ragged slices so that it
    traces to a static shape and can be jitted at the native workload size.

    Rows covered by no group are left as exact zeros by construction.  The
    migrated kernels do not agree with that, or with each other, outside the
    covered region -- see UNCOVERED_ROWS.  All of them agree bit-exactly
    whenever ``sum(group_sizes) == m``.
    """
    rows = lhs.shape[0]
    num_groups = rhs.shape[0]
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes)]
    )
    row_index = jnp.arange(rows)

    def body(group: int, out: jax.Array) -> jax.Array:
        active = (row_index >= offsets[group]) & (row_index < offsets[group + 1])
        product = jnp.dot(
            lhs.astype(preferred_element_type),
            rhs[group].astype(preferred_element_type),
            preferred_element_type=preferred_element_type,
        )
        return out + jnp.where(active[:, None], product, 0)

    out = jnp.zeros((rows, rhs.shape[-1]), dtype=preferred_element_type)
    return jax.lax.fori_loop(0, num_groups, body, out)


def grouped_matmul_transpose(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    preferred_element_type: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Reference for ``tgmm`` -- the backward-dW pass of ``grouped_matmul_2d``.

    Per group, ``out[g] = lhs[start_g:end_g, :].T @ rhs[start_g:end_g, :]``,
    giving ``[num_groups, k, n]``.  Written as a masked accumulation for the
    same reason as ``grouped_matmul_loop``: it traces to a static shape.

    **Two calling conventions exist and differ by a transpose of ``lhs``.**
    This reference takes the megablox v2 form, ``lhs[m, k]``.  JAXBench's
    ``tgmm`` takes ``lhs`` already transposed, ``[k, m]``, so a JAXBench call
    passes ``lhs.T``.  Nothing in either signature announces the difference --
    both are 2-D float arrays -- and at ``m == k`` the mistake is undetectable.
    """
    m, k = lhs.shape
    n = rhs.shape[1]
    num_groups = group_sizes.shape[0]
    offsets = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes)]
    )
    row_index = jnp.arange(m)

    def body(group: int, out: jax.Array) -> jax.Array:
        active = (row_index >= offsets[group]) & (row_index < offsets[group + 1])
        masked = jnp.where(active[:, None], lhs, 0)
        product = jnp.dot(
            masked.astype(preferred_element_type).T,
            rhs.astype(preferred_element_type),
            preferred_element_type=preferred_element_type,
        )
        return out.at[group].set(product)

    return jax.lax.fori_loop(
        0, num_groups, body,
        jnp.zeros((num_groups, k, n), preferred_element_type),
    )


def create_transpose_inputs(
    *,
    rows: int = 512,
    num_groups: int = 8,
    k: int = 256,
    n: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Inputs for ``grouped_matmul_transpose``: lhs[m, k], grad[m, n]."""
    keys = jax.random.split(jax.random.key(seed), 2)
    lhs = jax.random.normal(keys[0], (rows, k), dtype=dtype)
    grad = jax.random.normal(keys[1], (rows, n), dtype=dtype) * 0.02
    group_sizes = jnp.full((num_groups,), rows // num_groups, jnp.int32)
    return lhs, grad, group_sizes


# The common name used by the JAXBench workload.
# The pure-XLA denominator, not jax.lax.ragged_dot.
workload = grouped_matmul_batched_dense
kernel = grouped_matmul_batched_dense


def create_inputs(
    *,
    rows: int = 1024,
    num_groups: int = 8,
    k: int = 256,
    n: int = 256,
    dtype: jnp.dtype = jnp.bfloat16,
    balanced: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.key(42), 3)
    lhs = jax.random.normal(keys[0], (rows, k), dtype=dtype)
    rhs = jax.random.normal(keys[1], (num_groups, k, n), dtype=dtype) * 0.02
    if balanced:
        group_sizes = jnp.full((num_groups,), rows // num_groups, jnp.int32)
    else:
        # Unbalanced routing, still summing to exactly `rows`.
        weights = jax.random.uniform(keys[2], (num_groups,)) + 0.1
        sizes = jnp.floor(weights / weights.sum() * rows).astype(jnp.int32)
        group_sizes = sizes.at[-1].add(rows - sizes.sum())
    return lhs, rhs, group_sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        choices=("batched_dense", "loop", "ragged_dot"),
        default="batched_dense",
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    inputs = create_inputs(
        rows=args.rows, num_groups=args.groups, k=args.k, n=args.n
    )
    implementation = {
        "batched_dense": grouped_matmul_batched_dense,
        "loop": grouped_matmul_loop,
        "ragged_dot": grouped_matmul_ragged_dot,
    }[args.reference]
    compiled = jax.jit(implementation)
    start = time.perf_counter()
    output = compiled(*inputs)
    output.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    print(
        json.dumps(
            {
                "implementation": f"baseline-{args.reference}",
                "contract": "grouped_matmul_2d",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "compile_and_run_ms": elapsed_ms,
            }
        )
    )


if __name__ == "__main__":
    main()
