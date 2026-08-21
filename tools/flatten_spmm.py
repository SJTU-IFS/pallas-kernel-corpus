"""Collect tpu-inference's structured sparse matmul.

The most self-contained kernel in this corpus: `spmm.py` imports nothing from
tpu-inference at all, so the corpus file is the upstream file with a provenance
header and a `kernel` alias.  Nothing is inlined and nothing is renamed.

It is also unusual in carrying its own *input* machinery.  A structured-sparse
matmul does not take a dense matrix -- it takes a compressed pair of `nonzeros`
and `metadata` -- and the `Sparsifier` class plus `gen_sparse_mask` that produce
that pair live in the same file as the kernel.  They are left there rather than
moved to `baseline.py`, because they are part of how the kernel is *called*, not
part of what it is checked against.

The reference is `jnp.dot` on the densified matrices, which is upstream's own
choice in `tests/kernels/spmm_v1_test.py` and is a plain mathematical identity
rather than a reading of anyone's convention -- the same standing this corpus
gives `jnp.transpose` for the layout-transpose family and `jnp.linalg.inv` for
the triangular solve.

    python tools/flatten_spmm.py <tpu-inference-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REPOSITORY = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
UPSTREAM = "tpu_inference/kernels/structured_sparse_matmul/v1/spmm.py"

KERNEL_HEADER = '''"""Standalone vLLM tpu-inference structured sparse matmul (N:M sparsity).

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}
  transformation: none.  The upstream file imports only `jax` and the standard
    library -- no repo-local names at all -- so it is carried verbatim below
    this header, with a `kernel` alias appended.

Entry points: ``structured_spmm`` (the public wrapper), ``_structured_spmm``
(the audited Pallas launch), ``Sparsifier`` and ``gen_sparse_mask`` (how the
compressed operands are built), and ``kernel`` (alias of ``structured_spmm``).

Contract ``structured_sparse_matmul``::

    sparsity   (x, y)      x nonzeros in every y elements, along sparse_dim
    nonzeros               the surviving values, densely packed
    metadata               where they came from, bit-packed
    mat                    the dense operand
    ->                     the same product a dense matmul would give

The kernel never sees the sparse matrix. It takes the **compressed** pair that
`Sparsifier` produces -- values with the zeros squeezed out, plus bit-packed
indices saying where each survivor sat -- and reconstructs the tiles it needs
inside the kernel (`_decompress_nonzeros`, `_decompress_metadata`). That is the
whole point: the zeros never occupy memory or MXU cycles.

Either operand may be the sparse one (`rhs_sparse`), the sparsity may run along
the contracting dimension or the free one (`contract_sparse`), and the right
operand may be transposed -- eight combinations, all exercised by the tests.

The reference is `jnp.dot` on the densified matrices, which is what upstream's
own test compares against.

Native shape: none declared; upstream's tests sweep shapes with hypothesis,
at stride 128 and 128x128 blocks.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "structured_sparse_matmul",
    "family": "structured_sparse_matmul",
    "launch_points": 1,
    "native_shape": None,
}}

'''

BASELINE = '''"""JAX reference for the structured sparse matmul in this directory.

There is nothing to carry from upstream here, and that is the honest position
rather than an omission. A structured-sparse matmul is defined by what it must
*equal*: the dense product of the same matrices with the pruned entries set to
the default value. `tests/kernels/spmm_v1_test.py` says exactly that --
`expected = jnp.dot(lhs, rhs, preferred_element_type=out_dtype)` after
`jnp.where(mask, x, default_val)` -- so the reference is `jnp.dot`, a plain
identity, not a reading of anyone's convention.

The interesting machinery goes the other way. Turning a dense matrix into the
`(nonzeros, metadata)` pair the kernel takes is *not* obvious, and this corpus
does not re-derive it: `Sparsifier` and `gen_sparse_mask` are upstream's and
stay in the kernel file, where the kernel that consumes them can be read beside
them.

Source:
  repository: {repository}
  commit: {commit}
  path: tests/kernels/spmm_v1_test.py  (upstream's choice of reference)
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "tests/kernels/spmm_v1_test.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("structured_sparse_matmul",),
}}

import jax
import jax.numpy as jnp


def dense_matmul(lhs: jax.Array, rhs: jax.Array, *, rhs_transpose: bool = False,
                 out_dtype=None) -> jax.Array:
    """What the sparse kernel must equal, on already-densified operands.

    `lhs` and `rhs` are the matrices *after* pruning -- that is, with the
    entries the mask dropped already replaced by the kernel's `default_value`.
    Densifying is the caller's job precisely because the kernel never sees a
    dense matrix.
    """
    if rhs_transpose:
        rhs = rhs.T
    return jnp.dot(lhs, rhs, preferred_element_type=out_dtype)


def densify(x: jax.Array, mask: jax.Array, default_value) -> jax.Array:
    """Apply a sparsity mask the way upstream's test does before comparing."""
    return jnp.where(mask, x, default_value)
'''


def flatten(checkout: Path) -> str:
    text = (checkout / UPSTREAM).read_text()
    tree = ast.parse(text)
    doc = tree.body[0]
    if not (isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)):
        raise ValueError("expected a module docstring")
    lead = "".join(text.splitlines(keepends=True)[:doc.lineno - 1])
    body = "".join(text.splitlines(keepends=True)[doc.end_lineno:])

    if "tpu_inference" in body:
        raise ValueError(
            f"unresolved: {next(l for l in body.splitlines() if 'tpu_inference' in l)!r}")
    if body.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {body.count('pl.pallas_call')}")
    for name in ("def structured_spmm(", "def _structured_spmm(",
                 "class Sparsifier:", "def gen_sparse_mask("):
        if name not in body:
            raise ValueError(f"lost {name}")

    result = (lead
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM)
              + body.strip("\n") + "\n\n\nkernel = structured_spmm\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="tpu-inference checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/matmul/structured_sparse_matmul")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    (args.corpus_dir / "tpu_inference_optimized.py").write_text(flatten(args.checkout))
    (args.corpus_dir / "baseline.py").write_text(
        BASELINE.format(repository=REPOSITORY, commit=COMMIT))
    print(f"wrote tpu_inference_optimized.py and baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
