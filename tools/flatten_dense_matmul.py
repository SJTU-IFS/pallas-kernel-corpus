"""Collect JAXBench's 8p_GEMM matmul kernel and its own JAX baseline.

There is no flattening to do here: `benchmark/8p_GEMM/optimized.py` imports only
`jax`, `functools` and the public Pallas namespaces, so the corpus file is the
upstream file plus a provenance header and a `kernel` alias.  What this script
exists for is the *pair* -- JAXBench ships `baseline.py` beside `optimized.py`
in the same directory, computing the same product with `jnp.dot` and the same
`create_inputs`, and that file is carried to `jaxbench_reference.py` rather than
a `jnp.dot` the corpus wrote itself.  The distinction is not pedantry: the
baseline's `create_inputs` scales `B` by 0.02, and a reference built without
that scaling would compare against different numbers.

`kernel` is bound to `workload`, not to `matmul`.  `matmul`'s `block_shape` is
keyword-only with no default, so `matmul` alone is not callable; `workload`
applies the `TUNED_PARAMS` that JAXBench's own autotuner produced, which is what
the launch point means in practice.

    python tools/flatten_dense_matmul.py <8p_GEMM-dir> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REPOSITORY = "https://github.com/AI-Hypercomputer/accelerator-agents"
COMMIT = "6b6c44293c43976032ba12d2f72d6bebeaf2394f"
UPSTREAM = "JAXBench/benchmark/8p_GEMM"

KERNEL_HEADER = '''"""Standalone JAXBench dense GEMM kernel -- Llama-3.1-70B FFN dimensions.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/optimized.py
  transformation: none to the kernel.  The upstream file imports only `jax`,
    `functools` and the public Pallas namespaces, so it is carried verbatim
    below a provenance header, with a `kernel` alias appended.

Entry points: ``matmul`` (the Pallas launch, block sizes keyword-only),
``workload`` (``matmul`` at JAXBench's autotuned `TUNED_PARAMS`), and
``kernel`` (alias of ``workload``, since ``matmul`` has no default
``block_shape`` and so cannot be called on its own).

Contract ``dense_matmul_2d``::

    x  [M, K]  bf16
    y  [K, N]  bf16
    -> [M, N]  bf16,  x @ y accumulated in float32

The kernel body is upstream JAX's own -- JAXBench copied it from
`jax.experimental.pallas.ops.tpu.matmul`, documented at
https://docs.jax.dev/en/latest/pallas/tpu/matmul.html -- wrapped as a JAXBench
workload with CONFIG / create_inputs / workload.  It is a three-dimensional
grid (M, N, K) with a float32 VMEM accumulator carried across the K axis, so
`preferred_element_type` is what keeps a bf16 product from accumulating in
bf16.

The reference is JAXBench's own `baseline.py` from the same directory, carried
in this directory as `jaxbench_reference.py`; use its `create_inputs`, since it
scales the second operand by 0.02 and unscaled inputs compare against different
numbers.

Native shape: M=8192, K=8192, N=28672, bf16 (Llama-3.1-70B hidden -> FFN).
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}/optimized.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "dense_matmul_2d",
    "family": "dense_matmul",
    "launch_points": 1,
    "native_shape": "M=8192,K=8192,N=28672,bf16",
}}

'''

KERNEL_FOOTER = """

# `matmul` takes `block_shape` keyword-only with no default; `workload` is the
# same launch at JAXBench's autotuned sizes, so it is the callable entry point.
kernel = workload
"""

REFERENCE_HEADER = '''"""JAXBench's own JAX reference for the dense GEMM in this directory.

This is `{upstream}/baseline.py` at the commit below, unmodified apart from this
header: the file JAXBench itself measures `optimized.py` against.  It is carried
rather than rewritten because `create_inputs` is part of the contract -- it
seeds with `jax.random.key(42)` and scales the second operand by 0.02, and a
reference generating unscaled operands would be comparing different numbers,
not a different implementation.

`workload` here is `jnp.dot`, which on TPU lowers to an XLA dot rather than a
Mosaic custom call, so it is a valid non-Pallas reference; a test in
tests/test_dense_matmul_tpu.py asserts that.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/baseline.py
"""

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}/baseline.py",
    "backend": "jax",
    "target": "portable",
    "contracts": ("dense_matmul_2d",),
}}

'''


def _strip_module_docstring(text: str) -> tuple[str, str]:
    """Split a module into (leading comments, body after its docstring)."""
    tree = ast.parse(text)
    docstring = tree.body[0]
    if not (isinstance(docstring, ast.Expr)
            and isinstance(docstring.value, ast.Constant)
            and isinstance(docstring.value.value, str)):
        raise ValueError("expected a module docstring to replace")
    lines = text.splitlines(keepends=True)
    # Everything before the docstring is the Apache header comment, if present.
    return "".join(lines[:docstring.lineno - 1]), "".join(lines[docstring.end_lineno:])


def flatten_kernel(source_dir: Path) -> str:
    text = (source_dir / "optimized.py").read_text()
    license_header, body = _strip_module_docstring(text)
    for token in ("JAXBench", "benchmark."):
        if token in body:
            raise ValueError(f"unresolved upstream reference: {token}")
    if body.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {body.count('pl.pallas_call')}")
    for name in ("def matmul(", "def workload(", "def matmul_kernel("):
        if name not in body:
            raise ValueError(f"lost entry point: {name}")
    result = (license_header
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM)
              + body.strip("\n") + KERNEL_FOOTER)
    ast.parse(result)
    return result


def flatten_reference(source_dir: Path) -> str:
    text = (source_dir / "baseline.py").read_text()
    license_header, body = _strip_module_docstring(text)
    if "pallas" in body:
        raise ValueError("the reference must not reference Pallas")
    result = (license_header
              + REFERENCE_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                        upstream=UPSTREAM)
              + body.strip("\n") + "\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path, help="JAXBench benchmark/8p_GEMM")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/matmul/dense_matmul")
    args = parser.parse_args()
    (args.corpus_dir / "jaxbench_optimized.py").write_text(
        flatten_kernel(args.source_dir))
    (args.corpus_dir / "jaxbench_reference.py").write_text(
        flatten_reference(args.source_dir))
    print(f"wrote jaxbench_optimized.py and jaxbench_reference.py "
          f"into {args.corpus_dir}")


if __name__ == "__main__":
    main()
