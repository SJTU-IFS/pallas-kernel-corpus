"""Collect JAXBench's paged-attention kernel and its own JAX baseline.

Like `8p_GEMM`, there is no flattening to do: `benchmark/6p_Paged_Attention/
optimized.py` imports only `jax` plus
`jax.experimental.pallas.ops.tpu.paged_attention.quantization_utils`, which
resolves inside the pinned jax rather than inside JAXBench.  The corpus file is
the upstream file with a provenance header and a `kernel` alias.

**The family's other launch point is not migrated, and not because it was
skipped.**  sglang-jax's `paged_attention/paged_attention.py` is a *GPU* kernel:
it passes `plgpu.CompilerParams(num_warps=..., num_stages=...)` unconditionally,
and Mosaic's TPU lowering rule asserts `isinstance(compiler_params,
tpu_core.CompilerParams)`, so it cannot lower on TPU at all.  Its own test file
says as much -- "This test suite is designed to be executed on GPU".  The AST
audit had classified it `portable` because its import is spelled `from
jax.experimental.pallas import triton as plgpu`, which the dotted-path marker
never matched; `tools/audit_launch_points.py` now reads the parsed imports and
classifies it `gpu`, taking the corpus's TPU-compatible total from 157 to 156.

Unlike `8p_GEMM`, JAXBench's `baseline.py` here is **not** a drop-in reference.
It computes the same decode paged attention and returns the same shape, but its
inputs differ in two mechanical ways: KV pages are laid out `(total_pages,
page_size, num_kv_heads, head_dim)` rather than `(num_kv_heads, total_pages,
page_size, head_dim)`, and it takes a `cu_q_lens` the kernel has no parameter
for (one query per sequence, so it is `arange(num_seqs + 1)`).  Both are carried
as-is and reconciled in the test by an axis permutation, which is a relabelling
rather than a reading of what the kernel means.

    python tools/flatten_paged_attention.py <6p_Paged_Attention-dir> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REPOSITORY = "https://github.com/AI-Hypercomputer/accelerator-agents"
COMMIT = "6b6c44293c43976032ba12d2f72d6bebeaf2394f"
UPSTREAM = "JAXBench/benchmark/6p_Paged_Attention"

KERNEL_HEADER = '''"""Standalone JAXBench paged-attention kernel -- Llama-3.1-70B decode.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/optimized.py
  transformation: no repo-local imports to resolve -- the upstream file
    imports only `jax` and
    `jax.experimental.pallas.ops.tpu.paged_attention.quantization_utils`, which
    resolves inside the pinned jax rather than inside JAXBench.  One rename
    follows that jax: the six `pltpu.ANY` memory spaces become `pl.ANY`, the
    current spelling of the same value (`pl.ANY` prints as `any`), since
    `pltpu` no longer re-exports it.  A `kernel` alias is appended.

Entry points: ``paged_attention`` (the Pallas launch), ``workload``
(``paged_attention`` at JAXBench's autotuned `TUNED_PARAMS`), and ``kernel``
(alias of ``workload``).

Contract ``paged_attention_decode``::

    q             [B, H_q, D]                  one query token per sequence
    k_pages       [H_kv, total_pages, P, D]    paged KV cache
    v_pages       [H_kv, total_pages, P, D]
    lengths       [B]                          valid KV length per sequence
    page_indices  [B, pages_per_seq]           which pages each sequence owns
    ->            [B, H_q, D]

Decode attention over a **paged** KV cache: each sequence's keys and values are
scattered across `pages_per_seq` physical pages, and `page_indices` says which.
Grouped-query attention is the normal case here -- 64 query heads over 8 KV
heads at the native shape.

The reference is JAXBench's own `baseline.py` from the same directory, carried
in this directory under the same name.  It is **not** a drop-in: its KV
pages are `(total_pages, page_size, num_kv_heads, head_dim)` rather than the
`(num_kv_heads, total_pages, page_size, head_dim)` this kernel takes, and it
takes an extra `cu_q_lens`.  Both differences are relabellings, reconciled in
tests/test_paged_attention_tpu.py.

Native shape: B=64, H_q=64, H_kv=8, D=128, page_size=16, pages_per_seq=256
(a 4096-token cache per sequence), bf16.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}/optimized.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "paged_attention_decode",
    "family": "paged_attention",
    "launch_points": 1,
    "native_shape": "B=64,H_q=64,H_kv=8,D=128,page=16,pages_per_seq=256,bf16",
}}

'''

KERNEL_FOOTER = """

# `paged_attention` takes `pages_per_compute_block` keyword-only; `workload`
# supplies JAXBench's autotuned value, so it is the callable entry point.
kernel = workload
"""

REFERENCE_HEADER = '''"""JAXBench's own JAX reference for the paged attention in this directory.

This is `{upstream}/baseline.py` at the commit below, unmodified apart from this
header -- the file JAXBench measures `optimized.py` against.

Read its `create_inputs` before using it.  Unlike the GEMM pair in `8p_GEMM`,
this baseline is not argument-compatible with the kernel beside it:

* KV pages are `(total_pages, page_size, num_kv_heads, head_dim)`; the kernel
  wants `(num_kv_heads, total_pages, page_size, head_dim)` -- an axis
  permutation, nothing more;
* it takes `cu_q_lens`, which the kernel has no parameter for. In decode there
  is one query per sequence, so it is `arange(num_seqs + 1)`;
* it scales the generated pages by 0.02, which the kernel's own `create_inputs`
  does not.

None of those is a difference in what the two compute, and the corpus test
reconciles them explicitly rather than papering over them. `workload` here is
pure `jnp`/`jax.nn`, so it is a valid non-Pallas reference; a test asserts that.

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
    "contracts": ("paged_attention_decode",),
}}

'''


#: JAXBench pins a JAX where the "leave it wherever it is" memory space was
#: re-exported as `pltpu.ANY`.  In this corpus's jax 0.10.2 that attribute is
#: gone and the same value lives at `pl.ANY` -- `pl.ANY` prints as `any`, the
#: identical memory space.  A rename to the current spelling of the same thing,
#: not a change to what the kernel asks for.
ANY_MEMORY_SPACE = "pltpu.ANY"
#: No trailing comment: these appear inside multi-line call arguments, where
#: one would comment out the closing parenthesis.
ANY_MEMORY_SPACE_REPLACEMENT = "pl.ANY"


def _split_docstring(text: str) -> tuple[str, str]:
    tree = ast.parse(text)
    node = tree.body[0]
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
        raise ValueError("expected a module docstring to replace")
    lines = text.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]), "".join(lines[node.end_lineno:])


def flatten_kernel(source_dir: Path) -> str:
    text = (source_dir / "optimized.py").read_text()
    lead, body = _split_docstring(text)
    if "JAXBench" in body:
        raise ValueError("unresolved upstream reference")
    if body.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {body.count('pl.pallas_call')}")
    for name in ("def paged_attention(", "def workload(", "def create_inputs("):
        if name not in body:
            raise ValueError(f"lost entry point: {name}")
    occurrences = body.count(ANY_MEMORY_SPACE)
    if occurrences != 6:
        raise ValueError(
            f"expected 6 pltpu.ANY memory spaces, found {occurrences}")
    body = body.replace(ANY_MEMORY_SPACE, ANY_MEMORY_SPACE_REPLACEMENT)
    result = (lead
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM)
              + body.strip("\n") + KERNEL_FOOTER)
    ast.parse(result)
    return result


def flatten_reference(source_dir: Path) -> str:
    text = (source_dir / "baseline.py").read_text()
    lead, body = _split_docstring(text)
    if "pallas" in body:
        raise ValueError("the reference must not reference Pallas")
    result = (lead
              + REFERENCE_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                        upstream=UPSTREAM)
              + body.strip("\n") + "\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path,
                        help="JAXBench benchmark/6p_Paged_Attention")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/attention/paged_attention")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    (args.corpus_dir / "jaxbench_optimized.py").write_text(
        flatten_kernel(args.source_dir))
    # `baseline.py`, not `<source>_reference.py`: JAXBench is the only source
    # in this directory, so its baseline *is* the family's.
    (args.corpus_dir / "baseline.py").write_text(
        flatten_reference(args.source_dir))
    print(f"wrote jaxbench_optimized.py and baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
