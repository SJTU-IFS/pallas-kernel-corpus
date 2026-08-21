"""Collect MaxText's ragged attention kernel and its own JAX references.

One `pallas_call` serves three public entry points -- `ragged_mqa`,
`ragged_mha`, `ragged_gqa` -- so this is a single launch point with three
callers, not three launch points.  The audit counts `ragged_mqa`.

Flattening is one line: the file's only repo-local name is
`DEFAULT_MASK_VALUE`, a float constant from `maxtext.common.common_types`,
inlined with its defining expression rather than its evaluated value so it
stays legible as "0.7 of float32's max, negated".

Upstream ships its own `reference_mqa` / `reference_mha` / `reference_gqa` in
the same file; those go to `baseline.py`.

**Upstream validates this kernel on CPU itself.**  `tests/unit/kernels_test.py`
has a `RaggedAttentionCpuTest` that calls `ragged_mqa(..., block_size=16,
interpret=True)` at a reduced shape, beside the `@pytest.mark.tpu_only` tests at
the full one.  That matters for this corpus right now: CPU interpret mode is
upstream's own practice for this kernel, not a substitute the corpus invented
because it lacked a TPU.  It is still not a Mosaic lowering, so it does not make
the kernel *migrated* under this corpus's rule.

    python tools/flatten_ragged_mqa.py <maxtext-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


REPOSITORY = "https://github.com/AI-Hypercomputer/maxtext"
COMMIT = "ca420634a9e9e73feaacc8001f605163d2d80ea1"
UPSTREAM = "src/maxtext/kernels/attention/ragged_attention.py"
CONSTANT_SOURCE = "src/maxtext/common/common_types.py"

LOCAL_IMPORT = re.compile(
    r"^from maxtext\.common\.common_types import DEFAULT_MASK_VALUE\s*$", re.M)

#: Inserted *after* the kernel file's imports, not above them: the constant's
#: defining expression uses `np`, so hoisting it to the top of the file would
#: make the module raise NameError on import.
CONSTANT_INLINE = '''

# ---- inlined from {source}: DEFAULT_MASK_VALUE ----
# Kept as its defining expression rather than the evaluated float, so it still
# reads as "0.7 of float32's maximum, negated" -- the number itself is opaque.
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


'''

#: The references live in the kernel file; they move to `baseline.py`.
REFERENCES = ("reference_mqa", "reference_mha", "reference_gqa")

KERNEL_HEADER = '''"""Standalone MaxText ragged attention (MQA / MHA / GQA).

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}
  also inlines: {source}  (only: DEFAULT_MASK_VALUE)
  transformation: one repo-local import resolved -- `DEFAULT_MASK_VALUE`, a
    float constant, inlined with its defining expression.  Upstream's three
    `reference_*` functions were moved out to this directory's `baseline.py`
    so the kernel file holds only the kernel; nothing else was touched.

Entry points: ``ragged_mqa`` (the audited launch point), ``ragged_mha``,
``ragged_gqa``, and ``kernel`` (alias of ``ragged_mqa``).

Contract ``ragged_attention``::

    q        [batch, num_heads, head_dim]   one decode step per sequence
    k, v     [batch, seq_len, head_dim]     (mqa) or [batch, heads, seq, dim]
    lengths  i32[batch]                     valid prefix length per sequence
    ->       (out, logits_max, denominator)

"Ragged" is the `lengths` argument: each sequence attends only over its own
prefix, and the kernel **skips whole blocks past that prefix** rather than
masking them -- `compute_ragged_block_indices` rewrites the grid indices so a
finished sequence's blocks are never fetched. That is the point of the kernel,
and it is why the returned `logits_max` and `denominator` matter: they let a
caller combine partial results across a split sequence.

All three entry points share **one** `pallas_call`; they differ in how they
reshape and `vmap` around it, not in the kernel body.

The references are upstream's own `reference_mqa` / `reference_mha` /
`reference_gqa`, carried to `baseline.py`.

Native shape: MaxText's own TPU test uses batch=4, head_dim=128,
max_target_length=512, float32; its CPU test uses batch=2, head_dim=32,
seq 32, block_size=16.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_attention",
    "family": "ragged_mqa_attention",
    "launch_points": 1,
    "native_shape": "batch=4,heads=1,head_dim=128,seq=512,f32",
}}

'''

BASELINE_HEADER = '''"""MaxText's own JAX references for the ragged attention in this directory.

`reference_mqa`, `reference_mha` and `reference_gqa` are upstream's, lifted
verbatim out of the kernel file (they are defined beside the kernel there) and
placed here so this directory follows the corpus's one-baseline-per-family
shape.  Nothing is re-derived: these are the functions MaxText's own tests
compare the kernel against.

Each returns a **triple** -- output, per-head max logit, and softmax
denominator -- not just the attention output, because the kernel is designed to
be composed across sequence splits and the caller needs the running statistics
to do that.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}  (the reference_* functions)
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "backend": "jax",
    "target": "portable",
    "contracts": ("ragged_attention",),
}}

import functools

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp


# ---- inlined from {source}: DEFAULT_MASK_VALUE ----
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


'''


def _toplevel(source: str):
    return {n.name: n for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _segment(source: str, node) -> str:
    decorators = ["@" + ast.get_source_segment(source, d) for d in node.decorator_list]
    return "\n".join(decorators + [ast.get_source_segment(source, node)])


def build(checkout: Path) -> tuple[str, str]:
    text = (checkout / UPSTREAM).read_text()
    tree = ast.parse(text)
    doc = tree.body[0]
    lead = "".join(text.splitlines(keepends=True)[:doc.lineno - 1])
    body = "".join(text.splitlines(keepends=True)[doc.end_lineno:])

    body, n = LOCAL_IMPORT.subn("", body)
    if n != 1:
        raise ValueError(f"expected 1 DEFAULT_MASK_VALUE import, found {n}")

    tops = _toplevel(text)
    missing = [r for r in REFERENCES if r not in tops]
    if missing:
        raise ValueError(f"missing references: {missing}")
    reference_src = "\n\n\n".join(_segment(text, tops[r]) for r in REFERENCES)

    # Lift the references out of the kernel file.
    for name in REFERENCES:
        seg = _segment(text, tops[name])
        if seg not in body:
            raise ValueError(f"could not lift {name} out of the kernel file")
        body = body.replace(seg, "", 1)
    body = re.sub(r"\n{4,}", "\n\n\n", body)

    if "maxtext" in body:
        raise ValueError(f"unresolved: {next(l for l in body.splitlines() if 'maxtext' in l)!r}")
    if body.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {body.count('pl.pallas_call')}")
    for entry in ("def ragged_mqa(", "def ragged_mha(", "def ragged_gqa("):
        if entry not in body:
            raise ValueError(f"lost {entry}")
    for name in REFERENCES:
        if f"def {name}(" in body:
            raise ValueError(f"{name} still in the kernel file")

    # Place the constant just below the last import, since it needs `np`.
    stripped = body.strip("\n")
    imports = [n for n in ast.parse(stripped).body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not imports:
        raise ValueError("no imports found to anchor the constant against")
    cut = max(n.end_lineno for n in imports)
    lines = stripped.splitlines(keepends=True)
    with_constant = ("".join(lines[:cut])
                     + CONSTANT_INLINE.format(source=CONSTANT_SOURCE)
                     + "".join(lines[cut:]))
    kernel = (lead
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM, source=CONSTANT_SOURCE)
              + with_constant.strip("\n") + "\n\n\nkernel = ragged_mqa\n")
    baseline = (lead
                + BASELINE_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                         upstream=UPSTREAM, source=CONSTANT_SOURCE)
                + reference_src.strip("\n") + "\n")
    ast.parse(kernel); ast.parse(baseline)
    if "pallas" in baseline or "pl." in baseline.split('"""', 2)[-1]:
        raise ValueError("the baseline must not reference Pallas")
    return kernel, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="maxtext checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/attention/ragged_mqa_attention")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    kernel, baseline = build(args.checkout)
    (args.corpus_dir / "maxtext_optimized.py").write_text(kernel)
    (args.corpus_dir / "baseline.py").write_text(baseline)
    print(f"wrote maxtext_optimized.py and baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
