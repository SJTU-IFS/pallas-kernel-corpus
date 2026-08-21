"""Collect tpu-inference's ragged causal conv1d, and its reference.

One repo-local import to resolve: `strided_ldst`, an 80-line helper module whose
two functions -- `load_large_to_compact` and `store_compact_to_large` -- move a
strided window of a large VMEM buffer into a compact vector register and back.
It is inlined whole rather than trimmed to the two names, because the pair only
makes sense read together with the register-layout comments between them.

The reference lives in upstream's *test* file rather than beside the kernel, so
`reference_causal_conv1d` is carried from there into `baseline.py`. It is 64
lines of pure JAX with Python-level control flow -- it calls `int()` on
`distribution` and `query_start_loc` entries -- so it is **eager-only**, like
sglang-jax's v2 MoE reference. That is upstream's property, not something the
flattening introduced, and it is why the corpus checks this family's
Pallas-freeness structurally rather than by lowering.

This kernel is **not CPU-interpretable**: it uses eight explicit DMAs and a
semaphore, which the Pallas interpreter does not emulate. It is prepared here
and recorded UNVALIDATED until it can be run on a TPU.

    python tools/flatten_causal_conv1d.py <tpu-inference-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


REPOSITORY = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
BASE = "tpu_inference/kernels/causal_conv1d"
UPSTREAM = f"{BASE}/causal_conv1d.py"
HELPER = f"{BASE}/strided_ldst.py"
TEST = "tests/kernels/causal_conv1d_test.py"

LOCAL_IMPORT = re.compile(
    r"^from tpu_inference\.kernels\.causal_conv1d import strided_ldst\s*$", re.M)

KERNEL_HEADER = '''"""Standalone vLLM tpu-inference ragged causal conv1d.

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}
  also inlines: {helper}  (whole module)
  transformation: one repo-local import resolved. `strided_ldst` was inlined in
    full and its `strided_ldst.` qualifier dropped, since its two functions are
    now defined at top level. No kernel body was touched.

Entry point: ``ragged_causal_conv1d`` (also exported as ``kernel``).

Contract ``ragged_causal_conv1d``::

    x                 [num_tokens, dim]     tokens of every sequence, packed
    conv_state        [num_slots, k-1, dim] the carried tail of each sequence
    conv_weight       [dim, kernel_size]
    conv_bias         [dim] or None
    query_start_loc   i32[num_seqs + 1]     where each sequence starts in x
    state_indices     i32[num_seqs]         which cache slot each sequence owns
    distribution      i32[3]                decode / prefill / total split
    -> (out, new_conv_state)

The depthwise causal convolution every Mamba-style model runs before its state
update. "Ragged" is `query_start_loc`: many sequences of different lengths are
packed into one `x`, and each one convolves only over its own tokens plus the
`kernel_size - 1` it carried in `conv_state` from the previous step. Getting
that boundary right is the whole difficulty -- a token must see its own
sequence's history and never its neighbour's.

`distribution` splits the batch into decode and prefill regions, which the
kernel walks differently: a decode sequence contributes one token and reads its
whole state, a prefill sequence contributes many and rebuilds it.

Native shape: none declared; upstream's tests sweep batch configurations,
`dim`, `kernel_size` and bias, at rtol = atol = 1e-2.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "ragged_causal_conv1d",
    "family": "causal_conv1d",
    "launch_points": 1,
    "native_shape": None,
}}

'''

HELPER_BANNER = "\n# ---- inlined from {helper} ----\n"

BASELINE_HEADER = '''"""JAX reference for the ragged causal conv1d in this directory.

`reference_causal_conv1d` is upstream's own, carried verbatim from
`{test}` -- the function tpu-inference's tests compare the kernel against. It
lives in the test file rather than beside the kernel, which is why it is
extracted from there.

**It is eager-only.** The body calls `int()` on entries of `distribution` and
`query_start_loc` to decide how many sequences and tokens are real, so it does
not survive `jax.jit` with those traced. That is upstream's design -- the
reference is written to be obviously correct rather than fast, looping in Python
over every row -- and it means this module's Pallas-freeness is checked by
reading it rather than by lowering it.

Source:
  repository: {repository}
  commit: {commit}
  path: {test}
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{test}",
    "backend": "jax",
    "target": "portable",
    "contracts": ("ragged_causal_conv1d",),
}}

import jax
import jax.numpy as jnp


'''


def _unresolved_imports(code: str) -> list[str]:
    """Repo-local modules still *imported* by the code, ignoring comments."""
    found = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "tpu_inference":
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names
                      if a.name.split(".")[0] == "tpu_inference"}
        elif isinstance(node, ast.Name) and node.id == "strided_ldst":
            found.add("strided_ldst (qualifier not stripped)")
    return sorted(found)


def _after_imports_start(text: str) -> tuple[str, str]:
    """Split off any leading license comment from the code."""
    tree = ast.parse(text)
    first = tree.body[0]
    lines = text.splitlines(keepends=True)
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        return "".join(lines[:first.lineno - 1]), "".join(lines[first.end_lineno:])
    return "".join(lines[:first.lineno - 1]), "".join(lines[first.lineno - 1:])


def flatten_kernel(checkout: Path) -> str:
    text = (checkout / UPSTREAM).read_text()
    lead, body = _after_imports_start(text)

    body, n = LOCAL_IMPORT.subn("", body)
    if n != 1:
        raise ValueError(f"expected 1 strided_ldst import, found {n}")

    helper_text = (checkout / HELPER).read_text()
    _, helper_body = _after_imports_start(helper_text)
    # The helper's own imports duplicate the kernel's; drop them.
    helper_body = re.sub(r"^(import jax\n|import jax\.numpy as jnp\n"
                         r"|from jax\.experimental\.pallas import tpu as pltpu\n)",
                         "", helper_body, flags=re.M)

    used = sorted(set(re.findall(r"strided_ldst\.(\w+)", body)))
    if used != ["load_large_to_compact", "store_compact_to_large"]:
        raise ValueError(f"unexpected strided_ldst usage: {used}")
    body = body.replace("strided_ldst.", "")

    combined = (HELPER_BANNER.format(helper=HELPER) + helper_body.strip("\n")
                + "\n\n\n" + body.strip("\n"))
    # AST, not substring: the banner comment above legitimately names the
    # upstream path, and a text scan flags its own provenance note.
    leftover = _unresolved_imports(combined)
    if leftover:
        raise ValueError(f"unresolved repo-local references: {leftover}")
    if combined.count("pl.pallas_call") != 1:
        raise ValueError(f"expected 1 pallas_call, found {combined.count('pl.pallas_call')}")
    for name in ("def ragged_causal_conv1d(", "def load_large_to_compact(",
                 "def store_compact_to_large("):
        if name not in combined:
            raise ValueError(f"lost {name}")

    # The helper block has to sit *below* the kernel file's imports.
    imports = [n for n in ast.parse(body.strip("\n")).body
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    stripped = body.strip("\n")
    cut = max(n.end_lineno for n in imports)
    lines = stripped.splitlines(keepends=True)
    placed = ("".join(lines[:cut]) + "\n" + HELPER_BANNER.format(helper=HELPER)
              + helper_body.strip("\n") + "\n\n" + "".join(lines[cut:]))

    result = (lead
              + KERNEL_HEADER.format(repository=REPOSITORY, commit=COMMIT,
                                     upstream=UPSTREAM, helper=HELPER)
              + placed.strip("\n") + "\n\n\nkernel = ragged_causal_conv1d\n")
    ast.parse(result)
    return result


def build_baseline(checkout: Path) -> str:
    text = (checkout / TEST).read_text()
    tops = {n.name: n for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if "reference_causal_conv1d" not in tops:
        raise ValueError("reference_causal_conv1d not found in the test file")
    segment = ast.get_source_segment(text, tops["reference_causal_conv1d"])
    if "pl." in segment or "pallas" in segment:
        raise ValueError("the reference must not reference Pallas")
    result = (BASELINE_HEADER.format(repository=REPOSITORY, commit=COMMIT, test=TEST)
              + segment.strip("\n") + "\n")
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="tpu-inference checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/convolution/causal_conv1d")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    (args.corpus_dir / "tpu_inference_optimized.py").write_text(flatten_kernel(args.checkout))
    (args.corpus_dir / "baseline.py").write_text(build_baseline(args.checkout))
    print(f"wrote tpu_inference_optimized.py and baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
