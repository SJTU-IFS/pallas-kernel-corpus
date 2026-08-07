"""Flatten the three fused expert-parallel MoE kernels into the corpus.

Three implementations, one per launch point, and they are **not** the same
kernel wearing different names -- comparing the two v1 files by AST, only 3 of
7 shared top-level definitions are identical (`align_to`, `broadcast_minor`,
`swigluoai`), and the three that matter -- `_fused_ep_moe_kernel`,
`fused_ep_moe` and `ref_moe` -- all differ.  Their *signatures* differ too, so
they get separate contracts rather than one:

``sglang-jax v1/v2``  take `w1`, `w2`, `w3` as three separate weights and
                      **pre-computed** `topk_weights`/`topk_ids`
``tpu-inference v1``  takes `w1` with gate and up fused on axis 1, shape
                      `(E, 2, H, I)`, and raw `gating_output`, doing its own
                      routing

Flattening is nearly free: the two sglang files import only `jax` and stdlib,
and tpu-inference's has a single repo-local import (`get_tuned_block_sizes`).

The references cost nothing either, and that is the point of this family.  Each
upstream file already contains its own pure-JAX `ref_moe` beside the kernel, and
each needs only two or three small helpers -- so `baseline.py` is built by
extracting `ref_moe` and its transitive dependencies from all three files, with
per-source renaming since all three call theirs `ref_moe`.  Nothing about MoE
routing is re-derived here: not the grouped top-k, not the renormalisation, not
the sub-channel dequantisation.  Given how many ways there are to write "MoE",
a corpus-authored reference would be a coin flip.

    python tools/flatten_fused_moe.py kernels <pinned-root> <corpus-dir>
    python tools/flatten_fused_moe.py baseline <pinned-root> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from flatten_gdn import rename_top_level


SOURCES = {
    "sglang_jax": {
        "display": "sglang-jax (v1)",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py",
        "inputs_from": "python/sgl_jax/test/kernels/fused_moe_v1_test.py",
        "entry": "fused_ep_moe",
        "contract": "fused_ep_moe_split_weights",
        "file": "sglang_jax_optimized.py",
        "local_imports": (),
        "inline": "",
        "summary": (
            "Takes the three expert weights separately and pre-computed\n"
            "routing: `topk_weights` and `topk_ids` are the caller's job\n"
            "(upstream's own `TopK` layer produces them), so the kernel never\n"
            "sees `gating_output`.  `ref_moe` below *does* take raw gating and\n"
            "routes internally, which is why the corpus test derives top-k the\n"
            "way `ref_moe` does rather than guessing -- see\n"
            "tests/test_fused_moe_tpu.py."
        ),
    },
    "sglang_jax_v2": {
        "display": "sglang-jax (v2)",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py",
        # v2 defines its own generator, and it is not v1's: it returns eight
        # arrays rather than eleven (no biases) and defaults to a smaller
        # shape.  v2's `ref_moe` also takes pre-computed routing rather than
        # raw gating, unlike v1's.
        "inputs_from": "python/sgl_jax/test/kernels/fused_moe_v2_test.py",
        "entry": "fused_ep_moe_v2",
        "contract": "fused_ep_moe_split_weights_v2",
        "file": "sglang_jax_v2_optimized.py",
        "local_imports": (),
        "inline": "",
        "summary": (
            "Same operand layout as v1 -- three separate weights, routing\n"
            "supplied by the caller -- but a different kernel: it adds a\n"
            "SwiGLU clamp limit, block-wise fp8, and a compaction pass that\n"
            "skips empty experts.  Upstream keeps both, so the corpus does."
        ),
    },
    "tpu_inference": {
        "display": "vLLM tpu-inference (v1)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/fused_moe/v1/kernel.py",
        "inputs_from": "tests/kernels/fused_moe_v1_test.py",
        "entry": "fused_ep_moe",
        "contract": "fused_ep_moe_fused_w1",
        "file": "tpu_inference_optimized.py",
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.fused_moe\.v1\.tuned_block_sizes"
                r" import \\\n    get_tuned_block_sizes\s*$",
                re.M,
            ),
        ),
        "inline": "tuned_block_sizes.py",
        "summary": (
            "Takes `w1` with the gate and up projections **fused on axis 1** --\n"
            "shape `(num_experts, 2, hidden, intermediate)`, not two arrays --\n"
            "and raw `gating_output`, doing its own top-k and scoring inside.\n"
            "Neither of those is true of the sglang kernels next to it, so the\n"
            "two contracts do not substitute for each other."
        ),
    },
}

#: The one module tpu-inference's kernel imports, minus its logger.
INLINE_HEADER = """# ---- inlined from {path} ----
# The kernel calls `get_tuned_block_sizes` to pick eight block sizes from a
# lookup table keyed by shape; the table is data, and the fallback beside it is
# what runs for any shape upstream has not tuned.  Carried whole rather than
# summarised, since "the block sizes the kernel actually uses" is not something
# a reader should have to reconstruct.

"""

KERNEL_HEADER = '''"""Standalone {display} fused expert-parallel MoE kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
{inline_note}  transformation: {transformation}

Entry point: ``{entry}`` (also exported as ``kernel``).

Contract ``{contract}``.
{summary}

The reference is this file's own ``ref_moe``, which upstream ships beside the
kernel and its own tests compare against; it is carried into this directory's
`baseline.py` as ``{reference}`` so all three implementations' references sit
together.  It is left in place here too, because removing it would be an edit
to upstream's file.

Expert parallelism is the shape of this kernel: experts are sharded over the
mesh, tokens are dispatched to the devices holding their experts, and the
results are collected back.  On a one-device mesh that dispatch is local, so
this corpus validates the routing, the fused activation and the blocking, but
not the cross-device collective.

Native shape: none declared; upstream's tests parameterise
num_experts=128, hidden=1024, intermediate=1024, top_k=8, 256 tokens, bf16.
"""
{future}
SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "family": "fused_moe",
    "launch_points": 1,
    "native_shape": None,
}}

'''

BASELINE_HEADER = '''"""JAX references for the fused MoE kernels in this directory.

Every function here is upstream's own.  All three implementations ship a
pure-JAX ``ref_moe`` in the same file as their kernel, and all three of
upstream's test suites compare against it; this module extracts each one, with
its two or three helpers, and renames them per source because all three are
called ``ref_moe``.

Carrying them rather than writing one matters more here than almost anywhere
else in this corpus.  "Mixture of experts" does not name a single function: the
routing can be plain top-k or grouped top-k, the logits can be softmaxed or
sigmoided or used raw, the top-k weights can be renormalised or not, the
activation can be silu / gelu / clamped SwiGLU, the weights can be sub-channel
or per-channel quantised, and a shared expert may or may not be added on top.
Upstream's three files disagree with each other on several of those.  A
corpus-authored reference would be picking one combination and calling it "the"
MoE.

Source:
{sources}
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contracts": ("fused_ep_moe_split_weights", "fused_ep_moe_split_weights_v2",
                  "fused_ep_moe_fused_w1"),
    "sources": {source_list!r},
}}

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

'''


def _toplevel(source: str) -> dict[str, ast.AST]:
    return {node.name: node for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def _closure(source: str, root: str) -> list[str]:
    """`root` plus every top-level definition it transitively references."""
    tops = _toplevel(source)
    if root not in tops:
        raise ValueError(f"{root} is not a top-level definition")
    need, seen = {root}, set()
    while need - seen:
        name = (need - seen).pop()
        seen.add(name)
        if name in tops:
            need |= {n.id for n in ast.walk(tops[name])
                     if isinstance(n, ast.Name)} & set(tops)
    # File order, so definitions precede their use.
    return [n for n in tops if n in seen]


def flatten_kernel(name: str, pinned: Path) -> str:
    spec = SOURCES[name]
    root = pinned / ("sglang-jax" if "sglang" in spec["repository"]
                     else "tpu-inference")
    text = (root / spec["path"]).read_text()
    tree = ast.parse(text)
    node = tree.body[0]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        license_header = "".join(text.splitlines(keepends=True)[:node.lineno - 1])
        body = "".join(text.splitlines(keepends=True)[node.end_lineno:])
    else:
        license_header, body = "", text

    inline = ""
    for pattern in spec["local_imports"]:
        body, count = pattern.subn("", body)
        if count != 1:
            raise ValueError(f"{name}: expected 1 match for {pattern.pattern!r}")
    if spec["inline"]:
        inline_path = (root / spec["path"]).parent / spec["inline"]
        inline_text = inline_path.read_text()
        inline_tree = ast.parse(inline_text)
        first = inline_tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            inline_text = "".join(
                inline_text.splitlines(keepends=True)[first.end_lineno:])
        inline_text = re.sub(r"^from tpu_inference\.logger import init_logger\s*$",
                             "", inline_text, flags=re.M)
        inline_text = inline_text.replace("logger = init_logger(__name__)",
                                          "logger = logging.getLogger(__name__)")
        inline_text = re.sub(r"^from __future__ import .*$", "", inline_text,
                             flags=re.M)
        inline = (INLINE_HEADER.format(
            path=str(Path(spec["path"]).parent / spec["inline"]))
            + inline_text.strip("\n") + "\n\n")

    body = body.strip("\n")
    for token in ("tpu_inference.", "sgl_jax."):
        if token in body or token in inline:
            offender = next(l for l in (body + inline).splitlines() if token in l)
            raise ValueError(f"{name}: unresolved reference: {offender!r}")
    launches = body.count("pl.pallas_call") + body.count("pl.kernel(")
    if launches != 1:
        raise ValueError(f"{name}: expected 1 launch, found {launches}")
    for needed in (f"def {spec['entry']}(", "def ref_moe("):
        if needed not in body:
            raise ValueError(f"{name}: lost {needed}")

    # A `__future__` import must precede every statement, and the header below
    # opens with a `SOURCE` dict, so it is hoisted rather than left in place.
    future = ""
    if re.search(r"^from __future__ import annotations$", body, re.M):
        body = re.sub(r"^from __future__ import annotations$", "", body, flags=re.M)
        future = "\nfrom __future__ import annotations\n"

    header = KERNEL_HEADER.format(
        display=spec["display"], repository=spec["repository"],
        commit=spec["commit"], path=spec["path"], entry=spec["entry"],
        contract=spec["contract"], summary=spec["summary"],
        reference=f"ref_moe_{name}", future=future,
        inline_note=(f"  also inlines: "
                     f"{Path(spec['path']).parent / spec['inline']}\n"
                     if spec["inline"] else ""),
        transformation=(
            "none; the upstream file imports only jax and the\n"
            "    standard library, so it is carried verbatim below this header "
            "with a\n    `kernel` alias appended."
            if not spec["inline"] else
            "one repo-local import resolved. The tuned block-size\n"
            "    table and its untuned fallback were inlined and the vLLM "
            "logger\n    became the stdlib one. No kernel body was touched."
        ),
    )
    extra_import = "import logging\n\n" if spec["inline"] else ""
    result = (license_header + header + extra_import + inline + body
              + f"\n\n\nkernel = {spec['entry']}\n")
    ast.parse(result)
    return result


def build_baseline(pinned: Path) -> str:
    chunks, sources = [], []
    for name, spec in SOURCES.items():
        root = pinned / ("sglang-jax" if "sglang" in spec["repository"]
                         else "tpu-inference")
        text = (root / spec["path"]).read_text()
        wanted = _closure(text, "ref_moe")
        tops = _toplevel(text)
        segment = "\n\n\n".join(
            ast.get_source_segment(text, tops[n]) for n in wanted)
        # All three call theirs `ref_moe`, and two share helper names.
        segment = rename_top_level(segment, {n: f"{n}_{name}" for n in wanted})
        if "pl." in segment or "pallas" in segment:
            raise ValueError(f"{name}: the reference must not reference Pallas")
        chunks.append(
            f"# ---- extracted from {spec['display']}: {spec['path']} ----\n"
            f"# {', '.join(wanted)}, renamed with a `_{name}` suffix.\n\n"
            + segment + "\n")
        sources.append({"repository": spec["repository"],
                        "commit": spec["commit"], "path": spec["path"]})

        # The inputs are upstream's too: `gen_moe_inputs` from the test file
        # that exercises this kernel.  It is not a neutral generator -- it
        # builds gating logits with a strictly decreasing boost so each token's
        # top-k order is unambiguous, which is what makes a routing comparison
        # meaningful rather than tie-dependent.
        if spec["inputs_from"]:
            test_text = (root / spec["inputs_from"]).read_text()
            test_tops = _toplevel(test_text)
            gen = ast.get_source_segment(test_text, test_tops["gen_moe_inputs"])
            gen = rename_top_level(gen, {"gen_moe_inputs": f"gen_moe_inputs_{name}"})
            chunks.append(
                f"# ---- extracted from {spec['display']}: "
                f"{spec['inputs_from']} ----\n"
                f"# gen_moe_inputs, renamed with a `_{name}` suffix.\n\n"
                + gen + "\n")
            sources.append({"repository": spec["repository"],
                            "commit": spec["commit"],
                            "path": spec["inputs_from"]})

    header = BASELINE_HEADER.format(
        sources="\n".join(
            f"  {s['repository']}\n    commit: {s['commit']}\n"
            f"    path: {s['path']}" for s in sources),
        source_list=sources,
    )
    footer = (
        "\n\n#: The corpus protocol: one named reference per contract.\n"
        "REFERENCES = {\n"
        + "".join(f'    "{spec["contract"]}": ref_moe_{name},\n'
                  for name, spec in SOURCES.items())
        + "}\n"
    )
    result = header + "\n\n".join(chunks) + footer
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("kernels", "baseline"))
    parser.add_argument("pinned", type=Path, help="directory of pinned checkouts")
    parser.add_argument("corpus_dir", type=Path, help="kernels/moe/fused_moe")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    if args.what == "kernels":
        for name, spec in SOURCES.items():
            (args.corpus_dir / spec["file"]).write_text(
                flatten_kernel(name, args.pinned))
        print(f"wrote {len(SOURCES)} kernels into {args.corpus_dir}")
    else:
        (args.corpus_dir / "baseline.py").write_text(build_baseline(args.pinned))
        print(f"wrote baseline.py into {args.corpus_dir}")


if __name__ == "__main__":
    main()
