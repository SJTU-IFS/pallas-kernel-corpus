"""Mechanically flatten the Megablox v2 grouped-matmul kernels.

MaxText and Tokamax each ship ``pallas_mosaic_tpu_v2_gmm_kernel.py`` and
``pallas_mosaic_tpu_v2_tgmm_kernel.py``, where the tgmm module imports the gmm
module as a sibling.  This script inlines the pair into one file per repository.

Unlike the v1 kernels in these repositories, **the v2 kernels do not use qwix**
at all, so no quantization framework and no ``flax`` dependency is needed.

Two mechanical edits are required and are the only changes to either body:

1. The tgmm module defines four helpers that shadow same-named helpers in the
   gmm module -- ``get_cost_estimate``, ``get_scope_name``, ``zero_out_start``
   and ``zero_out_end`` -- and the two versions genuinely differ.  The tgmm
   copies are renamed with a ``tgmm_`` prefix, along with their unqualified
   call sites inside the tgmm chunk.  Qualified references (``gmm_v2.foo``) are
   left alone by the rename so they keep resolving to the gmm versions.
2. The ``gmm_v2.`` module qualifier is then dropped, so those references bind
   to the flattened gmm definitions.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


SOURCES = {
    "maxtext": {
        "display": "MaxText",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/megablox",
        "local_import": re.compile(
            r"^from maxtext\.kernels\.megablox import "
            r"pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2$",
            re.M,
        ),
        "extra": (
            "This copy carries a ``partial_sum`` argument that Tokamax's does\n"
            "    not, for accumulating across a pipelined outer loop."
        ),
    },
    "tokamax": {
        "display": "Tokamax",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/ragged_dot",
        "local_import": re.compile(
            r"^from tokamax\._src\.ops\.ragged_dot import "
            r"pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2$",
            re.M,
        ),
        "extra": (
            "This copy carries an ``lhs_scale`` argument that MaxText's does\n"
            "    not, for pre-quantized activations, and launches through\n"
            "    ``pl.kernel`` over a TensorCore mesh rather than\n"
            "    ``pl.pallas_call``.\n\n"
            "    One compatibility fix was required: upstream constructs the\n"
            "    mesh as ``pltpu.TensorCoreMesh(axis_name=...)``, which jax\n"
            "    0.10.2 no longer exposes publicly. It is spelled\n"
            "    ``pltpu.create_tensorcore_mesh(...)``, the public factory that\n"
            "    builds the same object. Note this v6e has a single TensorCore,\n"
            "    so the MegaCore scaling the mesh exists for is inactive here."
        ),
        "replacements": (
            (
                'pltpu.TensorCoreMesh(axis_name="core")',
                'pltpu.create_tensorcore_mesh("core")',
            ),
        ),
    },
}

GMM_FILE = "pallas_mosaic_tpu_v2_gmm_kernel.py"
TGMM_FILE = "pallas_mosaic_tpu_v2_tgmm_kernel.py"
SHADOWED = ("get_cost_estimate", "get_scope_name", "zero_out_start", "zero_out_end")

#: sglang-jax and tpu-inference ship the v2 forward on its own -- one
#: self-contained module, no sibling tgmm to flatten in and no repo-local
#: imports -- so these are straight copies with provenance.
SINGLE_FILE_SOURCES = {
    "sglang_jax_v2": {
        "display": "sglang-jax",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel/gmm_v2.py",
        "extra": (
            "Forward only: this copy has no ``tgmm_v2`` beside it, unlike\n"
            "    MaxText's and Tokamax's. It carries ``rhs_scale``,\n"
            "    ``rhs_bias`` and ``maybe_quantize_lhs`` but no activation\n"
            "    fusion."
        ),
    },
    "tpu_inference_v2": {
        "display": "vLLM tpu-inference",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/megablox/gmm_v2.py",
        "extra": (
            "Forward only, like sglang-jax's, and the closest of the four to\n"
            "    Tokamax's (44% of shared definitions AST-identical). It adds\n"
            "    a ``fuse_act`` argument the other three lack, and takes a\n"
            "    per-block ``rhs_scale`` where sglang-jax takes a per-tensor\n"
            "    one."
        ),
    },
}

SINGLE_FILE_HEADER = '''"""Standalone {display} Megablox v2 grouped matmul (gmm_v2).

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: self-contained upstream -- no repo-local imports -- so this
    is a straight copy with source metadata added.

    Unlike this repository's v1 megablox kernels, the v2 kernels do **not**
    depend on qwix, so no quantization framework and no flax dependency is
    required. {extra}

Entry point: ``gmm_v2`` (also exported as ``kernel``).

Contract ``grouped_matmul_2d``::

    gmm_v2(lhs[m, k], rhs[num_groups, k, n], group_sizes[num_groups]) -> [m, n]

**All four v2 implementations in this directory are genuinely different code**,
not vendored copies: pairwise, 23-57% of shared definitions are AST-identical.
They agree bit-exactly on the contract.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "grouped_matmul_2d",
    "launch_points": 1,
}}

'''


HEADER = '''"""Standalone {display} Megablox v2 grouped matmul (gmm_v2 + tgmm_v2).

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
      {gmm_file}
      {tgmm_file}
  transformation: the tgmm module imports the gmm module as a sibling, so the
    pair is flattened into one file. The tgmm module's four helpers that shadow
    same-named gmm helpers ({shadowed}) are renamed with a ``tgmm_`` prefix
    along with their unqualified call sites; the ``gmm_v2.`` qualifier is then
    dropped so the remaining references bind to the gmm definitions. The kernel
    bodies are otherwise unmodified.

    Unlike this repository's v1 megablox kernels, the v2 kernels do **not**
    depend on qwix, so no quantization framework and no flax dependency is
    required. {extra}

Entry points:
  ``gmm_v2``   forward grouped matmul
  ``tgmm_v2``  the transposed (dW) pass
  ``kernel``   alias of ``gmm_v2``

Contract ``grouped_matmul_2d`` (same as the v1 kernels in this directory)::

    gmm_v2(lhs[m, k], rhs[num_groups, k, n], group_sizes[num_groups]) -> [m, n]

with additional optional arguments for quantization scales, bias, activation
fusion and group sharding that the v1 contract does not have.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "grouped_matmul_2d",
}}

'''

FOOTER = "\n\nkernel = gmm_v2\n"


def rename_shadowed(text: str) -> str:
    """Prefix the tgmm copies of shadowed helpers, leaving ``gmm_v2.x`` alone."""
    for name in SHADOWED:
        # Negative lookbehind on '.' so qualified sibling references survive.
        text = re.sub(rf"(?<![\w.]){name}\b", f"tgmm_{name}", text)
    return text


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    gmm = (source_dir / GMM_FILE).read_text()
    tgmm = (source_dir / TGMM_FILE).read_text()

    tgmm, count = spec["local_import"].subn("", tgmm)
    if count != 1:
        raise ValueError(f"{name}: expected exactly one sibling import, found {count}")

    referenced = set(re.findall(r"gmm_v2\.(\w+)", tgmm))
    tgmm = rename_shadowed(tgmm)
    tgmm = tgmm.replace("gmm_v2.", "")
    gmm = re.sub(r"^from __future__ import .*$", "", gmm, flags=re.M)
    tgmm = re.sub(r"^from __future__ import .*$", "", tgmm, flags=re.M)

    body = (
        f"\n# ---- flattened from {GMM_FILE} ----\n\n{gmm}\n"
        f"\n# ---- flattened from {TGMM_FILE} ----\n\n{tgmm}\n"
    )
    for old, new in spec.get("replacements", ()):
        body, count = body.replace(old, new), body.count(old)
        if not count:
            raise ValueError(f"{name}: expected replacement not found: {old!r}")
    if "qwix" in body:
        raise ValueError(f"{name}: unexpected qwix reference in a v2 kernel")
    for token in ("maxtext.", "tokamax._src"):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")

    # Every sibling reference must now bind to a top-level gmm definition.
    defined = {
        node.name
        for node in ast.parse(body).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    } | {
        target.id
        for node in ast.parse(body).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    missing = sorted(referenced - defined)
    if missing:
        raise ValueError(f"{name}: sibling references left unresolved: {missing}")
    for entry in ("gmm_v2", "tgmm_v2"):
        if f"def {entry}(" not in body:
            raise ValueError(f"{name}: lost entry point {entry}")

    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        gmm_file=GMM_FILE,
        tgmm_file=TGMM_FILE,
        shadowed=", ".join(SHADOWED),
        extra=spec["extra"],
    )
    return header + body + FOOTER


def flatten_single(name: str, source: Path) -> str:
    """Straight copy with provenance; the file has no repo-local imports."""
    spec = SINGLE_FILE_SOURCES[name]
    text = source.read_text()
    for token in ("maxtext.", "tokamax._src", "sgl_jax.", "tpu_inference."):
        if token in text:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    if "qwix" in text:
        raise ValueError(f"{name}: unexpected qwix reference in a v2 kernel")
    text = re.sub(r"^from __future__ import .*$", "", text, flags=re.M)
    if "def gmm_v2(" not in text:
        raise ValueError(f"{name}: lost entry point gmm_v2")
    if "def tgmm_v2(" in text:
        raise ValueError(f"{name}: unexpected tgmm_v2; use the paired flatten")
    return SINGLE_FILE_HEADER.format(**spec) + text + FOOTER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", choices=[*sorted(SOURCES), *sorted(SINGLE_FILE_SOURCES)])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.name in SINGLE_FILE_SOURCES:
        args.output.write_text(flatten_single(args.name, args.source_dir))
    else:
        args.output.write_text(flatten(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
