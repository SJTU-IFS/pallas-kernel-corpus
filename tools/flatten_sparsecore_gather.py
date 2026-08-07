"""Attach corpus provenance to the SparseCore ragged gather kernels.

Every migrated implementation is self-contained upstream -- these files import
nothing outside ``jax`` -- so this script copies each one and prepends a
``SOURCE`` block and header.  No flattening or import rewriting is needed.

These are the corpus's first **SparseCore** kernels: they run on the v6e's
SparseCores via ``jax.experimental.pallas.tpu_sc`` rather than its TensorCore.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from flatten_gdn import rename_top_level  # noqa: E402


SOURCES = {
    "tokamax_gather": {
        "display": "Tokamax",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/ragged_gather/pallas_mosaic_tpu_kernel.py",
        "entry": "ragged_gather_pallas",
        "contract": "ragged_gather",
        "summary": (
            "Gathers rows ``x[indices]`` for the live range ``[start, end)``.\n"
            "The output is padded up to the SparseCore block size, so only the\n"
            "first ``end - start`` rows and the first ``x.shape[-1]`` columns\n"
            "carry data."
        ),
    },
    "tokamax_gather_v2": {
        "display": "Tokamax (v2)",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/ragged_gather/pallas_mosaic_v2_tpu_kernel.py",
        "entry": "ragged_gather_pallas",
        "contract": "ragged_gather",
        "summary": (
            "The v2 kernel for the same contract as the v1 file beside it;\n"
            "both were measured and agree bit-exactly with ``x[indices]``."
        ),
    },
    "tokamax_gather_reduce": {
        "display": "Tokamax",
        "repository": "https://github.com/openxla/tokamax",
        "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "path": "tokamax/_src/ops/ragged_gather_reduce/pallas_mosaic_tpu_kernel.py",
        "entry": "ragged_gather_reduce_pallas",
        "contract": "ragged_gather_reduce",
        "summary": (
            "The MoE combine step: gather ``x[indices]``, scale each row by\n"
            "``topk_weights``, then sum consecutive groups of\n"
            "``reduce_group_size`` rows. Rows masked out by\n"
            "``valid_rows_mask`` contribute nothing."
        ),
    },
    "maxtext_gather_reduce": {
        "display": "MaxText",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/ragged/ragged_gather_reduce.py",
        "entry": "ragged_gather_reduce",
        "contract": "ragged_gather_reduce",
        "summary": (
            "MaxText's v1 MoE combine. It hard-codes 8 column partitions and\n"
            "derives the row partitions from the SparseCore geometry, which\n"
            "gives it a SHAPE FLOOR: at ``indices.shape[0] < 1024`` it returns\n"
            "a wrong answer or halts the SparseCore. See the ledger."
        ),
    },
    "maxtext_gather_reduce_v2": {
        "display": "MaxText (v2)",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/ragged/ragged_gather_reduce_v2.py",
        "entry": "ragged_gather_reduce",
        "contract": "ragged_gather_reduce",
        "summary": (
            "The v2 rewrite of the file beside it: it computes the column and\n"
            "row partitioning from a cost model instead of hard-coding it, and\n"
            "asserts ``num_row_partitions <= num_simd_lanes``.  That assert is\n"
            "what a narrow ``hidden_size`` trips -- it needs >= 2048 here."
        ),
    },
    "maxtext_sc_gather_reduce": {
        "display": "MaxText (gather_reduce_pallas)",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/gather_reduce_pallas.py",
        "entry": "sc_gather_reduce",
        "contract": "sc_gather_reduce",
        "summary": (
            "A third gather-reduce shape entirely: positional ``op``/``idx``,\n"
            "keyword-only ``reduce_group_size``, NO ``valid_rows_mask``, and\n"
            "explicit ``row_chunk_size``/``col_chunk_size`` tiling.  bf16 only,\n"
            "despite an error message that says otherwise."
        ),
    },
    "tpu_inference_dense_gather_reduce": {
        "display": "vLLM tpu-inference (dense)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/sparse_core/dense_gather_reduce.py",
        "entry": "dense_gather_reduce",
        "contract": "dense_gather_reduce",
        "summary": (
            "The dense counterpart of the ragged combine: no\n"
            "``valid_rows_mask``, and ``topk_weights`` is **2-D**\n"
            "``[tokens, reduce_group_size]`` rather than one weight per gathered\n"
            "row.  It checks ``is_compatible`` itself and falls back to JAX."
        ),
    },
    "tpu_inference_gather_v2": {
        "display": "vLLM tpu-inference (v2)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/sparse_core/ragged_gather_v2.py",
        "entry": "ragged_gather_v2",
        "contract": "ragged_gather",
        "inline_helper": True,
        "summary": (
            "tpu-inference's gather for the same contract as Tokamax's, and\n"
            "bit-exact against ``x[indices]`` like them.  It reaches for one\n"
            "helper outside its module -- ``core_map_helper.kernel``, a thin\n"
            "``pl.core_map`` wrapper -- which is inlined here."
        ),
    },
    "tpu_inference_gather_reduce_v2": {
        "display": "vLLM tpu-inference (v2)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/sparse_core/ragged_gather_reduce_v2.py",
        "entry": "ragged_gather_reduce",
        "contract": "ragged_gather_reduce",
        "inline_helper": True,
        "summary": (
            "The MoE combine, sharing 12 of 14 definitions with MaxText's v2\n"
            "file (85% AST-identical) -- a vendored pair that has since\n"
            "diverged.  Same ``core_map_helper.kernel`` inlined."
        ),
    },
    "maxtext_gather": {
        "display": "MaxText",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/ragged/ragged_gather.py",
        "entry": "ragged_gather",
        "contract": "ragged_gather",
        "summary": (
            "Same core contract as Tokamax's gather, plus an optional\n"
            "``weights`` argument: with ``has_weights=True`` each gathered row\n"
            "is scaled, giving ``weights[:, None] * x[indices]``. It also\n"
            "exposes ``enforce_fallback`` to take the non-SparseCore path."
        ),
    },
}

HEADER = '''"""Standalone {display} SparseCore ragged gather kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: copied as a standalone implementation with source metadata
    added; nothing else changed. The file imports nothing outside jax.

Entry point: ``{entry}`` (also exported as ``kernel``).

Contract ``{contract}``.
{summary}

This is a **SparseCore** kernel: it launches through ``pl.kernel`` over a
``plsc.VectorSubcoreMesh`` and runs on the v6e's SparseCores, not its
TensorCore. ``pltpu.get_tpu_info().sparse_core`` reports 2 cores, 16 subcores
and 8 lanes on this device.

The kernel checks for SparseCore availability itself and falls back to plain
XLA indexing when there is none, so it stays callable on hardware without it.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu-sparsecore",
    "target": "tpu",
    "contract": "{contract}",
}}

'''


#: `ragged_gather_v2` and `ragged_gather_reduce_v2` are the only files here that
#: are not self-contained: each calls `core_map_helper.kernel`, a thin
#: `pl.core_map` wrapper living beside them.  It is inlined rather than left as
#: an import.  Its name is `kernel`, which would collide with the corpus's own
#: `kernel = <entry>` export, so it is renamed -- via the AST, so the word
#: "kernel" in prose is untouched.
HELPER_FILE = "core_map_helper.py"
HELPER_RENAMES = {"kernel": "core_map_kernel"}
HELPER_IMPORT = re.compile(
    r"^from tpu_inference\.kernels\.sparse_core import core_map_helper\s*$", re.M
)


def build(name: str, source_dir: Path, filename: str) -> str:
    spec = SOURCES[name]
    text = (source_dir / filename).read_text()
    # The upstream file may carry its own __future__ import; the generated
    # header adds one, and two is a SyntaxError.
    text = text.replace("from __future__ import annotations\n", "", 1)

    if spec.get("inline_helper"):
        helper = (source_dir / HELPER_FILE).read_text()
        helper = helper.replace("from __future__ import annotations\n", "", 1)
        helper = re.sub(r"\A(?:#[^\n]*\n)+", "", helper)
        helper = rename_top_level(helper, HELPER_RENAMES)
        text, count = HELPER_IMPORT.subn("", text)
        if not count:
            raise ValueError(f"{name}: expected core_map_helper import not found")
        text = rename_top_level(
            re.sub(r"\bcore_map_helper\.kernel\b", "core_map_kernel", text), {}
        )
        text = (f"\n# ---- inlined from {HELPER_FILE} "
                f"(kernel -> core_map_kernel) ----\n\n"
                f"{helper.strip()}\n\n\n# ---- {filename} ----\n\n{text}")

    # Check for real references, not text: MaxText's v1 file names
    # `maxtext.src.maxtext.kernels.gather_reduce_sc` in a docstring, which a
    # substring scan would reject.
    import ast

    roots = ("maxtext", "tokamax", "tpu_inference", "sgl_jax")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in roots:
                raise ValueError(
                    f"{name}: unresolved upstream import {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in roots:
                    raise ValueError(
                        f"{name}: unresolved upstream import {alias.name}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in roots:
                raise ValueError(
                    f"{name}: unresolved upstream reference "
                    f"{node.value.id}.{node.attr}")
    if f"def {spec['entry']}(" not in text:
        raise ValueError(f"{name}: lost entry point {spec['entry']}")

    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        entry=spec["entry"],
        contract=spec["contract"],
        summary=spec["summary"],
    )
    return header + text + f"\n\nkernel = {spec['entry']}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("filename")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build(args.name, args.source_dir, args.filename))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
