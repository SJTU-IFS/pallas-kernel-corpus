"""Mechanically flatten a v3-contract ragged-paged-attention kernel.

vLLM tpu-inference and sglang-jax both split their v3 ragged-paged-attention
kernel across a small ``util`` module (and, for sglang-jax, a large tuned
block-size table) plus ``kernel.py``. This script inlines those modules in
dependency order and removes the repo-local imports, so the result runs without
either upstream package installed.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


SOURCES = {
    "tpu_inference_v3": {
        "display": "vLLM tpu-inference",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/ragged_paged_attention/v3",
        "files": ("util.py", "kernel.py"),
        "kernel_file": "kernel.py",
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.util import \(\n(?:.*\n)*?\s*[^\n(]*\)\s*$",
                re.M,
            ),
        ),
        "note": (
            "The sibling v3/tuned_block_sizes.py and v3/tuned_block_sizes_hd64.py\n"
            "    are not imported by this kernel; it uses its own\n"
            "    get_default_block_sizes, so they are not flattened in."
        ),
    },
    "tpu_inference_hd64": {
        "display": "vLLM tpu-inference (head-dim 64)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/ragged_paged_attention/v3",
        # Unlike the general v3 kernel this one *does* import a tuned-size
        # table, so that module is flattened in alongside util.  The table also
        # reaches outside the kernels package for one helper and for vLLM's
        # logger; the helper is extracted and the logger replaced with the
        # stdlib one, the same treatment kv_cache_update needed.
        "extract": (
            ("utils.py", "tpu_inference/utils.py", ("get_device_name",)),
        ),
        "replace": (
            ("from tpu_inference.logger import init_logger\n", ""),
            ("from tpu_inference.utils import get_device_name\n", ""),
            ("logger = init_logger(__name__)",
             "logger = logging.getLogger(__name__)"),
            # vLLM's logger adds *_once helpers the stdlib logger lacks.
            ("logger.warning_once(", "logger.warning("),
            ("logger.info_once(", "logger.info("),
        ),
        "files": ("util.py", "tuned_block_sizes_hd64.py", "kernel_hd64.py"),
        "kernel_file": "kernel_hd64.py",
        "entry": "ragged_paged_attention_hd64",
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.[\w.]+ import \(\n(?:.*\n)*?\s*[^\n(]*\)\s*$",
                re.M,
            ),
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.[\w.]+ import \\\n\s+[\w, ]+$",
                re.M,
            ),
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.[\w.]+ import [\w, ]+$",
                re.M,
            ),
        ),
        "note": (
            "A head-dim-64 specialisation of the v3 kernel, not a tuning\n"
            "    variant of it: it carries its own tuned-size table\n"
            "    (tuned_block_sizes_hd64.py) which the general v3 kernel does\n"
            "    not use."
        ),
    },
    "tpu_inference_v3_cp": {
        "display": "vLLM tpu-inference (context-parallel)",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/experimental/rpa_v3_cp",
        "contract": "rpa_v3_cp",
        # The audit records `run_rpa_kernel` -- the inner function holding the
        # pallas_call. The public API is `ragged_paged_attention`.
        "entry": "ragged_paged_attention",
        # Its only repo-local dependency is the same v3 util module the
        # non-context-parallel kernel uses, so util.py is copied in beside it.
        "files": ("util.py", "kernel.py"),
        "kernel_file": "kernel.py",
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.ragged_paged_attention\.v3\.util import \(\n(?:.*\n)*?\s*[^\n(]*\)\s*$",
                re.M,
            ),
        ),
        "note": (
            "The context-parallel variant of the v3 kernel: it shards the KV\n"
            "    sequence across devices, so its contract carries the extra\n"
            "    per-shard bookkeeping the single-device kernel does not."
        ),
    },
    "sglang_jax_v2": {
        "display": "sglang-jax (v2)",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/ragged_paged_attention",
        # NOT rpa_v2 despite the filename: the same 10-parameter signature as
        # the v3 file beside it, plus a `custom_mask` no other migrated
        # contract has, and a **4-D** fused KV cache
        # (pages, page_size, 2*num_kv_heads, head_dim) where v3 uses the 5-D
        # packed layout.  Hence its own contract name.
        "contract": "rpa_sglang_fused_4d",
        "entry": "ragged_paged_attention",
        # Same two out-of-package helpers the v3 table needs; `get_simplified_key`
        # is not extracted here because tuned_block_sizes.py *is* flattened in
        # whole, being this kernel's own table rather than a sibling's.
        "extract": (
            ("_jax_utils.py", "sgl_jax/srt/utils/jax_utils.py", ("get_device_name",)),
            (
                "_common_utils.py",
                "sgl_jax/srt/utils/common_utils.py",
                ("next_power_of_2",),
            ),
        ),
        "files": ("util.py", "tuned_block_sizes.py", "ragged_paged_attention.py"),
        "kernel_file": "ragged_paged_attention.py",
        "local_imports": (
            re.compile(
                r"^from sgl_jax\.srt\.[\w.]+ import \(\n(?:.*\n)*?\)\s*$", re.M
            ),
            re.compile(r"^from sgl_jax\.srt\.[\w.]+ import [\w, ]+$", re.M),
        ),
        "note": (
            "This is the **v2** kernel and its own tuned_block_sizes.py, a\n"
            "    different contract from the v3 file beside it: sglang-jax ships\n"
            "    both and the corpus carries both."
        ),
    },
    "sglang_jax_v3": {
        "display": "sglang-jax",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/ragged_paged_attention",
        # The tuned-size table reaches outside the kernels package for three
        # small helpers.  Rather than inline two large unrelated utility
        # modules, extract exactly the functions that are reachable.
        "extract": (
            ("_jax_utils.py", "sgl_jax/srt/utils/jax_utils.py", ("get_device_name",)),
            (
                "_common_utils.py",
                "sgl_jax/srt/utils/common_utils.py",
                ("next_power_of_2",),
            ),
            (
                "tuned_block_sizes.py",
                "sgl_jax/srt/kernels/ragged_paged_attention/tuned_block_sizes.py",
                ("get_simplified_key",),
            ),
        ),
        "files": ("util.py", "tuned_block_sizes_v3.py", "ragged_paged_attention_v3.py"),
        "kernel_file": "ragged_paged_attention_v3.py",
        "local_imports": (
            re.compile(
                r"^from sgl_jax\.srt\.[\w.]+ import \(\n(?:.*\n)*?\)\s*$", re.M
            ),
            re.compile(r"^from sgl_jax\.srt\.[\w.]+ import [\w, ]+$", re.M),
        ),
        "note": (
            "The sibling tuned_block_sizes.py belongs to the older\n"
            "    ragged_paged_attention.py kernel; only the one function the v3\n"
            "    table reaches into it for is carried over."
        ),
    },
}

HEADER = '''"""Standalone {display} ragged paged attention v3 (rpa_v3 contract).

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
{file_list}
{extracted_list}  transformation: the repo-local util{extra} and kernel modules were flattened
    in dependency order and the repo-local imports removed. The kernel body
    itself is unmodified.
    {note}

Entry point: ``ragged_paged_attention`` (also exported as ``kernel``).

This is the rpa_v3 contract: K and V arrive separately to be appended to a
paged ``kv_cache``, and ``distribution`` splits the batch into decode / prefill
/ mixed regions instead of carrying a scalar sequence count. It is NOT
interchangeable with the rpa_v2 kernels in this directory -- see baseline.py.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}/{kernel_file}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
}}

'''

FOOTER = '''

kernel = {entry}
'''


def extract_functions(path: Path, names: tuple[str, ...]) -> str:
    """Return the source of exactly the named top-level functions."""
    import ast

    tree = ast.parse(path.read_text())
    found = {
        node.name: ast.get_source_segment(path.read_text(), node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{path}: could not extract {missing}")
    return "\n\n".join(found[n] for n in names)


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename, upstream, functions in spec.get("extract", ()):
        source = extract_functions(source_dir / filename, functions)
        chunks.append(
            f"\n# ---- extracted from {upstream}: "
            f"{', '.join(functions)} ----\n\n{source}\n"
        )
    replacements = spec.get("replace", ())
    for filename in spec["files"]:
        text = (source_dir / filename).read_text()
        for pattern in spec["local_imports"]:
            new_text = pattern.sub("", text)
            if new_text == text and filename == spec["kernel_file"]:
                # Only the kernel file is expected to carry local imports;
                # failing silently here would leave an unresolvable import.
                continue
            text = new_text
        for old_text, new_piece in replacements:
            text = text.replace(old_text, new_piece)
        chunks.append(f"\n# ---- flattened from {filename} ----\n\n{text}\n")
    body = "".join(chunks)
    for token in ("sgl_jax.srt", "tpu_inference."):
        if token in body:
            offender = next(
                line for line in body.splitlines() if token in line
            )
            raise ValueError(
                f"{name}: unresolved upstream import for {token}: {offender!r}"
            )
    if any("logging.getLogger" in chunk for chunk in chunks):
        body = "import logging\n\n" + body
    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        contract=spec.get("contract", "rpa_v3"),
        kernel_file=spec["kernel_file"],
        file_list="\n".join(f"      {f}" for f in spec["files"]),
        extracted_list="".join(
            f"    {upstream}  (only: {', '.join(functions)})\n"
            for _, upstream, functions in spec.get("extract", ())
        ),
        extra=", tuned-block-size" if len(spec["files"]) > 2 else "",
        note=spec["note"],
    )
    return header + body + FOOTER.format(entry=spec.get('entry', 'ragged_paged_attention'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(flatten(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
