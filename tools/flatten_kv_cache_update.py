"""Mechanically flatten a KV-cache-update kernel into one standalone file.

Both migrated implementations are single modules that reach outside their
package for two or three trivial helpers.  This script inlines exactly those
helpers and strips the repo-local import, so the result runs without either
upstream package installed.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


SOURCES = {
    "tpu_inference": {
        "display": "vLLM tpu-inference",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/ragged_paged_attention/v2/ragged_kv_cache_update.py",
        "kernel_file": "ragged_kv_cache_update.py",
        "extract": (
            (
                "utils.py",
                "tpu_inference/utils.py",
                ("get_dtype_packing",),
                ("TPU_HEAD_SIZE_ALIGNMENT",),
            ),
        ),
        "local_imports": (
            re.compile(r"^from tpu_inference\.utils import .*$", re.M),
        ),
        "contract_note": (
            "Unsharded when mesh is None; pass mesh and kv_cache_pspec for the\n"
            "    shard_map path."
        ),
    },
    "sglang_jax": {
        "display": "sglang-jax",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/update_kv_cache/update_kv_cache.py",
        "kernel_file": "update_kv_cache.py",
        "extract": (
            ("common_utils.py", "sgl_jax/srt/utils/common_utils.py", ("cdiv",), ()),
        ),
        "local_imports": (
            re.compile(r"^from sgl_jax\.srt\.utils import .*$", re.M),
        ),
        "contract_note": (
            "Always wrapped in jax.shard_map, so an active mesh is required\n"
            "    even on one device.  The sibling tuned_block_sizes.py is not\n"
            "    imported by this kernel and is not flattened in."
        ),
    },
}

HEADER = '''"""Standalone {display} paged KV-cache update (kv_cache_update contract).

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
{extracted_list}  transformation: the two or three repo-local helpers this module imports were
    inlined and the repo-local import removed. The kernel body itself is
    unmodified.
    {contract_note}

Entry point: ``kv_cache_update`` (also exported as ``kernel``).

Contract ``kv_cache_update``::

    new_kv     [total_num_tokens, num_combined_kv_heads, head_dim]
    slices     [3, padded_num_slices] int32, rows are
               (kv_cache_start, new_kv_start, slice_len)
    kv_cache   [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    num_slices [1] int32, how many columns of `slices` are live
    ->         kv_cache with, for every i < num_slices[0],
               kv_cache[kv_cache_start_i : +len_i] = new_kv[new_kv_start_i : +len_i]

Columns of ``slices`` at or beyond ``num_slices[0]`` are ignored. The kernel
donates ``kv_cache``: the input buffer is consumed, so reusing it after a call
raises "Array has been deleted".
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "kv_cache_update",
}}

'''

FOOTER = '''

kernel = kv_cache_update
'''


def extract(path: Path, functions: tuple[str, ...], names: tuple[str, ...]) -> str:
    """Return the source of exactly the named functions and assignments."""
    text = path.read_text()
    tree = ast.parse(text)
    pieces: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            pieces.append(ast.get_source_segment(text, node))
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in functions
        ):
            pieces.append(ast.get_source_segment(text, node))
    wanted = len(functions) + len(names)
    if len(pieces) != wanted:
        raise KeyError(f"{path}: extracted {len(pieces)} of {wanted}")
    return "\n\n".join(pieces)


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename, upstream, functions, names in spec["extract"]:
        source = extract(source_dir / filename, functions, names)
        label = ", ".join((*names, *functions))
        chunks.append(f"\n# ---- extracted from {upstream}: {label} ----\n\n{source}\n")

    text = (source_dir / spec["kernel_file"]).read_text()
    for pattern in spec["local_imports"]:
        text, count = pattern.subn("", text)
        if not count:
            raise ValueError(f"{name}: expected local import not found")
    chunks.append(f"\n# ---- flattened from {spec['kernel_file']} ----\n\n{text}\n")

    body = "".join(chunks)
    for token in ("sgl_jax.", "tpu_inference."):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        extracted_list="".join(
            f"  also inlines: {upstream}  (only: "
            f"{', '.join((*names, *functions))})\n"
            for _, upstream, functions, names in spec["extract"]
        ),
        contract_note=spec["contract_note"],
    )
    return header + body + FOOTER


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
