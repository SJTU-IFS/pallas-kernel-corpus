"""Mechanically flatten the MoE router top-k kernels into standalone files.

sglang-jax ships two router top-k packages, ``biased_topk`` and
``grouped_topk``.  They are kept as two corpus files rather than merged,
because they define three colliding names (``get_interpret``, ``NEG_INF``,
``SAFE_AUTO_BT``) and concatenating them would silently shadow the first
definitions.  Upstream keeps them separate for the same reason.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshot; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


COMMIT = "a7353325e8c00d287294c2cd679a77173f1a4594"
REPOSITORY = "https://github.com/sgl-project/sglang-jax"

SOURCES = {
    "biased_topk": {
        "path": "python/sgl_jax/srt/kernels/biased_topk",
        "files": ("tuned_block_sizes.py", "v1/kernel.py"),
        "kernel_file": "v1/kernel.py",
        "launches": 2,
        "entry_points": ("topk_pallas", "biased_topk_pallas"),
        "summary": (
            "plain and bias-corrected MoE router top-k.  ``biased_topk_pallas``\n"
            "selects on ``logits + correction_bias`` but returns the *pre-bias*\n"
            "weights, which is what a router needs for the combine step."
        ),
        "contract": "router_biased_topk",
    },
    "grouped_topk": {
        "path": "python/sgl_jax/srt/kernels/grouped_topk",
        "files": ("v1/kernel.py",),
        "kernel_file": "v1/kernel.py",
        "launches": 1,
        "entry_points": ("grouped_topk_pallas",),
        "summary": (
            "DeepSeek-style hierarchical router top-k: experts are split into\n"
            "``num_expert_group`` groups, the best ``topk_group`` groups are\n"
            "kept, and ``topk`` experts are chosen within them."
        ),
        "contract": "router_grouped_topk",
    },
}

TOKAMAX = {
    "repository": "https://github.com/openxla/tokamax",
    "commit": "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
    "path": "tokamax/_src/ops/experimental/tpu/topk/pallas_mosaic_tpu_kernel.py",
}

TOKAMAX_HEADER = '''"""Standalone Tokamax SparseCore top-k kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: copied as a standalone implementation with ``from absl import
    logging`` replaced by stdlib ``logging`` -- the only edit -- plus source
    metadata. The sibling ``pallas_mosaic_tpu.py`` wrapper is not migrated: it
    needs ``pydantic`` and the Tokamax op framework, and the Pallas launch
    point is in this file.

Entry point: ``top_k`` (also exported as ``kernel``).

This is the only **SparseCore** kernel in the corpus. It uses
``jax.experimental.pallas.tpu_sc`` and runs on the v6e's SparseCores, not its
TensorCore -- ``plsc.get_sparse_core_info()`` reports 2 cores / 16 subcores /
8 lanes on this device.

Contract ``sparsecore_key_value_topk``::

    keys   [rows, n]  ranked descending
    values [rows, n]  carried alongside, not used for ranking
    k
    ->     (keys[rows, k], values[rows, k])

Unlike ``jax.lax.top_k`` this carries a payload array rather than returning
indices, so a router can pass expert ids through directly.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu-sparsecore",
    "target": "tpu",
    "contract": "sparsecore_key_value_topk",
}}

'''

#: The DeepSeek-V4 "lightning indexer" top-k, vendored into both repositories.
#: Each is a single self-contained module -- only `jax` imports -- so flattening
#: is a straight copy with provenance, as for the Tokamax MLA kernel.
STREAMINDEX = {
    "streamindex_tpu_inference": {
        "display": "vLLM tpu-inference",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/experimental/deepseek_v4/streamindex_topk.py",
    },
    "streamindex_sglang_jax": {
        "display": "sglang-jax",
        "repository": REPOSITORY,
        "commit": COMMIT,
        "path": "python/sgl_jax/srt/kernels/dsa/streamindex_topk.py",
    },
}

STREAMINDEX_HEADER = '''"""Standalone {display} StreamIndex top-k (DeepSeek-V4 lightning indexer).

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: self-contained upstream -- no repo-local imports at all, so
    this is a straight copy with source metadata added.

Entry point: ``streamindex_topk`` (also exported as ``kernel``).

This is a **retrieval** kernel, not a router: it scores every compressed KV
position against each query token and returns the indices of the best ``k``::

    q                [num_tokens, num_q_heads, head_dim]   indexer queries
    indexer_weights  [num_tokens, num_q_heads]             per-head mixing weights
    cache_kv         uint8[total_num_pages, page_size // 4, 4, width]
                         fp8_e4m3fn keys with ue8m0 block scales packed in
    seq_lens         i32[max_num_seqs]   UNCOMPRESSED kv length per sequence
    page_indices     i32[max_num_seqs * pages_per_seq]     flattened
    cu_q_lens        i32[max_num_seqs + 1]
    distribution     i32[3]   decode / prefill / mixed split, as in rpa_v3
    ->               i32[num_tokens, k]  positions in COMPRESSED space

The score is ``sum_h relu(q_h . k_s) * w_h`` -- the ReLU sits *before* the head
sum, which is what makes it a DSA lightning indexer rather than an attention
score.  ``seq_lens`` is in uncompressed units and the kernel divides by
``compression_ratio`` itself, so passing an already-divided length silently
retrieves from a prefix.

``k`` must be a multiple of 128; the kernel asserts it.  Being a selection
kernel, correctness here is **exact** -- a nearly-right index is a wrong index.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "streamindex_topk",
    "launch_points": 1,
}}

'''


LOCAL_IMPORTS = (
    re.compile(r"^from sgl_jax\.srt\.kernels\.[\w.]+ import \([^)]*\)\s*$", re.M),
    re.compile(r"^from sgl_jax\.srt\.kernels\.[\w.]+ import .*$", re.M),
    # The generated header already carries one; a second, mid-file, is a
    # SyntaxError because __future__ imports must come first.
    re.compile(r"^from __future__ import .*$", re.M),
)

HEADER = '''"""Standalone sglang-jax {name} MoE router kernel.

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
{file_list}
  transformation: the repo-local modules were flattened in dependency order and
    the repo-local imports removed. The kernel bodies are unmodified.

Entry points: {entry_points}

{summary}

All entry points return batch-major ``(weights[batch, topk], ids[batch, topk])``.
Inside the kernels the Pallas grid works transposed -- the batch dimension is
the lane dimension -- and the wrappers transpose back before returning, which is
why their internal variables are named ``weights_t``/``ids_t``.

``num_experts`` must be a multiple of 128; the kernels reject anything else.
``block_tokens="auto"`` picks the largest safe 128-aligned divisor of the batch.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
}}

'''


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename in spec["files"]:
        text = (source_dir / filename).read_text()
        for pattern in LOCAL_IMPORTS:
            text = pattern.sub("", text)
        chunks.append(f"\n# ---- flattened from {filename} ----\n\n{text}\n")
    body = "".join(chunks)
    if "sgl_jax." in body:
        raise ValueError(f"{name}: unresolved upstream reference sgl_jax.")
    for entry in spec["entry_points"]:
        if f"def {entry}(" not in body:
            raise ValueError(f"{name}: lost entry point {entry}")

    header = HEADER.format(
        name=name.replace("_", " "),
        repository=REPOSITORY,
        commit=COMMIT,
        path=spec["path"],
        file_list="\n".join(f"      {f}" for f in spec["files"]),
        entry_points=", ".join(f"``{e}``" for e in spec["entry_points"]),
        summary=spec["summary"],
        contract=spec["contract"],
    )
    footer = f"\n\nkernel = {spec['entry_points'][-1]}\n"
    return header + body + footer


def flatten_tokamax(source: Path) -> str:
    text = source.read_text()
    marker = "from absl import logging"
    if marker not in text:
        raise ValueError("expected absl logging import not found")
    # absl is not a corpus dependency; stdlib logging has the calls used here.
    text = text.replace(marker, "import logging")
    text = re.sub(r"^from __future__ import .*$", "", text, flags=re.M)
    if "def top_k(" not in text:
        raise ValueError("lost entry point top_k")
    header = TOKAMAX_HEADER.format(**TOKAMAX)
    return header + text + "\n\nkernel = top_k\n"


def flatten_streamindex(name: str, source: Path) -> str:
    """Straight copy with provenance; the file has no repo-local imports."""
    spec = STREAMINDEX[name]
    text = source.read_text()
    for token in ("tpu_inference.", "sgl_jax.", "tokamax._src"):
        if token in text:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    text = re.sub(r"^from __future__ import .*$", "", text, flags=re.M)
    if "def streamindex_topk(" not in text:
        raise ValueError(f"{name}: lost entry point streamindex_topk")
    return (STREAMINDEX_HEADER.format(**spec) + text
            + "\n\nkernel = streamindex_topk\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=[*sorted(SOURCES), *sorted(STREAMINDEX), "tokamax_sparsecore"],
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.name == "tokamax_sparsecore":
        args.output.write_text(flatten_tokamax(args.source_dir))
    elif args.name in STREAMINDEX:
        args.output.write_text(flatten_streamindex(args.name, args.source_dir))
    else:
        args.output.write_text(flatten(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
