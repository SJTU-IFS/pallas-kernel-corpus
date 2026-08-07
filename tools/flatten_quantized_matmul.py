"""Mechanically flatten a quantized-matmul kernel pair into one standalone file.

vLLM tpu-inference and sglang-jax both ship the same four-module layout:
``tuned_block_sizes.py`` -> ``util.py`` -> {``kernel.py``, ``blockwise_kernel.py``}.
The two kernel modules share the helpers, so they are flattened into a single
corpus file per repository rather than duplicating the helpers twice.

Both kernel modules define a public function named ``quantized_matmul_kernel``.
Concatenating them would silently shadow the first, so the block-wise one is
renamed to ``blockwise_quantized_matmul_kernel``.  That rename is the only edit
made to either kernel body.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


SOURCES = {
    "tpu_inference": {
        "display": "vLLM tpu-inference",
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/quantized_matmul",
        "files": (
            "tuned_block_sizes.py",
            "util.py",
            "kernel.py",
            "blockwise_kernel.py",
        ),
        "local_imports": (
            re.compile(
                r"^from tpu_inference\.kernels\.quantized_matmul(\.\w+)? import \([^)]*\)\s*$",
                re.M,
            ),
            re.compile(
                r"^from tpu_inference\.kernels\.quantized_matmul(\.\w+)? import .*$",
                re.M,
            ),
            re.compile(r"^from tpu_inference\.logger import init_logger$", re.M),
        ),
        "replacements": (
            ("logger = init_logger(__name__)", "logger = logging.getLogger(__name__)"),
            # warning_once is a vLLM logger extension, not stdlib logging.
            ("logger.warning_once(", "logger.warning("),
        ),
        "extra_imports": "import logging\n",
    },
    "sglang_jax": {
        "display": "sglang-jax",
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels",
        "files": (
            "tuned_block_sizes.py",
            "util.py",
            "kernel.py",
            "blockwise_kernel.py",
        ),
        "local_imports": (
            re.compile(r"^from \. import .*$", re.M),
            re.compile(r"^from \.\w+ import \([^)]*\)\s*$", re.M),
            re.compile(r"^from \.\w+ import .*$", re.M),
        ),
        "replacements": (),
        "extra_imports": "",
    },
}

BLOCKWISE_FILE = "blockwise_kernel.py"

HEADER = '''"""Standalone {display} quantized matmul (per-channel and block-wise).

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
{file_list}
  transformation: the tuned-block-size, util, and two kernel modules were
    flattened in dependency order and the repo-local imports removed. The only
    edit to a kernel body is a rename: ``blockwise_kernel.py`` also defines
    ``quantized_matmul_kernel``, so its copy is exposed as
    ``blockwise_quantized_matmul_kernel`` to avoid shadowing the per-channel
    one. The ``util.`` module qualifier is dropped because the helpers are now
    defined at top level.

Entry points:
  ``quantized_matmul_kernel``            per-channel weight quantization
  ``blockwise_quantized_matmul_kernel``  block-wise weight quantization
  ``kernel``                             alias of the per-channel entry point

Contract ``quantized_matmul_per_channel``::

    x        [n_batch, n_in]     unquantized activations
    w_q      [n_out, n_in]       quantized weights
    w_scale  [n_out]             per-output-channel weight scale
    w_zp     must be None        asymmetric quantization is not implemented
    x_q_dtype                    if set and != x.dtype, activations are
                                 dynamically quantized per token
    ->       [n_batch, n_out]    (x_q @ w_q.T) * w_scale * x_scale

Activation quantization is symmetric and per token::

    x_scale = max(|x|, axis=-1) / dtype_max
    x_q     = x / x_scale

``block_size`` is rejected by the per-channel entry point; use the block-wise
one, whose ``w_scale`` carries the extra block dimensions.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "quantized_matmul_per_channel",
}}

{extra_imports}
'''

FOOTER = '''

kernel = quantized_matmul_kernel
'''


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename in spec["files"]:
        text = (source_dir / filename).read_text()
        for pattern in spec["local_imports"]:
            text = pattern.sub("", text)
        if filename == BLOCKWISE_FILE:
            if "def quantized_matmul_kernel(" not in text:
                raise ValueError(f"{name}: {filename} lost its public entry point")
            text = text.replace(
                "def quantized_matmul_kernel(",
                "def blockwise_quantized_matmul_kernel(",
            )
        chunks.append(f"\n# ---- flattened from {filename} ----\n\n{text}\n")

    body = "".join(chunks)
    for old, new in spec["replacements"]:
        if old not in body:
            raise ValueError(f"{name}: expected replacement text not found: {old!r}")
        body = body.replace(old, new)
    # `util.` is only ever a module qualifier here; the helpers are now local.
    body = body.replace("util.", "")
    for token in ("tpu_inference.", "sgl_jax."):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")

    header = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        file_list="\n".join(f"      {f}" for f in spec["files"]),
        extra_imports=spec["extra_imports"],
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
