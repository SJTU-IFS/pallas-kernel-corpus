"""Attach corpus provenance to the Splash Attention kernels.

Both migrated implementations are self-contained upstream: their only non-stdlib
dependencies are ``jax`` itself, including ``mask_lib`` and ``mask_info_lib``,
which resolve inside ``jax.experimental.pallas.ops.tpu.splash_attention`` in the
pinned ``jax==0.10.2`` -- not inside either upstream repository.  So this script
copies each file and injects a ``SOURCE`` block and header; no flattening or
import rewriting is needed.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SOURCES = {
    "jaxbench": {
        "display": "JAXBench",
        "repository": "https://github.com/AI-Hypercomputer/accelerator-agents",
        "commit": "6b6c44293c43976032ba12d2f72d6bebeaf2394f",
        "path": "JAXBench/benchmark/2p_GQA_Attention/optimized.py",
        "filename": "optimized.py",
        "entry_points": (
            "``workload`` (native shape, autotuned blocks), "
            "``make_splash_mqa_single_device`` /\n"
            "``make_splash_mha_single_device`` (kernel builders), "
            "``attention_reference`` (pure JAX)"
        ),
        "extra": (
            "JAXBench's ``4p_Sparse_Attention`` is the **same kernel**: the two\n"
            "  files differ only in ``create_inputs``, ``get_flops`` and\n"
            "  ``workload`` -- 28 of their 31 top-level definitions are\n"
            "  AST-identical. Its three launch points are therefore audited but\n"
            "  not separately migrated, the same treatment given to\n"
            "  ``3p_MLA_Attention`` in the mla_attention family."
        ),
    },
    "maxtext": {
        "display": "MaxText",
        "repository": "https://github.com/AI-Hypercomputer/maxtext",
        "commit": "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "path": "src/maxtext/kernels/attention/splash_attention_kernel.py",
        "filename": "splash_attention_kernel.py",
        "entry_points": (
            "``make_splash_mqa_single_device`` / "
            "``make_splash_mha_single_device``\n(kernel builders), "
            "``attention_reference`` (pure JAX)"
        ),
        "extra": (
            "MaxText also vendors a *second* splash kernel at\n"
            "  ``tokamax_splash_attention/splash_attention_kernel.py``. That copy\n"
            "  has diverged substantially from both this one and Tokamax's own\n"
            "  (only 35% of shared definitions are AST-identical to Tokamax's),\n"
            "  and it needs three repo-local modules, so it is audited but not\n"
            "  migrated here."
        ),
    },
}

HEADER = '''"""Standalone {display} Splash Attention kernel.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}
  transformation: copied as a standalone implementation with source metadata
    added; nothing else changed. The file is self-contained on the pinned
    dependency set -- ``mask_lib`` and ``mask_info_lib`` resolve inside
    ``jax.experimental.pallas.ops.tpu.splash_attention`` in jax==0.10.2, not
    inside {display}.

  {extra}

Entry points: {entry_points}

This file carries **three** Pallas launch points -- the forward pass, the
backward dQ pass and the backward dKV pass -- which is why the corpus counts
launch points rather than files.

Splash Attention is block-sparse: a ``mask_lib`` Mask object is compiled into
block metadata ahead of the call, and the kernel visits only the blocks the
mask marks as live. Masks are built outside the jitted call, matching upstream.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "splash_attention_mha",
}}

'''


def build(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    text = (source_dir / spec["filename"]).read_text()

    # The upstream file already carries `from __future__ import annotations`;
    # the generated header adds one, and two is a SyntaxError.
    text = text.replace("from __future__ import annotations\n", "", 1)

    metadata = HEADER.format(
        display=spec["display"],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        entry_points=spec["entry_points"],
        extra=spec["extra"],
    )
    for token in ("maxtext.", "sgl_jax.", "tpu_inference."):
        if token in text:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    for entry in ("_splash_attention_forward", "_splash_attention_bwd_dkv"):
        if f"def {entry}(" not in text:
            raise ValueError(f"{name}: lost launch point {entry}")
    return metadata + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(SOURCES))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
