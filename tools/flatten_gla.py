"""Attach corpus provenance to the gated linear attention kernels.

Gated linear attention is the same shape of idea as the gated delta net next
door -- an O(T) recurrence carrying a per-head state instead of an O(T^2)
softmax -- but with a *scalar* per-head decay rather than a rank-one delta
update. sglang-jax ships two variants, and they are separate contracts:

- **KDA** (``kernels/kda/``): a chunked forward with per-channel gates, an
  L2-normalised q/k option, and a ``beta`` delta term. Four launch points.
- **Simple GLA** (``kernels/simple_gla/``): a lighter recurrence with a single
  decay ``g_gamma`` per head. Three launch points, split across a chunked
  prefill file and a fused decode file.

As in gated_delta_net, **upstream ships its own pure-JAX references** --
``kda/naive.py`` and ``simple_gla/native.py``, both importing nothing outside
jax -- so ``baseline.py`` copies those rather than re-deriving a recurrence with
ragged sequence boundaries by hand.

The text-munging helpers are imported from ``flatten_gdn`` rather than copied:
they encode fixes for parenthesised imports and aliased sibling modules that
were found the hard way, and a second copy would drift.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from flatten_gdn import (  # noqa: E402
    sibling_module_aliases,
    strip_module_prefixes,
    strip_module_preamble,
    unbound_module_refs,
)


SGLANG = "https://github.com/sgl-project/sglang-jax"
SGLANG_COMMIT = "a7353325e8c00d287294c2cd679a77173f1a4594"
BASE = "python/sgl_jax/srt/kernels"

VARIANTS = {
    "kda": {
        "display": "sglang-jax KDA",
        "path": f"{BASE}/kda",
        # One module, so SOURCE names the file: the audit records launch points
        # by file, and `test_inventory_corpus_claims_match_the_filesystem`
        # requires the two to agree.
        "source_path": f"{BASE}/kda/kda.py",
        "modules": ("kda.py",),
        "entry": "chunk_kda_fwd",
        "contract": "kda_chunk_fwd",
        "launch_points": 4,
        "required": (
            "chunk_local_cumsum_vector",
            "kda_fwd_intra",
            "chunk_gated_delta_rule_fwd_h",
            "chunk_kda_fwd_o_gk",
            "chunk_kda_fwd",
        ),
        "summary": (
            "Chunked KDA forward. Four launch points, one per stage of the\n"
            "chunked algorithm: the per-chunk gate cumsum, the intra-chunk\n"
            "attention, the inter-chunk state recurrence, and the output\n"
            "projection with gates."
        ),
    },
    "simple-gla": {
        "display": "sglang-jax Simple GLA",
        "path": f"{BASE}/simple_gla",
        "modules": ("simple_gla.py", "simple_gla_fused.py"),
        "entry": "simple_gla_fwd",
        "contract": "simple_gla_fwd",
        "launch_points": 3,
        "required": (
            "chunk_fwd_h_kernel_varlen",
            "simple_gla_fwd",
            "chunk_simple_gla_fwd_varlen",
            "decode_simple_gla_fused",
        ),
        "summary": (
            "Simple GLA: one scalar decay `g_gamma` per head rather than KDA's\n"
            "per-channel gates. Three launch points -- two in the chunked\n"
            "prefill path and one in the fused decode path, which upstream\n"
            "keeps in a separate module that imports the first."
        ),
    },
}

HEADER = '''"""Standalone {display} kernels.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}/
  files: {files}
  transformation: {transformation}

Entry point: ``{entry}`` (also exported as ``kernel``).

Contract ``{contract}`` -- **{launch_points} Pallas launch points**.
{summary}

Validated against upstream's own pure-JAX reference, which is copied into
``baseline.py`` from the same directory.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{source_path}",
    "files": {files!r},
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "launch_points": {launch_points},
}}

from functools import singledispatch
import enum
import functools
import inspect as _inspect
import math
import os

import jax
from jax.experimental.pallas import dslice
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

'''


def build(name: str, source_dir: Path) -> str:
    """Concatenate a variant's modules into one standalone file."""
    import re

    spec = VARIANTS[name]
    stems = {module[:-3] for module in spec["modules"]}
    chunks = []
    shadowed_all: list[tuple[str, int, str]] = []
    for filename in spec["modules"]:
        raw = (source_dir / filename).read_text()
        prefixes = stems | sibling_module_aliases(raw, stems)
        text = strip_module_preamble(raw)
        text, shadowed = strip_module_prefixes(text, prefixes)
        shadowed_all.extend((filename, lineno, id_)
                            for lineno, _, id_ in shadowed)
        chunks.append(
            f"# --- from {filename} " + "-" * max(4, 56 - len(filename))
            + "\n" + text.strip("\n") + "\n"
        )

    body = "\n\n".join(chunks)
    for token in ("sgl_jax.", "tpu_inference.", "tokamax._src", "maxtext."):
        if token in body:
            raise ValueError(f"{name}: unresolved upstream reference {token}")
    for entry in spec["required"]:
        if f"def {entry}(" not in body:
            raise ValueError(f"{name}: lost entry point {entry}")

    code = "\n".join(
        line for line in body.splitlines() if not line.startswith("# --- from ")
    )
    leftover = unbound_module_refs(code, stems)
    if leftover:
        raise ValueError(
            f"{name}: module qualifiers survived flattening: {leftover[:5]}"
        )
    if shadowed_all:
        print(
            f"{name}: kept {len(shadowed_all)} qualifier(s) whose name is a "
            f"local, not the module: {shadowed_all[:6]}"
        )

    fields = {k: v for k, v in spec.items()
              if k not in {"modules", "required", "source_path"}}
    header = HEADER.format(
        repository=SGLANG,
        commit=SGLANG_COMMIT,
        files=spec["modules"],
        source_path=spec.get("source_path", spec["path"]),
        transformation=(
            "copied as a standalone implementation with source metadata added"
            if len(spec["modules"]) == 1
            else "the module files above flattened into one file in dependency "
            "order, with repo-local imports dropped"
        ),
        **fields,
    )
    result = header + body + f"\n\nkernel = {spec['entry']}\n"
    import ast

    ast.parse(result)  # a mangled flatten must fail here, not at import time
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(VARIANTS))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build(args.name, args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
