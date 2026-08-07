"""Flatten tpu-inference's DeepSeek-V4 compressor into two standalone files.

The compressor is the KV-cache-update half of DeepSeek-V4 sparse attention: it
projects hidden states into a compressed KV state, saves that state, then at
every ``compress_ratio`` boundary normalises, RoPEs, quantises and stores one
record into the shared cache.

Upstream ships **two implementations of the same operation**, which is what
makes this family testable:

``compressor_v1.py``    two Pallas kernels chained -- ``proj_and_save_state``
                        fuses the projection with the state scatter, and
                        ``compress_and_store.kernel.compress_norm_rope_store``
                        does the boundary compress-and-store.  Six modules.
``compressor.py``       pure JAX over ``compress_norm_rope.py``, Pallas-free.
                        Upstream's own ``tests/kernels/deepseek_v4/
                        compressor_test.py`` checks it against a NumPy ground
                        truth, so it is used here as the corpus reference
                        rather than a re-derivation.

Both are flattened by this script: ``kernel`` produces the Pallas file (two
launch points) and ``reference`` produces the Pallas-free one.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshot; the generated files are the runnable artifacts.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from flatten_gdn import (  # noqa: E402
    rename_top_level,
    sibling_module_aliases,
    strip_module_prefixes,
    strip_module_preamble,
    unbound_module_refs,
)


TPU_INFERENCE = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
UPSTREAM = "tpu_inference/kernels/experimental/deepseek_v4"

#: `proj_and_save_state.py` and `compress_and_store/config.py` both define
#: `Configs`, `Dimensions` and `TileSizes` with different bodies.  Each is
#: module-local upstream, so upstream has no conflict; concatenating them into
#: one file does.  The proj-side three are renamed, because the compress side is
#: reached through a `config.` qualifier that the flatten strips to the bare
#: name.
VARIANTS = {
    "kernel": {
        "output_kind": "pallas",
        "modules": (
            ("proj_and_save_state.py", {
                "Configs": "ProjConfigs",
                "Dimensions": "ProjDimensions",
                "TileSizes": "ProjTileSizes",
                # Upstream calls its inner kernel plainly `kernel`, which the
                # corpus's own `kernel = <entry>` footer would shadow.
                "kernel": "proj_and_save_state_kernel",
            }),
            ("config.py", {}),
            ("compute.py", {}),
            ("buffered_ref.py", {}),
            ("kernel.py", {}),
            ("compressor_v1.py", {}),
        ),
        "required": (
            "proj_and_save_state",
            "compress_norm_rope_store",
            "compressor_forward",
        ),
        "launch_points": 2,
        "contract": "dsv4_compress_and_store",
        "entry": "compressor_forward",
    },
    "reference": {
        "output_kind": "reference",
        "modules": (
            ("compress_norm_rope.py", {}),
            ("compressor.py", {}),
        ),
        "required": (
            "compressor_forward",
            "compress_norm_rope_store",
            "save_partial_states",
            "unpack_state_cache",
            "pack_state_cache",
            "unpack_sparse_kv_cache",
        ),
        "launch_points": 0,
        "contract": "dsv4_compress_and_store",
        "entry": "compressor_forward",
    },
}

#: Every repo-local import in these six modules, as an exact pattern.  A regex
#: over line starts would leave the continuation of the parenthesised ones
#: behind as stray indented text.
LOCAL_IMPORTS = (
    re.compile(
        r"^from tpu_inference\.kernels\.experimental\.deepseek_v4"
        r"\.compress_and_store import \(\n\s+[^)]*\)\s*$",
        re.M,
    ),
    re.compile(
        r"^from tpu_inference\.kernels\.experimental\.deepseek_v4"
        r"\.compress_and_store import \\\n\s+\w+\s*$",
        re.M,
    ),
    re.compile(
        r"^from tpu_inference\.kernels\.experimental\.deepseek_v4"
        r"\.compress_norm_rope import \(\n\s+[^)]*\)\s*$",
        re.M,
    ),
    re.compile(
        r"^import tpu_inference\.kernels\.experimental\.deepseek_v4"
        r"[\w.]* as \w+\s*$",
        re.M,
    ),
)

PALLAS_HEADER = '''"""Standalone vLLM tpu-inference DeepSeek-V4 compressor (Pallas).

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/
  files: {files}
  transformation: the six modules above flattened in dependency order, with the
    repo-local imports and module qualifiers removed.  `proj_and_save_state.py`
    and `compress_and_store/config.py` both define `Configs`, `Dimensions` and
    `TileSizes` with different bodies, so the proj-side three are renamed
    `Proj*` (see below).  No other change.

Entry point: ``compressor_forward`` (also exported as ``kernel``), which chains
both kernels exactly as upstream's `compressor_v1` does.

**Two Pallas launch points** live here:

- ``proj_and_save_state`` -- fuses the ``hidden_states @ wkv_wgate`` projection
  with the scatter of the resulting state into the packed uint8 cache;
- ``compress_norm_rope_store`` -- at every ``compress_ratio`` boundary, reads
  the saved state back, RMS-normalises it, applies interleaved RoPE, quantises
  the non-positional part to fp8 with per-block ue8m0 scales, and writes one
  packed record.

The state and the compressed records share **one uint8 buffer**: the kernel
reads its own input cache as the state source and writes the boundary records
back into it, which is why ``cache`` is donated.  ``rope_cache`` is a second
donated buffer; the pure-JAX reference next door instead packs rope into the
single cache, so the two are compared after unpacking, not byte-for-byte.

**Renamed on flattening.** `Configs`, `Dimensions` and `TileSizes` from
`proj_and_save_state.py` are `ProjConfigs`, `ProjDimensions` and
`ProjTileSizes` here; the identically-named `compress_and_store/config.py`
classes keep their names, because the kernel reaches them through a `config.`
qualifier the flatten strips to the bare name.  That module's inner Pallas
kernel, which upstream calls plainly `kernel`, is `proj_and_save_state_kernel`
here so it does not collide with this file's `kernel = compressor_forward`
export.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "files": {files!r},
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "launch_points": {launch_points},
    "renamed_on_flatten": {{
        "Configs": "ProjConfigs",
        "Dimensions": "ProjDimensions",
        "TileSizes": "ProjTileSizes",
        "kernel": "proj_and_save_state_kernel",
    }},
}}

import dataclasses
import enum
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

'''

REFERENCE_HEADER = '''"""Standalone vLLM tpu-inference DeepSeek-V4 compressor (pure JAX).

Source:
  repository: {repository}
  commit: {commit}
  path: {upstream}/
  files: {files}
  transformation: the two modules above flattened in dependency order with the
    repo-local import removed.  No other change.

**This file contains no Pallas.**  It is upstream's own reference
implementation of the same operation the Pallas compressor performs, and
upstream's ``tests/kernels/deepseek_v4/compressor_test.py`` checks it against a
NumPy ground truth.  The corpus uses it as this contract's reference rather
than re-deriving the packed cache layout, for the same reason as the
ragged-paged-attention and MLA families: a subtly wrong reference produces
false failures that look like kernel bugs.

Entry point: ``compressor_forward``.  Note its signature is **not** the Pallas
one: it takes ``kv_score`` already projected, where the Pallas path takes
``hidden_states`` and ``wkv_wgate`` and does the projection itself inside
``proj_and_save_state``.  It also packs rope into the single ``cache`` where the
Pallas path writes a separate ``rope_cache``.  ``baseline.py`` records how to
line the two up.

Also exported, and used by the corpus tests: ``unpack_state_cache`` /
``pack_state_cache`` (the fp32 state view of the uint8 buffer) and
``unpack_sparse_kv_cache`` (nope, rope and scales out of a packed record).
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{upstream}",
    "files": {files!r},
    "kind": "reference",
    "backend": "jax",
    "target": "portable",
    "contract": "{contract}",
    "launch_points": {launch_points},
}}

import jax
import jax.numpy as jnp

'''


def build(name: str, source_dir: Path) -> str:
    spec = VARIANTS[name]
    stems = {module[:-3] for module, _ in spec["modules"]}
    chunks = []
    for filename, renames in spec["modules"]:
        raw = (source_dir / filename).read_text()
        prefixes = stems | sibling_module_aliases(raw, stems)
        text = strip_module_preamble(raw)
        for pattern in LOCAL_IMPORTS:
            text = pattern.sub("", text)
        text = rename_top_level(text, renames)
        text, shadowed = strip_module_prefixes(text, prefixes)
        if shadowed:
            print(f"{name} {filename}: kept {len(shadowed)} shadowed qualifier(s)")
        chunks.append(
            f"# --- from {filename} " + "-" * max(4, 56 - len(filename))
            + "\n" + text.strip("\n") + "\n"
        )

    body = "\n\n".join(chunks)
    code = "\n".join(
        line for line in body.splitlines() if not line.startswith("# --- from ")
    )
    for token in ("tpu_inference", "sgl_jax", "tokamax._src", "maxtext."):
        if token in code:
            offender = next(line for line in code.splitlines() if token in line)
            raise ValueError(f"{name}: unresolved upstream reference: {offender!r}")
    leftover = unbound_module_refs(code, stems)
    if leftover:
        raise ValueError(f"{name}: module qualifiers survived: {leftover[:5]}")
    for entry in spec["required"]:
        if f"def {entry}(" not in body:
            raise ValueError(f"{name}: lost entry point {entry}")

    launches = code.count("pl.pallas_call(")
    if launches != spec["launch_points"]:
        raise ValueError(
            f"{name}: expected {spec['launch_points']} pallas_call(s), "
            f"found {launches}"
        )

    template = (PALLAS_HEADER if spec["output_kind"] == "pallas"
                else REFERENCE_HEADER)
    header = template.format(
        repository=TPU_INFERENCE,
        commit=COMMIT,
        upstream=UPSTREAM,
        files=tuple(module for module, _ in spec["modules"]),
        contract=spec["contract"],
        launch_points=spec["launch_points"],
    )
    result = header + body
    if spec["output_kind"] == "pallas":
        result += f"\n\nkernel = {spec['entry']}\n"
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
