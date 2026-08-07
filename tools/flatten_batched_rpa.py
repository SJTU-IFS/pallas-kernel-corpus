"""Flatten tpu-inference's batched ragged paged attention into one file.

`batched_rpa` is the largest single implementation in the corpus by module
count: nine files that reference each other through the **package**
(``from ...batched_rpa import configs, utils``) and then use ``configs.X`` /
``utils.Y``.  Flattening therefore has to strip those prefixes as well as the
imports, which is what ``flatten_gdn``'s helpers already do -- they are imported
rather than copied so the parenthesised-import and aliased-module fixes stay in
one place.

Two references reach outside the package and are handled by substitution:

- ``tpu_inference.logger.init_logger`` -> the stdlib logger, as the hd64 and
  kv_cache_update kernels needed;
- ``tpu_inference.envs`` -> a tiny stand-in.  Upstream reads
  ``envs.USE_BATCHED_RPA_SEQ_ON_LANE``, an environment-variable switch, so the
  corpus supplies it as a real module-level flag rather than importing vLLM's
  whole settings module.  It defaults to False, which is upstream's default.

Like the other ``flatten_*`` scripts this is a reproducibility aid pinned to the
audited snapshot; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from flatten_gdn import (  # noqa: E402
    sibling_module_aliases,
    strip_module_prefixes,
    strip_module_preamble,
    unbound_module_refs,
)


TPU_INFERENCE = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
UPSTREAM_PATH = "tpu_inference/kernels/experimental/batched_rpa"

# Dependency order, derived from the imports rather than assumed:
#   utils -> configs -> {schedule, stitch_utils, flash_attention}
#         -> bref_override -> tuned_params -> kernel -> wrapper
MODULES = (
    "utils.py",
    "configs.py",
    "schedule.py",
    "stitch_utils.py",
    "flash_attention.py",
    "bref_override.py",
    "tuned_params.py",
    "kernel.py",
    "wrapper.py",
)

REPLACEMENTS = (
    ("from tpu_inference.logger import init_logger\n", ""),
    ("logger = init_logger(__name__)", "logger = logging.getLogger(__name__)"),
    ("logger.warning_once(", "logger.warning("),
    ("logger.info_once(", "logger.info("),
    ("from tpu_inference import envs\n", ""),
)

HEADER = '''"""Standalone vLLM tpu-inference batched ragged paged attention.

Source:
  repository: {repository}
  commit: {commit}
  path: {path}/
  files: {files}
  transformation: the nine modules above flattened in dependency order.  They
    reference each other through the package (`from ...batched_rpa import
    configs, utils`) and then use `configs.X`, so both the imports and the
    module prefixes were stripped.  Two out-of-package references were
    substituted: vLLM's `init_logger` became the stdlib logger, and
    `tpu_inference.envs` became the `envs` shim below.

Entry point: ``ragged_paged_attention`` (also exported as ``kernel``).

Contract ``batched_rpa`` -- **two Pallas launch points**: the attention kernel
itself and the schedule-metadata kernel that plans which (sequence, page) pairs
each grid step handles.  It batches work across sequences rather than looping
per sequence, which is what distinguishes it from the v3 kernels.

`envs.USE_BATCHED_RPA_SEQ_ON_LANE` is upstream an environment-variable switch
read from vLLM's settings module.  It is a plain module-level flag here,
defaulting to **False** as upstream does; set `envs.USE_BATCHED_RPA_SEQ_ON_LANE
= True` to take the other layout.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}",
    "files": {files!r},
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "batched_rpa",
    "launch_points": 2,
    "substituted": (
        "tpu_inference.logger.init_logger -> logging.getLogger",
        "tpu_inference.envs -> the `envs` shim below",
    ),
}}

from dataclasses import asdict, dataclass
import dataclasses
import enum
import functools
import logging
import math
from typing import Any, Literal, NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


class _Envs:
    """Stand-in for `tpu_inference.envs`.

    Upstream reads this from vLLM's settings module, which the corpus does not
    depend on.  Only the one flag these kernels consult is provided, with
    upstream's default.
    """

    USE_BATCHED_RPA_SEQ_ON_LANE = False


envs = _Envs()

'''


def build(source_dir: Path) -> str:
    stems = {module[:-3] for module in MODULES}
    chunks = []
    shadowed_all: list[tuple[str, int, str]] = []
    for filename in MODULES:
        raw = (source_dir / filename).read_text()
        prefixes = stems | sibling_module_aliases(raw, stems)
        text = strip_module_preamble(raw)
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        text, shadowed = strip_module_prefixes(text, prefixes)
        shadowed_all.extend((filename, lineno, name)
                            for lineno, _, name in shadowed)
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
            raise ValueError(f"unresolved upstream reference {token}: {offender!r}")
    leftover = unbound_module_refs(code, stems)
    if leftover:
        raise ValueError(f"module qualifiers survived flattening: {leftover[:5]}")
    for entry in ("ragged_paged_attention", "rpa_kernel", "generate_rpa_metadata"):
        if f"def {entry}(" not in body:
            raise ValueError(f"lost entry point {entry}")
    if shadowed_all:
        print(
            f"kept {len(shadowed_all)} qualifier(s) whose name is a local, not "
            f"the module: {shadowed_all[:6]}"
        )

    header = HEADER.format(
        repository=TPU_INFERENCE, commit=COMMIT, path=UPSTREAM_PATH, files=MODULES
    )
    result = header + body + "\n\nkernel = ragged_paged_attention\n"
    import ast

    ast.parse(result)  # a mangled flatten must fail here, not at import time
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(build(args.source_dir))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
