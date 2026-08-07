"""Mechanically flatten a Megablox GMM kernel into one standalone file.

vLLM tpu-inference and sglang-jax both split their grouped-matmul kernel across
``common.py``, ``tuned_block_sizes.py``, and ``gmm.py``.  This script inlines
those three modules in dependency order and removes the repo-local imports and
logging shims, so the result runs without either upstream package installed.

Like ``flatten_tokamax_splash.py`` this is a reproducibility aid pinned to the
audited snapshots; the generated file is the runnable artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


SOURCES = {
    "tpu_inference": {
        "repository": "https://github.com/vllm-project/tpu-inference",
        "commit": "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "path": "tpu_inference/kernels/megablox",
        "files": ("common.py", "tuned_block_sizes.py", "gmm.py"),
        "local_imports": (
            re.compile(r"^from tpu_inference\.kernels\.megablox import .*$", re.M),
            re.compile(
                r"^from tpu_inference\.kernels\.megablox\.tuned_block_sizes import \\\n\s+.*$",
                re.M,
            ),
            re.compile(r"^from tpu_inference\.logger import init_logger$", re.M),
        ),
        "replacements": (
            ("logger = init_logger(__name__)", "logger = logging.getLogger(__name__)"),
            ("logger.warning_once(", "logger.warning("),
        ),
        "extra_imports": "import logging\n",
    },
    "sglang_jax": {
        "repository": "https://github.com/sgl-project/sglang-jax",
        "commit": "a7353325e8c00d287294c2cd679a77173f1a4594",
        "path": "python/sgl_jax/srt/kernels/gmm/megablox_gmm_kernel",
        "files": ("common.py", "tuned_block_sizes.py", "gmm.py"),
        "local_imports": (
            re.compile(
                r"^from sgl_jax\.srt\.kernels\.gmm\.megablox_gmm_kernel import .*$", re.M
            ),
            re.compile(
                r"^from sgl_jax\.srt\.kernels\.gmm\.megablox_gmm_kernel\.tuned_block_sizes import \(\n(?:.*\n)*?\)$",
                re.M,
            ),
        ),
        "replacements": (
            # Upstream prints the selected tiling on every call; that is trace
            # noise inside a profiling loop, so route it through the logger.
            (
                'print(f"using tuned block sizes for key: {key} = '
                '{TUNED_BLOCK_SIZES.get(key)}", flush=True)',
                'logger.info("[GMM kernel] using tuned block sizes for key: '
                '%s: %s", key, TUNED_BLOCK_SIZES.get(key))',
            ),
        ),
        "extra_imports": "",
    },
}

HEADER = '''"""Standalone {name} Megablox grouped-matmul TPU Pallas implementation.

Source:
  repository: {repository}
  commit: {commit}
  paths:
    {path}/
{file_list}
  transformation: the repo-local common, tuned-block-size, and kernel modules
    were flattened in dependency order; module qualifiers and the upstream
    logging shim were removed. The kernel body itself is unmodified.

Entry point: ``gmm`` (also exported as ``kernel``), the forward grouped matmul.
The upstream ``gmm_v2`` launch in the same package is audited but not migrated.
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "{path}/gmm.py",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "grouped_matmul_2d",
}}

{extra_imports}
'''

RUNNER = '''

kernel = gmm


def _main() -> None:
    import argparse
    import json
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args()

    keys = jax.random.split(jax.random.key(42), 2)
    lhs = jax.random.normal(keys[0], (args.rows, args.k), dtype=jnp.bfloat16)
    rhs = jax.random.normal(
        keys[1], (args.groups, args.k, args.n), dtype=jnp.bfloat16
    ) * 0.02
    group_sizes = jnp.full(
        (args.groups,), args.rows // args.groups, dtype=jnp.int32
    )

    start = time.perf_counter()
    out = gmm(lhs, rhs, group_sizes, interpret=args.interpret)
    out.block_until_ready()
    elapsed_ms = (time.perf_counter() - start) * 1e3

    expected = jax.lax.ragged_dot(
        lhs, rhs, group_sizes, preferred_element_type=jnp.float32
    )
    print(json.dumps({
        "implementation": SOURCE["repository"].rsplit("/", 1)[-1],
        "contract": SOURCE["contract"],
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        "compile_and_run_ms": elapsed_ms,
        "max_abs_error": float(jnp.max(jnp.abs(out - expected))),
    }))


if __name__ == "__main__":
    _main()
'''


def flatten(name: str, source_dir: Path) -> str:
    spec = SOURCES[name]
    chunks = []
    for filename in spec["files"]:
        text = (source_dir / filename).read_text()
        for pattern in spec["local_imports"]:
            text = pattern.sub("", text)
        chunks.append(f"\n# ---- flattened from {filename} ----\n\n{text}\n")
    body = "".join(chunks)
    for old, new in spec["replacements"]:
        if old not in body:
            raise ValueError(f"{name}: expected replacement text not found: {old!r}")
        body = body.replace(old, new)
    # ``common.`` is only ever a module qualifier in these files; the flattened
    # module defines those helpers at top level.
    body = body.replace("common.", "")
    header = HEADER.format(
        name={"tpu_inference": "vLLM tpu-inference", "sglang_jax": "sglang-jax"}[name],
        repository=spec["repository"],
        commit=spec["commit"],
        path=spec["path"],
        file_list="\n".join(f"      {f}" for f in spec["files"]),
        extra_imports=spec["extra_imports"],
    )
    return header + body + RUNNER


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
