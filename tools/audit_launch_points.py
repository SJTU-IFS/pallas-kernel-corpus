"""Re-derive the Pallas launch-point ledger from pinned upstream checkouts.

The corpus ships an ``inventory.json`` ledger whose ``launch_points`` counts
must be reproducible, not asserted.  This script walks each pinned upstream
tree, finds every Pallas launch call with the AST, records its enclosing
function, and classifies the launch backend as TPU, GPU, or portable.

It is deliberately conservative: a launch is TPU-compatible only when nothing
in its file marks it as a Mosaic-GPU or Pallas-Triton implementation.

Usage (on any machine holding the pinned trees):

  python tools/audit_launch_points.py --checkouts <dir-with-pinned-trees> \
      --output /tmp/launch_points.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys


# repo key -> (upstream URL, pinned commit, in-repo scope, display name)
REPOS = {
    "JAXBench": (
        "https://github.com/AI-Hypercomputer/accelerator-agents",
        "6b6c44293c43976032ba12d2f72d6bebeaf2394f",
        "accelerator-agents/JAXBench",
    ),
    "PallasBench": (
        "https://github.com/Tyronita/PallasBench",
        "30a6ee07fd4923f3877906a94002d994e972d6fe",
        "PallasBench",
    ),
    "MaxText": (
        "https://github.com/AI-Hypercomputer/maxtext",
        "ca420634a9e9e73feaacc8001f605163d2d80ea1",
        "maxtext/src/maxtext/kernels",
    ),
    "tokamax": (
        "https://github.com/openxla/tokamax",
        "927e3f94e8ffe0430cf38bd1423112bb2f69ec66",
        "tokamax",
    ),
    "tpu-inference": (
        "https://github.com/vllm-project/tpu-inference",
        "8b9c90928c94c7230d1bc891534a301510a6a30d",
        "tpu-inference/tpu_inference/kernels",
    ),
    "sglang-jax": (
        "https://github.com/sgl-project/sglang-jax",
        "a7353325e8c00d287294c2cd679a77173f1a4594",
        "sglang-jax/python/sgl_jax/srt/kernels",
    ),
}

# Call names that launch a Pallas program.  ``pl.kernel``/``plgpu.kernel`` are
# the newer API; ``core_map_helper.kernel`` is tpu-inference's documented
# drop-in for ``pl.kernel`` that lowers through ``core_map``.
LAUNCH_SUFFIXES = (
    "pallas_call",
    "custom_buffered_pallas_call",
)
KERNEL_LAUNCH_QUALIFIERS = (
    "pl",
    "plgpu",
    "pltpu",
    "core_map_helper",
    "mosaic_tpu",
)

# Files that are not production kernel sources.
EXCLUDED_PATH_PARTS = (
    "/test/",
    "/tests/",
    "/benchmarks/",
    "/examples/",
    "/docs/",
    "/third_party/",
    "/tuning/",
)
EXCLUDED_NAME_SUFFIXES = (
    "_test.py",
    "_tests.py",
    "_benchmark.py",
    "_benchmarking.py",
    "_bench.py",
)
EXCLUDED_NAMES = (
    "baseline.py",
    "conftest.py",
)

GPU_MARKERS = (
    "mosaic_gpu",
    "pallas_triton",
    "triton_kernel",
    "_sm90",
    "_sm100",
)
GPU_IMPORT_MARKERS = (
    "jax.experimental.pallas.mosaic_gpu",
    "jax.experimental.pallas.triton",
    "pallas.mosaic_gpu",
    "pallas.triton",
)
TPU_IMPORT_MARKERS = (
    "jax.experimental.pallas.tpu",
    "jax.experimental.pallas.mosaic",
    "pallas.tpu",
)

# Launch *infrastructure*, not kernels: these modules define generic wrappers
# that take an arbitrary kernel body from their caller.  Counting them would
# double-count every kernel that goes through them.
INFRASTRUCTURE_MODULES = {
    ("tokamax", "tokamax/_src/pallas/block.py"),
    ("tokamax", "tokamax/_src/mosaic_tpu.py"),
}
INFRASTRUCTURE_FUNCTIONS = {
    ("tokamax", "pallas_call"),
    ("tokamax", "custom_buffered_pallas_call"),
    ("tokamax", "_pallas_call"),
    ("tpu-inference", "kernel_helper"),
}


def call_name(node: ast.Call) -> str | None:
    """Return a dotted name for a call target, or ``None``."""
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif parts:
        parts.append("<expr>")
    else:
        return None
    return ".".join(reversed(parts))


def is_launch(name: str) -> bool:
    tail = name.rsplit(".", 1)[-1]
    if tail in LAUNCH_SUFFIXES:
        return True
    if tail == "kernel":
        head = name.rsplit(".", 1)[0] if "." in name else ""
        return head in KERNEL_LAUNCH_QUALIFIERS
    return False


def excluded(path: Path, scope: Path) -> bool:
    relative = "/" + str(path.relative_to(scope))
    if any(part in relative for part in EXCLUDED_PATH_PARTS):
        return True
    if path.name.startswith("test_"):
        return True
    if path.name.endswith(EXCLUDED_NAME_SUFFIXES):
        return True
    return path.name in EXCLUDED_NAMES


def enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Map every line inside a function body to that function's name."""
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno or node.lineno, node.name))
    spans.sort(key=lambda span: span[1] - span[0])  # innermost wins
    mapping: dict[int, str] = {}
    for start, end, name in reversed(spans):
        for line in range(start, end + 1):
            mapping[line] = name
    return mapping


#: The Pallas backend modules that mean "not TPU", by module path.
GPU_BACKEND_MODULES = ("triton", "mosaic_gpu")


def imports_a_gpu_backend(source: str) -> bool:
    """Does this module import a Pallas GPU backend, in any spelling?

    The string markers below only match the dotted form (`import
    jax.experimental.pallas.triton`).  sglang-jax's paged attention writes
    `from jax.experimental.pallas import triton as plgpu` instead, where that
    substring never appears -- so it was classified `portable` and counted as a
    TPU migration candidate, though it passes `plgpu.CompilerParams`
    unconditionally and Mosaic's TPU lowering asserts TPU compiler params.  It
    was the only launch point affected, but the blind spot was the check, not
    the kernel, so this looks at the parsed imports instead of the text.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("pallas") or ".pallas." in f"{node.module}.":
                if any(alias.name in GPU_BACKEND_MODULES for alias in node.names):
                    return True
            if any(f".pallas.{backend}" in f".{node.module}"
                   for backend in GPU_BACKEND_MODULES):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(f".pallas.{backend}" in f".{alias.name}"
                       for backend in GPU_BACKEND_MODULES):
                    return True
    return False


def classify_backend(source: str, path: Path, launch: str) -> str:
    lowered = str(path).lower()
    if any(marker in lowered for marker in GPU_MARKERS):
        return "gpu"
    if launch.endswith("plgpu.kernel") or launch.startswith("plgpu."):
        return "gpu"
    if any(marker in source for marker in GPU_IMPORT_MARKERS):
        return "gpu"
    if imports_a_gpu_backend(source):
        return "gpu"
    if "mosaic_tpu" in lowered or any(m in source for m in TPU_IMPORT_MARKERS):
        return "tpu"
    return "portable"  # generic Pallas: runs on TPU via Mosaic lowering


def audit(checkouts: Path) -> dict:
    records: list[dict] = []
    parse_failures: list[dict] = []
    for repo, (url, commit, scope_suffix) in REPOS.items():
        scope = checkouts / scope_suffix
        if not scope.is_dir():
            raise FileNotFoundError(f"missing pinned scope for {repo}: {scope}")
        for path in sorted(scope.rglob("*.py")):
            if excluded(path, scope):
                continue
            relative = str(path.relative_to(scope))
            if (repo, relative) in INFRASTRUCTURE_MODULES:
                continue
            source = path.read_text(errors="replace")
            if "pallas" not in source and "plgpu" not in source:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as error:
                # Never skip silently: an unparsed file is a potential
                # undercount, so record it.  Upstream uses Python 3.12 syntax,
                # so this must be run with 3.12 or newer.
                parse_failures.append(
                    {
                        "repository": repo,
                        "path": relative,
                        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                        "error": str(error),
                        "contains_launch_text": any(
                            token in source
                            for token in ("pallas_call", "pl.kernel", "plgpu.kernel")
                        ),
                    }
                )
                continue
            functions = enclosing_functions(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node)
                if not name or not is_launch(name):
                    continue
                function = functions.get(node.lineno, "<module>")
                if (repo, function) in INFRASTRUCTURE_FUNCTIONS:
                    continue
                records.append(
                    {
                        "repository": repo,
                        "repository_url": url,
                        "commit": commit,
                        "path": relative,
                        "line": node.lineno,
                        "launch_api": name,
                        "enclosing_function": function,
                        "backend": classify_backend(source, path, name),
                    }
                )
    totals: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = totals.setdefault(
            record["repository"], {"all": 0, "tpu": 0, "gpu": 0, "portable": 0}
        )
        bucket["all"] += 1
        bucket[record["backend"]] += 1
    for bucket in totals.values():
        bucket["tpu_compatible"] = bucket["tpu"] + bucket["portable"]
    return {
        "pinned_commits": {repo: commit for repo, (_, commit, _) in REPOS.items()},
        "totals": totals,
        "grand_total": len(records),
        "grand_total_tpu_compatible": sum(
            bucket["tpu_compatible"] for bucket in totals.values()
        ),
        "parse_failures": parse_failures,
        "launch_points": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "launch_points"}, indent=2))


if __name__ == "__main__":
    main()
