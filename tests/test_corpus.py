"""Repository-level invariants that do not require a TPU."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
FORBIDDEN_IMPORT_PREFIXES = (
    "JAXBench",
    "pallasbench",
    "MaxText",
    "maxtext",
    "tokamax",
    "tpu_inference",
    "sgl_jax",
)
REQUIRED_SOURCE_KEYS = ("repository", "commit", "path", "backend", "target", "contract")
FORBIDDEN_DIRECTORIES = ("upstream", "adapters", "environments")


def implementation_files():
    return sorted((ROOT / "kernels").glob("*/*/*_optimized.py"))


def module_assignments(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_optimized_files_are_syntax_valid_and_standalone():
    assert implementation_files()
    for path in implementation_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        forbidden = [
            name
            for name in imports
            if name.startswith(FORBIDDEN_IMPORT_PREFIXES)
        ]
        assert not forbidden, f"{path}: upstream package imports: {forbidden}"


def test_each_family_has_a_baseline():
    families = {path.parent for path in implementation_files()}
    missing = [family for family in families if not (family / "baseline.py").is_file()]
    assert not missing


def test_every_optimized_file_declares_tpu_provenance():
    for path in implementation_files():
        source = module_assignments(ast.parse(path.read_text(), filename=str(path)))
        assert "SOURCE" in source, f"{path}: no SOURCE metadata"
        metadata = source["SOURCE"]
        assert isinstance(metadata, dict), f"{path}: SOURCE is not a dict"
        missing = [key for key in REQUIRED_SOURCE_KEYS if key not in metadata]
        assert not missing, f"{path}: SOURCE missing {missing}"
        assert metadata["target"] == "tpu", f"{path}: non-TPU target"
        assert "gpu" not in metadata["backend"], f"{path}: GPU backend"
        assert "triton" not in metadata["backend"], f"{path}: Triton backend"
        assert len(metadata["commit"]) == 40, f"{path}: commit is not a full SHA"


def test_no_forbidden_directories():
    for name in FORBIDDEN_DIRECTORIES:
        assert not (ROOT / name).exists(), f"{name}/ must not exist"


def test_inventory_is_self_consistent():
    inventory = json.loads((ROOT / "inventory.json").read_text())
    counts = inventory["counts"]
    points = inventory["launch_points"]

    assert len(points) == counts["audited_launch_points"]
    assert sum(p["tpu_compatible"] for p in points) == counts["audited_tpu_launch_points"]
    assert (
        counts["audited_tpu_launch_points"] + counts["audited_gpu_launch_points"]
        == counts["audited_launch_points"]
    )
    assert not inventory["generated_from"]["parse_failures"]

    families = inventory["families"]
    assert len(families) == counts["semantic_families"]
    assert sum(e["audited_launch_points"] for e in families.values()) == len(points)

    migrated = 0
    for name, entry in families.items():
        assert entry["migrated_launch_points"] <= entry["audited_tpu_launch_points"], (
            f"{name}: claims more migrated launch points than exist upstream"
        )
        migrated += entry["migrated_launch_points"]
        expected = (
            "not_started"
            if entry["migrated_launch_points"] == 0
            else "complete"
            if entry["migrated_launch_points"] >= entry["audited_tpu_launch_points"]
            else "partial"
        )
        assert entry["migration_status"] == expected, f"{name}: wrong status"
    assert migrated == counts["migrated_launch_points"]


def test_inventory_corpus_claims_match_the_filesystem():
    inventory = json.loads((ROOT / "inventory.json").read_text())
    for entry in inventory["families"].values():
        for implementation in entry["corpus_implementations"]:
            directory = ROOT / implementation["directory"]
            path = directory / implementation["file"]
            assert path.is_file(), f"inventory names a missing file: {path}"
            assert (directory / implementation["baseline"]).is_file()

            source = module_assignments(ast.parse(path.read_text(), filename=str(path)))
            metadata = source["SOURCE"]
            assert metadata["contract"] == implementation["contract"], (
                f"{path}: SOURCE contract disagrees with the inventory"
            )
            # SOURCE names a file, or the directory whose modules were
            # flattened into one standalone file; the inventory always names
            # the primary implementation file.  One must contain the other.
            declared = metadata["path"]
            upstream = implementation["upstream_path"]
            assert declared.endswith(upstream) or upstream.startswith(declared), (
                f"{path}: SOURCE path {declared!r} disagrees with the "
                f"inventory path {upstream!r}"
            )

            has_native_shape = implementation["native_shape"] is not None
            if implementation["profile"] is None:
                # Normally allowed only when there is no upstream-defined native
                # shape to profile at; the reason must be recorded, not implied.
                # The one exception is a kernel whose upstream already publishes
                # its own numbers, where re-measuring under this corpus's
                # tracing flags would produce a second, worse set.  That has to
                # be claimed in a named field rather than buried in prose, so it
                # cannot become a way to skip profiling quietly.
                assert not has_native_shape or implementation.get(
                    "unprofiled_reason"
                ), f"{path}: has a native shape but no profile"
                assert implementation.get("notes"), (
                    f"{path}: unprofiled implementations must say why"
                )
                continue

            profile = ROOT / implementation["profile"]
            assert profile.is_file(), f"inventory names a missing profile: {profile}"
            result = json.loads(profile.read_text())
            # The ledger's native_shape and the profile's native_source_shape
            # must agree, so a validation-shape run can never be read as a
            # native-shape result.
            assert result["native_source_shape"] is has_native_shape, (
                f"{path}: inventory native_shape={implementation['native_shape']!r} "
                f"disagrees with profile native_source_shape="
                f"{result['native_source_shape']}"
            )
            if not has_native_shape:
                assert implementation.get("notes"), (
                    f"{path}: validation-shape profiles must say why there is "
                    "no native shape"
                )
            assert result["warmup_iterations"] == 5
            assert result["profiled_iterations"] == 50
            assert result["correctness"]["status"] == implementation["correctness"]


@pytest.mark.skipif(
    not os.environ.get("PINNED_CHECKOUTS"),
    reason="set PINNED_CHECKOUTS to the directory holding the pinned upstream trees",
)
def test_source_paths_exist_in_pinned_checkouts():
    """Verify provenance against the real pinned trees when they are available."""
    checkouts = Path(os.environ["PINNED_CHECKOUTS"])
    scopes = {
        "https://github.com/AI-Hypercomputer/accelerator-agents": "accelerator-agents",
        "https://github.com/Tyronita/PallasBench": "PallasBench",
        "https://github.com/AI-Hypercomputer/maxtext": "maxtext",
        "https://github.com/openxla/tokamax": "tokamax",
        "https://github.com/vllm-project/tpu-inference": "tpu-inference",
        "https://github.com/sgl-project/sglang-jax": "sglang-jax",
    }
    for path in implementation_files():
        metadata = module_assignments(
            ast.parse(path.read_text(), filename=str(path))
        )["SOURCE"]
        root = checkouts / scopes[metadata["repository"]]
        target = root / metadata["path"]
        assert target.exists(), f"{path}: SOURCE path not found upstream: {target}"
