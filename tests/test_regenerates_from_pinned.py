"""Every corpus file must still regenerate from the commit it names.

This is the corpus's foundational claim — each kernel file is reproducible from
its recorded upstream commit by the matching `tools/flatten_*.py` — and until an
audit ran it wholesale, nothing enforced it. That audit found one file
(`ragged_paged_attention/sglang_jax_optimized.py`) had drifted 16 bytes from the
tool that claims to produce it: inert, but undetected. This test is the guard
that would have caught it the day it appeared.

It needs the pinned upstream trees, which are deliberately NOT vendored — they
are ~210 MB across six repositories. Point `PINNED_CHECKOUTS` at a directory
holding them, or place them in a `pinned/` directory beside this repository, and
these run. Otherwise they skip, which is why this file is safe in CI.

To reconstruct the trees:

    for repo in accelerator-agents PallasBench maxtext tokamax \
                tpu-inference sglang-jax; do ...
    # each cloned and checked out at the commit in THIRD_PARTY_NOTICES.md

Only the scripts taking a plain checkout root are covered. Several older
flatteners take a *staging directory* of renamed files instead, and working out
that staging from their `SOURCES` tables is exactly the friction that let the
drift go unnoticed; extending this table to them is worthwhile follow-up work.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).parents[1]


def pinned_root() -> Path | None:
    candidates = []
    if os.environ.get("PINNED_CHECKOUTS"):
        candidates.append(Path(os.environ["PINNED_CHECKOUTS"]))
    candidates.append(ROOT.parent / "pinned")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "PallasBench").is_dir():
            return candidate
    return None


PINNED = pinned_root()

pytestmark = pytest.mark.skipif(
    PINNED is None,
    reason="pinned upstream trees not present; set PINNED_CHECKOUTS to enable",
)

#: script -> (argv builder, the corpus files it is responsible for)
#: `out` is a temp directory standing in for the corpus directory.
CASES = {
    "flatten_ragged_mqa.py": (
        lambda p, out: [str(p / "maxtext"), out],
        "kernels/attention/ragged_mqa_attention",
        ("maxtext_optimized.py", "baseline.py"),
    ),
    "flatten_spmm.py": (
        lambda p, out: [str(p / "tpu-inference"), out],
        "kernels/matmul/structured_sparse_matmul",
        ("tpu_inference_optimized.py", "baseline.py"),
    ),
    "flatten_causal_conv1d.py": (
        lambda p, out: [str(p / "tpu-inference"), out],
        "kernels/convolution/causal_conv1d",
        ("tpu_inference_optimized.py", "baseline.py"),
    ),
    "flatten_collectives.py": (
        lambda p, out: [str(p / "tpu-inference"), out],
        "kernels/collectives/collective_matmul",
        ("tpu_inference_all_gather_matmul_optimized.py",
         "tpu_inference_hierarchical_reduce_scatter_optimized.py"),
    ),
    "flatten_speculative.py": (
        lambda p, out: [str(p / "sglang-jax"), out],
        "kernels/sampling/speculative_decoding",
        ("sglang_jax_verify_tree_greedy_optimized.py",
         "sglang_jax_tree_sampling_optimized.py",
         "sglang_jax_build_tree_optimized.py", "baseline.py"),
    ),
    "flatten_fused_mlp.py": (
        lambda p, out: [str(p / "sglang-jax"), out],
        "kernels/moe/gated_mlp",
        ("sglang_jax_optimized.py", "sglang_jax_reference.py"),
    ),
    "flatten_linear_cross_entropy.py": (
        lambda p, out: [
            str(p / "tokamax" / "tokamax" / "_src" / "ops"
                / "linear_softmax_cross_entropy_loss"), out],
        "kernels/loss/cross_entropy",
        ("tokamax_optimized.py", "tokamax_reference.py"),
    ),
    "flatten_dense_matmul.py": (
        lambda p, out: [
            str(p / "accelerator-agents" / "JAXBench" / "benchmark" / "8p_GEMM"), out],
        "kernels/matmul/dense_matmul",
        ("jaxbench_optimized.py", "jaxbench_reference.py"),
    ),
    "flatten_paged_attention.py": (
        lambda p, out: [
            str(p / "accelerator-agents" / "JAXBench" / "benchmark"
                / "6p_Paged_Attention"), out],
        "kernels/attention/paged_attention",
        ("jaxbench_optimized.py", "baseline.py"),
    ),
    "flatten_transpose.py": (
        lambda p, out: [str(p / "tpu-inference"), out],
        "kernels/memory/layout_transpose",
        ("tpu_inference_optimized.py", "baseline.py"),
    ),
}


@pytest.mark.parametrize("script", sorted(CASES))
def test_regenerates_byte_identically(script):
    builder, directory, filenames = CASES[script]
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / script)] + builder(PINNED, tmp),
            cwd=ROOT / "tools", capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{script} failed to run:\n{result.stderr[-1500:]}")
        for name in filenames:
            regenerated = Path(tmp) / name
            committed = ROOT / directory / name
            assert regenerated.is_file(), f"{script} did not write {name}"
            assert committed.is_file(), f"{directory}/{name} is missing"
            if regenerated.read_bytes() != committed.read_bytes():
                import difflib
                diff = "\n".join(difflib.unified_diff(
                    committed.read_text().splitlines(),
                    regenerated.read_text().splitlines(),
                    "committed", "regenerated", lineterm="", n=2))[:2000]
                pytest.fail(
                    f"{directory}/{name} has drifted from {script}:\n{diff}")


def test_pallasbench_regenerates_byte_identically():
    """The largest block: 43 kernels and 19 baselines from one script."""
    import json
    tasks = json.loads((ROOT / "tools" / "pallasbench_tasks.json").read_text())
    checkout = PINNED / "PallasBench"
    with tempfile.TemporaryDirectory() as tmp:
        for what in ("kernels", "baselines"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "flatten_pallasbench.py"),
                 what, str(checkout), tmp],
                cwd=ROOT / "tools", capture_output=True, text=True)
            assert result.returncode == 0, result.stderr[-1500:]

        drifted, checked = [], 0
        for task, spec in tasks.items():
            if spec["family"] == "flash_attention" or spec.get("excluded"):
                continue
            relative = Path("kernels") / spec["category"] / spec["family"]
            for name in (f"pallasbench_{task}_optimized.py", "baseline.py"):
                regenerated = Path(tmp) / relative / name
                committed = ROOT / relative / name
                if not regenerated.is_file():
                    continue
                checked += 1
                if regenerated.read_bytes() != committed.read_bytes():
                    drifted.append(str(relative / name))
        assert checked >= 60, f"expected to check 60+ files, checked {checked}"
        assert not drifted, f"{len(drifted)} PallasBench files drifted: {drifted[:5]}"
