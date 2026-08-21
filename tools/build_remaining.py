"""Regenerate ``REMAINING.md``: the audited launch points not yet collected.

Every launch point in ``inventory.json`` gets exactly one status:

  ``migrated``   a standalone corpus file covers it and it has been validated;
  ``excluded``   deliberately not collected, with a recorded reason;
  ``qwix``       blocked on a dependency decision the user has settled;
  ``open``       still to collect.

The ``migrated`` set is derived by matching each corpus implementation's
declared ``upstream_path`` and ``migrated_roles`` against the audit, then taking
exactly the ``migrated_launch_points`` it claims.  Taking *exactly* that many
matters: a looser match over-counts (a directory prefix like ``megablox``
swallows the v1 files that are deliberately deferred) and a stricter one
under-counts.  The script asserts the total reconciles with the ledger, so a
drifting count fails here rather than silently misreporting progress.

    python tools/build_remaining.py            # writes REMAINING.md
    python tools/build_remaining.py --summary  # counts only, no file
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))

# Launch points the corpus deliberately does not collect, and why.  These are
# conclusions with evidence behind them, recorded in the family reports.
EXCLUSIONS = {
    "3p_MLA_Attention":
        "same kernel as JAXBench flash attention (24/27 defs AST-identical)",
    "4p_Sparse_Attention":
        "same kernel as JAXBench 2p_GQA (28/31 defs AST-identical)",
    "ragged_scatter":
        "returns gather semantics under the obvious calling convention",
    "experimental/tpu/topk/pallas_mosaic_tpu_kernel.py":
        "SparseCore top-k; runs on this v6e but does not reproduce lax.top_k "
        "semantics -- see kernels/sampling/topk_routing/baseline.py",
    "tree_speculative_sampling_target_only_kernel.py":
        "carried but does not run, and the defect is upstream's: sglang-jax's "
        "own test for it begins with `return`, above the comment 'this kernel "
        "still have some problems', and both the corpus's copy and upstream's "
        "unflattened file raise the same scan pytree-structure TypeError -- "
        "pinned by tests/test_speculative_tpu.py so a fix surfaces",
    "level1/embedding_lookup.py":
        "does not lower on TPU at any shape: the body is a vector gather from "
        "a VMEM ref (`table_ref[idx, :]`), which Mosaic rejects during tracing "
        "with `Cannot do int indexing on TPU` -- rechecked by "
        "tests/test_pallasbench_tpu.py so the exclusion fails if that changes",
}

#: Launch points that are flattened into the corpus and awaiting validation --
#: distinct from `excluded`, which means the corpus decided not to collect them.
#: These are collected; what is missing is a TPU run. The two collectives are
#: further distinguished: they need hardware this corpus never had, so no amount
#: of TPU time on a v6e-1 would move them.
PREPARED = {
    ("MaxText", "attention/ragged_attention.py"):
        "flattened; matches upstream's own references to ~1e-7 under CPU "
        "interpret mode, awaiting a TPU run",
    ("tpu-inference", "structured_sparse_matmul/v1/spmm.py"):
        "flattened; all 8 sparsity arrangements match jnp.dot exactly under "
        "CPU interpret mode, awaiting a TPU run",
    ("tpu-inference", "causal_conv1d/causal_conv1d.py"):
        "flattened; DMA-based, so not checkable under CPU interpret mode -- "
        "awaiting a TPU run",
    ("tpu-inference", "collectives/all_gather_matmul.py"):
        "flattened; needs EXACTLY 8 DEVICES (upstream's own test skips below "
        "that), which this corpus's v6e-1 cannot provide",
    ("tpu-inference", "collectives/hierrs_sc/wrapper.py"):
        "flattened; needs a MULTI-CHIP topology for its Die-to-Die and "
        "Chip-to-Chip pipelining, which a v6e-1 cannot provide",
}

# Settled with the user on 2026-08-03: the v1 grouped matmuls stay out rather
# than take a qwix dependency, since v2 already covers the same contract.
QWIX_BLOCKED = {
    ("MaxText", "megablox/backend.py"),
    ("tokamax", "tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_kernel.py"),
}


def classify(inventory: dict) -> list[dict]:
    import build_inventory as bi

    points = [dict(entry) for entry in inventory["launch_points"]
              if entry["tpu_compatible"]]
    for index, entry in enumerate(points):
        entry["_index"] = index
        entry["status"] = "open"
        entry["reason"] = ""
        for token, reason in EXCLUSIONS.items():
            if token in entry["path"]:
                entry["status"], entry["reason"] = "excluded", reason
        if (entry["repository"], entry["path"]) in PREPARED:
            entry["status"] = "prepared"
            entry["reason"] = PREPARED[(entry["repository"], entry["path"])]
        if (entry["repository"], entry["path"]) in QWIX_BLOCKED:
            entry["status"] = "qwix"
            entry["reason"] = "v1 kernel needs qwix; deliberately deferred"

    for _, directory in bi.CORPUS_DIRECTORIES.items():
        for _, impl in directory["implementations"].items():
            roles = set(impl.get("migrated_roles") or [])
            entries = set(impl.get("entry_points") or [])
            claimed = impl["migrated_launch_points"]
            candidates = [
                e for e in points
                if e["repository"] == impl["repository"]
                and e["path"].startswith(impl["upstream_path"])
                and e["role"] in roles
                and e["status"] == "open"
            ]
            # A directory-prefix claim can cover more launches than it migrated;
            # the declared entry points are what separate them.
            if len(candidates) > claimed and entries:
                narrowed = [e for e in candidates
                            if e["enclosing_function"] in entries]
                if len(narrowed) >= claimed:
                    candidates = narrowed
            candidates.sort(key=lambda e: (e["path"], e["role"], e["line"]))
            if len(candidates) < claimed:
                raise SystemExit(
                    f"{impl['repository']} {impl['upstream_path']}: claims "
                    f"{claimed} migrated launch points but only "
                    f"{len(candidates)} audited launches match"
                )
            for entry in candidates[:claimed]:
                entry["status"] = "migrated"
    return points


def render(points: list[dict], inventory: dict) -> str:
    families = inventory["families"]
    by_family = collections.defaultdict(list)
    for entry in points:
        by_family[entry["family"]].append(entry)

    def open_count(family: str) -> int:
        return sum(1 for e in by_family[family] if e["status"] == "open")

    started = {
        name for name, f in families.items()
        if f.get("migrated_launch_points", 0) > 0
    }
    counts = collections.Counter(e["status"] for e in points)
    total_open = counts["open"]

    lines = [
        "# Kernels remaining to collect",
        "",
        f"Generated by `tools/build_remaining.py` from `inventory.json`. Of "
        f"**{len(points)} audited TPU launch points**: {counts['migrated']} "
        f"migrated, {counts['prepared']} prepared and awaiting a TPU, "
        f"{counts['excluded']} deliberately excluded, "
        f"{counts['qwix']} blocked on qwix, **{total_open} open**.",
        "",
        "A launch point is one `pallas_call` / `pl.kernel` site, not one file —",
        "a single file can hold a forward, a backward-dQ and a backward-dKV",
        "launch.",
        "",
        "Migrated launch points are omitted. `[excluded]` and `[qwix]` rows are",
        "listed so the ledger stays complete, but are not counted as open.",
        "",
    ]

    def section(title: str, note: str, names: list[str]) -> None:
        lines.extend([f"## {title}", "", note, ""])
        for family in names:
            remaining = [e for e in by_family[family] if e["status"] != "migrated"]
            if not remaining:
                continue
            audited = families[family]["audited_tpu_launch_points"]
            migrated = families[family].get("migrated_launch_points", 0)
            suffix = (f" ({migrated} of {audited} migrated)" if migrated
                      else f" of {audited}")
            lines.append(f"### `{family}` — {open_count(family)} open{suffix}")
            lines.append("")
            rows = collections.Counter(
                (e["repository"], e["path"], e["enclosing_function"],
                 e["role"], e["status"], e["reason"])
                for e in remaining
            )
            for (repo, path, fn, role, status, reason), n in sorted(rows.items()):
                mark = {"open": "", "qwix": " `[qwix]`",
                        "excluded": " `[excluded]`",
                        "prepared": " `[prepared]`"}[status]
                lines.append(
                    f"- **{repo}** `{path}` → `{fn}` ({role})"
                    + (f" ×{n}" if n > 1 else "") + mark
                    + (f" — {reason}" if reason else "")
                )
            lines.append("")

    in_started = sorted(
        (f for f in by_family if f in started), key=lambda f: -open_count(f)
    )
    untouched = sorted(
        (f for f in by_family if f not in started), key=lambda f: -open_count(f)
    )
    started_open = sum(open_count(f) for f in in_started)
    untouched_open = sum(open_count(f) for f in untouched)

    section(
        f"A. Families already started — {started_open} open",
        "Depth work: sibling variants and backward passes next to kernels "
        "already validated.",
        in_started,
    )
    section(
        f"B. Untouched families — {untouched_open} open",
        "Breadth work: each needs a new corpus directory, `baseline.py` and "
        "contract.",
        untouched,
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=ROOT / "inventory.json")
    parser.add_argument("--output", type=Path, default=ROOT / "REMAINING.md")
    parser.add_argument("--summary", action="store_true",
                        help="print counts only; do not write the file")
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text())
    points = classify(inventory)
    counts = collections.Counter(e["status"] for e in points)

    declared = inventory["counts"]["migrated_launch_points"]
    if counts["migrated"] != declared:
        raise SystemExit(
            f"classification found {counts['migrated']} migrated launch points "
            f"but inventory.json declares {declared}"
        )

    print(f"audited TPU launch points : {len(points)}")
    for status in ("migrated", "prepared", "excluded", "qwix", "open"):
        print(f"  {status:9s} : {counts[status]:3d}")

    if not args.summary:
        args.output.write_text(render(points, inventory))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
