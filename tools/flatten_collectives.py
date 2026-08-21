"""Flatten tpu-inference's two collective kernels.

**Neither can be validated on this corpus's hardware, and that is a property of
the kernels rather than of the week the TPU was down.** Both are genuinely
multi-device:

``all_gather_matmul``
    A ring all-gather fused into a matmul: each device DMAs its shard to its
    left and right neighbours while multiplying, using 14 send/recv semaphores
    and `lax.axis_index` to find its place in the ring. Upstream's own test
    opens `if jax.device_count() != 8: self.skipTest(...)`.

``hierarchical_reduce_scatter_local``
    A recursive-halving reduce-scatter running on **SparseCore**, pipelined
    across Die-to-Die and Chip-to-Chip ICI, with devices ordered by physical
    topology coordinates.

The corpus validated on a v6e-1 -- one chip, one device -- where a ring has no
neighbours. So these are flattened and carried with full provenance, and
recorded UNVALIDATED with zero migrated launch points. The blocker is stated in
the ledger as a hardware requirement, not as unfinished work.

A note on the flattening, because the two differ in a way that matters. The
`hierrs_sc` modules are imported as `from ...config import Config`, so the
`config.num_chips` reads scattered through them are attribute access on a
**Config instance**, not a module qualifier -- stripping a `config.` prefix
would corrupt every one of them. Only `all_gather_matmul.py` uses true module
qualifiers (`util.`, `all_gather_matmul_tuned_block_sizes.`), and those are
stripped with the scope-aware helper rather than a blind regex.

    python tools/flatten_collectives.py <tpu-inference-checkout> <corpus-dir>
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from flatten_gdn import strip_module_prefixes, unbound_module_refs


REPOSITORY = "https://github.com/vllm-project/tpu-inference"
COMMIT = "8b9c90928c94c7230d1bc891534a301510a6a30d"
BASE = "tpu_inference/kernels/collectives"

SOURCES = {
    "all_gather_matmul": {
        # Dependency order; the entry-point module last.
        "modules": ("util.py", "all_gather_matmul_tuned_block_sizes.py",
                    "all_gather_matmul.py"),
        "prefixes": ("util", "all_gather_matmul_tuned_block_sizes"),
        "entry": "all_gather_matmul",
        "launch": "_all_gather_matmul_call",
        "contract": "all_gather_matmul",
        "file": "tpu_inference_all_gather_matmul_optimized.py",
        "requires": (
            "exactly 8 devices -- upstream's own test skips below that. The\n"
            "kernel indexes a ring with `lax.axis_index` and exchanges shards\n"
            "with its left and right neighbours over 14 send/recv semaphores."
        ),
        "summary": (
            "An all-gather fused into a matmul. Rather than gathering every\n"
            "shard and then multiplying, each device multiplies the shard it\n"
            "holds while the next one is still in flight -- the communication\n"
            "hides under the MXU work instead of preceding it."
        ),
    },
    "hierarchical_reduce_scatter": {
        "modules": ("hierrs_sc/config.py", "hierrs_sc/topology.py",
                    "hierrs_sc/dma_pipeline.py", "hierrs_sc/kernel.py",
                    "hierrs_sc/wrapper.py"),
        # Every hierrs_sc import is `from X import Name`, so there is no module
        # qualifier to strip -- and `config.` is an instance, not a module.
        "prefixes": (),
        "entry": "hierarchical_reduce_scatter_local",
        "launch": "hierarchical_reduce_scatter_local",
        "contract": "hierarchical_reduce_scatter",
        "file": "tpu_inference_hierarchical_reduce_scatter_optimized.py",
        "requires": (
            "a multi-chip topology. The kernel runs on SparseCore and\n"
            "pipelines Die-to-Die against Chip-to-Chip ICI, with devices\n"
            "ordered by physical topology coordinates."
        ),
        "summary": (
            "Reduce-scatter by recursive halving, on SparseCore rather than\n"
            "the TensorCore -- so the reduction runs beside the model's own\n"
            "compute instead of competing with it. Two-stage pipelining\n"
            "overlaps the intra-die and inter-chip hops with local adds."
        ),
    },
}

LOCAL_IMPORT = re.compile(
    r"^from tpu_inference\.kernels\.collectives[.\w]* import \([^)]*\)\s*$"
    r"|^from tpu_inference\.kernels\.collectives[.\w]* import .*$",
    re.M,
)

HEADER = '''"""Standalone vLLM tpu-inference {title}.

Source:
  repository: {repository}
  commit: {commit}
  paths:
{path_list}
  transformation: the modules above were concatenated in dependency order and
    the repo-local imports between them removed.{prefix_note} No kernel body
    was touched.

Entry point: ``{entry}`` (also exported as ``kernel``); the audited Pallas
launch is ``{launch}``.

Contract ``{contract}``.
{summary}

**NOT VALIDATED, and not for want of trying.** This kernel requires
{requires}
This corpus was built and validated on a **v6e-1** -- a single chip, a single
device -- where that ring has no neighbours and the topology has one node. So
the file is carried here with its provenance intact and recorded UNVALIDATED
with zero migrated launch points. It is a hardware requirement, not unfinished
work: no amount of TPU time on a one-device machine would close it.

Native shape: none declared.
"""

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    # `path` is the entry-point module, matching every other file in the
    # corpus; `also_inlines` lists what was concatenated ahead of it.
    "path": "{primary}",
    "also_inlines": {inlined!r},
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{contract}",
    "family": "collective_matmul",
    "launch_points": 1,
    "native_shape": None,
    "requires_devices": {devices},
    "validated": False,
}}

'''


def flatten(name: str, checkout: Path) -> str:
    spec = SOURCES[name]
    chunks, lead = [], ""
    for index, module in enumerate(spec["modules"]):
        text = (checkout / BASE / module).read_text()
        tree = ast.parse(text)
        first = tree.body[0]
        lines = text.splitlines(keepends=True)
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            head, body = "".join(lines[:first.lineno - 1]), "".join(lines[first.end_lineno:])
        else:
            head, body = "".join(lines[:first.lineno - 1]), "".join(lines[first.lineno - 1:])
        if index == 0:
            lead = head  # keep the first module's licence header
        body = LOCAL_IMPORT.sub("", body)
        chunks.append(f"\n# ---- flattened from {BASE}/{module} ----\n\n"
                      + body.strip("\n") + "\n")

    combined = "".join(chunks)
    if spec["prefixes"]:
        combined, shadowed = strip_module_prefixes(combined, spec["prefixes"])
        leftover = unbound_module_refs(combined, spec["prefixes"])
        if leftover:
            raise ValueError(f"{name}: unresolved module refs {leftover}")

    for node in ast.walk(ast.parse(combined)):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.split(".")[0] == "tpu_inference":
            raise ValueError(f"{name}: unresolved import {node.module}")
    launches = combined.count("pl.pallas_call") + combined.count("pl.kernel(")
    if launches != 1:
        raise ValueError(f"{name}: expected 1 launch, found {launches}")
    for needed in (f"def {spec['entry']}(", f"def {spec['launch']}("):
        if needed not in combined:
            raise ValueError(f"{name}: lost {needed}")

    paths = [f"{BASE}/{m}" for m in spec["modules"]]
    header = HEADER.format(
        title=name.replace("_", " "), repository=REPOSITORY, commit=COMMIT,
        path_list="\n".join(f"    {p}" for p in paths),
        primary=paths[-1], inlined=paths[:-1],
        entry=spec["entry"], launch=spec["launch"], contract=spec["contract"],
        summary=spec["summary"], requires=spec["requires"],
        devices=8 if name == "all_gather_matmul" else None,
        prefix_note=(f" The `{'`/`'.join(spec['prefixes'])}` module\n"
                     "    qualifiers were dropped with a scope-aware pass, so a"
                     " local name that\n    happens to match is left alone."
                     if spec["prefixes"] else ""),
    )
    result = lead + header + combined.strip("\n") + f"\n\n\nkernel = {spec['entry']}\n"
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="tpu-inference checkout root")
    parser.add_argument("corpus_dir", type=Path,
                        help="kernels/collectives/collective_matmul")
    args = parser.parse_args()
    args.corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.corpus_dir / "__init__.py").touch()
    for name, spec in SOURCES.items():
        (args.corpus_dir / spec["file"]).write_text(flatten(name, args.checkout))
    print(f"wrote {len(SOURCES)} kernels into {args.corpus_dir}")


if __name__ == "__main__":
    main()
