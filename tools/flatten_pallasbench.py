"""Flatten the PallasBench kernel suite into the corpus.

PallasBench is the most uniform source the corpus draws on, and this script
leans on that rather than fighting it.  Every kernel file:

* is self-contained apart from one import, ``pallasbench.provenance``, used
  only to prepend a task description to ``__doc__``;
* holds exactly **one** ``pl.pallas_call``;
* exposes ``pallas_<task>`` plus ``task_name``, ``input_shapes``, ``category``
  and ``level`` as module constants.

Upstream also ships ``pallasbench/baselines/jax_baseline.py`` -- "pure JAX
reference implementations for all PallasBench tasks ... no Pallas" -- with a
``jax_<task>`` for every ``pallas_<task>``, and ``utils.generate_inputs``, the
generator its own benchmark harness uses.  So the references and the input
specification are upstream's, not the corpus's: nothing here is re-derived.

``pallasbench_tasks.json`` beside this file records, per task, the upstream
file, entry point, corpus category/family, native ``input_shapes``, and the
``input_dtypes``/``input_ranges`` upstream attaches to the three integer tasks
and the two that need a positive domain (``log``, ``rsqrt``).

Two further keys are the corpus's, not upstream's, and both exist to keep a
disagreement with upstream visible rather than quietly resolved:

``validation``
    Seventeen tasks do not compile at their native shape on a v6e, against two
    different limits.  Eleven exhaust **total** VMEM -- a level-1 elementwise
    kernel maps a whole 4096x4096 f32 array into VMEM at once, so input plus
    output is 128 MiB against 127.94 MiB.  Six exceed the 32 MiB **scoped**
    allocation instead, because of intermediates the kernel materialises:
    softmax's max/sum/exp pair, and `triangle_update`'s (c, n, n, n) outer
    product.  Each carries a smaller ``shapes`` used for correctness plus the
    measured ``reason``.  The native shape stays in the kernel file's
    ``SOURCE["native_shape"]`` and in the baseline's ``native_shapes``, so
    nothing is silently restated.

``excluded``
    A task that does not lower on TPU at all, with the evidence.  Excluded
    tasks are not written to the corpus.

Usage::

    python tools/flatten_pallasbench.py kernels <pallasbench-checkout> <corpus-root>
    python tools/flatten_pallasbench.py baselines <pallasbench-checkout> <corpus-root>
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import textwrap


HERE = Path(__file__).parent
TASKS = json.loads((HERE / "pallasbench_tasks.json").read_text())

REPOSITORY = "https://github.com/Tyronita/PallasBench"
COMMIT = "30a6ee07fd4923f3877906a94002d994e972d6fe"

#: The one repo-local import, and the line that consumes it.
PROVENANCE_IMPORT = re.compile(
    r"^from pallasbench\.provenance import describe_task as _describe_task\s*$",
    re.M,
)
PROVENANCE_DOC = re.compile(r"^__doc__ = _describe_task\([^)]*\)\s*$", re.M)

KERNEL_HEADER = '''"""Standalone PallasBench {task} kernel (level {level}).

Source:
  repository: {repository}
  commit: {commit}
  path: pallasbench/kernels/{file}
  transformation: self-contained upstream apart from
    `pallasbench.provenance.describe_task`, which only prepends a task
    description to `__doc__`; that import and the line using it are removed.
    The kernel body is unmodified.

Entry point: ``{entry}`` (also exported as ``kernel``).

{contract}

Native shape: {shapes}.{validation}
"""

from __future__ import annotations

SOURCE = {{
    "repository": "{repository}",
    "commit": "{commit}",
    "path": "pallasbench/kernels/{file}",
    "backend": "pallas-mosaic-tpu",
    "target": "tpu",
    "contract": "{task}",
    "family": "{family}",
    "level": {level},
    "launch_points": 1,
    "native_shape": {shapes},
    "validation_shape": {validation_shape},
    "validation_reason": {validation_reason!r},
}}

'''

CONTRACT_NOTE = (
    "Contract ``{task}``, in the ``{family}`` family.  The reference is "
    "upstream's own ``jax_{task}`` from "
    "`pallasbench/baselines/jax_baseline.py`, carried in this directory's "
    "`baseline.py`; ``create_inputs(\"{task}\")`` there reproduces the dtypes "
    "and value ranges upstream's benchmark harness uses, so a comparison here "
    "is against upstream's definition of the task rather than a corpus "
    "reading of it."
)

#: Appended to the kernel docstring when validation runs below native shape.
VALIDATION_NOTE = """Validated at {shapes} instead, because {reason}
The kernel itself is untouched -- only the shape it is called with differs, and
the native shape stays recorded above and in ``SOURCE``."""

BASELINE_HEADER = '''"""JAX references for the PallasBench {family} kernels in this directory.

Every function here is upstream's own, copied from
``pallasbench/baselines/jax_baseline.py`` at the commit below -- a module whose
docstring reads "pure JAX reference implementations ... Each function uses only
jax.numpy / jax.lax / jax.nn -- no Pallas".  ``generate_inputs`` is likewise
upstream's, from ``pallasbench/utils.py``: it is what PallasBench's own
benchmark harness feeds these kernels, including the integer dtypes and the
positive-domain ranges that ``log`` and ``rsqrt`` need.

Carrying upstream's reference rather than writing one matters here for the same
reason it did in the attention families: the risk is not that ``jnp.maximum(x,
0)`` is hard to write, it is that a corpus-written reference quietly encodes a
*different task* than the kernel implements.

Source:
  repository: {repository}
  commit: {commit}
  paths:
    pallasbench/baselines/jax_baseline.py  (jax_* references)
    pallasbench/utils.py                   (generate_inputs)

Tasks in this directory: {tasks}
"""

from __future__ import annotations

SOURCE = {{
    "kind": "reference",
    "repository": "{repository}",
    "commit": "{commit}",
    "backend": "jax",
    "target": "portable",
    "family": "{family}",
    "contracts": {tasks!r},
}}

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


#: Per task: the shapes correctness runs at, PallasBench's own ``native_shapes``
#: (identical unless ``reason`` says why they had to differ), and the dtypes and
#: value ranges upstream attaches to the tasks that need them.  ``None`` for a
#: dtype or range means float32 standard normal, which is what PallasBench uses
#: everywhere else.
TASK_INPUTS = {task_inputs}


def create_inputs(task: str, seed: int = 0) -> list[jax.Array]:
    """Upstream's inputs for ``task``, at the shape correctness runs at."""
    spec = TASK_INPUTS[task]
    return generate_inputs(
        [tuple(s) for s in spec["input_shapes"]], seed=seed,
        dtypes=spec["dtypes"], ranges=spec["ranges"],
    )


def runs_below_native_shape(task: str) -> str | None:
    """Why ``task`` is validated below its native shape, or None if it is not."""
    return TASK_INPUTS[task]["reason"]

'''


def _wrap(text: str) -> str:
    """Fill to the corpus's 79 columns, leaving `--` and long paths intact."""
    return textwrap.fill(text, width=79, break_long_words=False,
                         break_on_hyphens=False)


def _extract(source: str, names: list[str]) -> str:
    """Pull named top-level functions out of a module, in file order."""
    tree = ast.parse(source)
    wanted = set(names)
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segment = ast.get_source_segment(source, node)
            decorators = [
                ast.get_source_segment(source, d) for d in node.decorator_list
            ]
            chunks.append(
                "\n".join(f"@{d}" for d in decorators) + ("\n" if decorators else "")
                + segment
            )
            wanted.discard(node.name)
    if wanted:
        raise ValueError(f"not found in baseline module: {sorted(wanted)}")
    return "\n\n\n".join(chunks)


def flatten_kernel(task: str, checkout: Path) -> str:
    spec = TASKS[task]
    text = (checkout / "pallasbench" / "kernels" / spec["file"]).read_text()
    text, n_import = PROVENANCE_IMPORT.subn("", text)
    text, n_doc = PROVENANCE_DOC.subn("", text)
    if (n_import, n_doc) != (1, 1):
        raise ValueError(
            f"{task}: expected one provenance import and one __doc__ line, "
            f"found {n_import} and {n_doc}"
        )
    if "pallasbench" in text:
        offender = next(l for l in text.splitlines() if "pallasbench" in l)
        raise ValueError(f"{task}: unresolved upstream reference: {offender!r}")
    if text.count("pl.pallas_call") != 1:
        raise ValueError(
            f"{task}: expected 1 pallas_call, found {text.count('pl.pallas_call')}"
        )
    if f"def {spec['entry']}(" not in text:
        raise ValueError(f"{task}: lost entry point {spec['entry']}")

    validation = spec.get("validation")
    header = KERNEL_HEADER.format(
        task=task, level=spec["level"], repository=REPOSITORY, commit=COMMIT,
        file=spec["file"], entry=spec["entry"], family=spec["family"],
        shapes=spec["input_shapes"],
        contract=_wrap(CONTRACT_NOTE.format(task=task, family=spec["family"])),
        validation=(
            "\n" + _wrap(VALIDATION_NOTE.format(shapes=validation["shapes"],
                                                reason=validation["reason"]))
            if validation else ""
        ),
        validation_shape=(validation["shapes"] if validation
                          else spec["input_shapes"]),
        validation_reason=validation["reason"] if validation else None,
    )
    result = header + text.strip("\n") + f"\n\n\nkernel = {spec['entry']}\n"
    ast.parse(result)
    return result


def migrated(family: str) -> list[str]:
    """The tasks of ``family`` the corpus carries: not excluded, not preexisting."""
    if family == "flash_attention":
        return []  # already migrated as pallasbench_optimized.py, with its own tests
    return sorted(t for t, s in TASKS.items()
                  if s["family"] == family and not s.get("excluded"))


def build_baseline(family: str, checkout: Path) -> str:
    tasks = migrated(family)
    baseline_src = (
        checkout / "pallasbench" / "baselines" / "jax_baseline.py"
    ).read_text()
    utils_src = (checkout / "pallasbench" / "utils.py").read_text()

    task_inputs = {
        t: {
            "input_shapes": (TASKS[t]["validation"]["shapes"]
                             if TASKS[t].get("validation")
                             else TASKS[t]["input_shapes"]),
            "native_shapes": TASKS[t]["input_shapes"],
            "reason": (TASKS[t]["validation"]["reason"]
                       if TASKS[t].get("validation") else None),
            "dtypes": TASKS[t]["dtypes"],
            "ranges": TASKS[t]["ranges"],
        }
        for t in tasks
    }
    # A Python literal, not JSON: json.dumps writes `null`, which does not
    # parse.  pprint keeps it readable at these sizes.
    import pprint

    header = BASELINE_HEADER.format(
        family=family, repository=REPOSITORY, commit=COMMIT, tasks=tasks,
        task_inputs=pprint.pformat(task_inputs, indent=1, width=76,
                                   sort_dicts=True),
    )
    body = (
        _extract(utils_src, ["generate_inputs"])
        + "\n\n\n"
        + _extract(baseline_src, [f"jax_{t}" for t in tasks])
    )
    footer = (
        "\n\n\n#: The corpus protocol: one named reference per task.\n"
        "REFERENCES = {\n"
        + "".join(f'    "{t}": jax_{t},\n' for t in tasks)
        + "}\n"
    )
    result = header + body + footer
    ast.parse(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("kernels", "baselines"))
    parser.add_argument("checkout", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--only", help="one task (kernels) or family (baselines)")
    args = parser.parse_args()

    written = skipped = 0
    if args.what == "kernels":
        for task, spec in sorted(TASKS.items()):
            if args.only and task != args.only:
                continue
            if task not in migrated(spec["family"]):
                skipped += 1
                continue
            out = (args.corpus / "kernels" / spec["category"] / spec["family"]
                   / f"pallasbench_{task}_optimized.py")
            out.parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "__init__.py").touch()
            out.write_text(flatten_kernel(task, args.checkout))
            written += 1
    else:
        families = sorted({s["family"] for s in TASKS.values()})
        for family in families:
            if args.only and family != args.only:
                continue
            if not migrated(family):
                skipped += 1
                continue
            category = next(s["category"] for s in TASKS.values()
                            if s["family"] == family)
            out = args.corpus / "kernels" / category / family / "baseline.py"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(build_baseline(family, args.checkout))
            written += 1
    print(f"wrote {written} {args.what}, skipped {skipped}")


if __name__ == "__main__":
    main()
