"""Import the pinned PallasBench task registry into the corpus layout."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


COMMIT = "30a6ee07fd4923f3877906a94002d994e972d6fe"
PROVENANCE_IMPORT = re.compile(
    r"\nfrom pallasbench\.provenance import describe_task as _describe_task\n"
)
PROVENANCE_CALL = re.compile(
    r'\n__doc__ = _describe_task\("[^"]+", __doc__\)\n'
)


def assignments(tree: ast.Module) -> dict[str, object]:
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


def metadata(module_path: Path, category: str, task: str) -> str:
    upstream = (
        f"pallasbench/kernels/{module_path.parent.name}/{module_path.name}"
    )
    return f'''

SOURCE = {{
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "{COMMIT}",
    "path": "{upstream}",
    "backend": "pallas",
    "target": "tpu",
    "contract": "{task}",
    "category": "{category}",
}}
kernel = pallas_kernel
'''


def runner(input_shapes: object, integer_inputs: set[int]) -> str:
    return f'''

def _corpus_main() -> None:
    import json
    import time

    shapes = {input_shapes!r}
    integer_inputs = {sorted(integer_inputs)!r}
    keys = jax.random.split(jax.random.key(42), len(shapes))
    inputs = tuple(
        jax.random.randint(key, shape, 0, 4, dtype=jnp.int32)
        if index in integer_inputs
        else jax.random.normal(key, shape, dtype=jnp.float32)
        for index, (key, shape) in enumerate(zip(keys, shapes))
    )
    start = time.perf_counter()
    output = kernel(*inputs)
    output.block_until_ready()
    print(json.dumps({{
        "implementation": "pallasbench",
        "contract": SOURCE["contract"],
        "shape": list(output.shape),
        "compile_and_run_ms": (time.perf_counter() - start) * 1e3,
    }}))


if __name__ == "__main__":
    _corpus_main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    kernel_root = args.checkout / "pallasbench" / "kernels"
    baseline_source = (
        args.checkout / "pallasbench" / "baselines" / "jax_baseline.py"
    ).read_text()
    for source in sorted(kernel_root.glob("level[123]/*.py")):
        if source.name == "__init__.py":
            continue
        original = source.read_text()
        values = assignments(ast.parse(original))
        task = str(values["task_name"])
        category = str(values["category"])
        input_shapes = values["input_shapes"]
        family = args.destination / category / task
        family.mkdir(parents=True, exist_ok=True)

        optimized = PROVENANCE_IMPORT.sub("\n", original)
        optimized = PROVENANCE_CALL.sub("\n", optimized)
        optimized += metadata(source, category, task)
        integer_inputs = {
            "embedding_lookup": {1},
            "one_hot": {0},
            "nucleotide_onehot": {0},
        }.get(task, set())
        optimized += runner(input_shapes, integer_inputs)
        output = family / "pallasbench_optimized.py"
        if output.exists() and task == "flash_attention":
            # This family is the hand-curated protocol example.
            pass
        else:
            output.write_text(optimized)

        baseline_name = {
            "tanh": "jax_tanh",
        }.get(task, f"jax_{task}")
        baseline = baseline_source + f"\n\nkernel = {baseline_name}\n"
        baseline += f'''
SOURCE = {{
    "repository": "https://github.com/Tyronita/PallasBench",
    "commit": "{COMMIT}",
    "path": "pallasbench/baselines/jax_baseline.py",
    "backend": "jax",
    "target": "portable",
    "contract": "{task}",
}}
'''
        baseline_path = family / "baseline.py"
        if not baseline_path.exists():
            baseline_path.write_text(baseline)
        (family / "__init__.py").touch()


if __name__ == "__main__":
    main()
