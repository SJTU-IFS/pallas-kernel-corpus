"""Check that every migrated launch point is actually exercised by a test.

The ledger records `migrated_launch_points` per implementation, and the test
suite is what backs the `correctness: PASS` beside it.  Nothing connected the
two, and twice that let a launch point be counted as migrated while no test
ever ran it:

* grouped matmul's ``tgmm_v2`` -- flattened alongside ``gmm_v2``, counted as a
  migrated *backward* launch point, with no test differentiating it;
* flash attention's PallasBench ``dense_2d`` kernel -- its ``PASS`` came from
  the profiling run, which is not part of the suite.

Both were found by recording which Pallas launches actually fire.  This script
makes that repeatable.  It runs pytest with a wrapper around ``pl.pallas_call``
and ``pl.kernel`` that records the *source file and qualified name* of every
kernel body launched, then compares that against the launch sites the corpus
files contain.

Requires a TPU and takes about as long as the suite.  Run it after adding an
implementation, not on every edit::

    uv run --frozen --with pytest python tools/launch_coverage.py
    uv run --frozen --with pytest python tools/launch_coverage.py --tests tests/test_grouped_matmul_tpu.py
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).parents[1]

#: Written to a temp dir and passed to pytest with ``-p``.  It has to patch
#: before the corpus modules are imported, which a plugin does and a fixture
#: does not.
PLUGIN = '''
import atexit, functools, os
import jax.experimental.pallas as pl

SEEN = set()
LOG = os.environ["LAUNCH_LOG"]


def _identify(body):
    while isinstance(body, functools.partial):
        body = body.func
    code = getattr(body, "__code__", None)
    name = getattr(body, "__qualname__", getattr(body, "__name__", "?"))
    return f"{getattr(code, 'co_filename', '?')}::{name}"


def _wrap(original):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        body = args[0] if args else kwargs.get("kernel")
        if callable(body):
            SEEN.add(_identify(body))
            return original(*args, **kwargs)
        # `pl.kernel(out_type=..., mesh=...)` is a decorator FACTORY: the body
        # arrives on the returned decorator, not here.  The SparseCore kernels
        # all launch this way.
        made = original(*args, **kwargs)
        if callable(made):
            @functools.wraps(made)
            def decorator(fn, *rest, **kw):
                SEEN.add(_identify(fn))
                return made(fn, *rest, **kw)
            return decorator
        return made
    return wrapper


pl.pallas_call = _wrap(pl.pallas_call)
for _api in ("kernel", "core_map"):
    if hasattr(pl, _api):
        setattr(pl, _api, _wrap(getattr(pl, _api)))


@atexit.register
def _dump():
    with open(LOG, "w") as handle:
        handle.write("\\n".join(sorted(SEEN)))
'''

LAUNCH_APIS = {"pl.pallas_call", "pl.kernel", "pl.core_map",
               "plsc.kernel", "core_map_kernel"}


def launch_sites(path: Path) -> list[str]:
    """Qualified names of the kernel bodies this file launches."""
    text = path.read_text()
    found: list[str] = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            deeper = stack
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                deeper = (*stack, child.name)
                # `@pl.kernel(...)` decorates the body rather than taking it as
                # an argument; the decorated function IS the kernel body.
                for deco in child.decorator_list:
                    call = deco.func if isinstance(deco, ast.Call) else deco
                    if (isinstance(call, ast.Attribute)
                            and f"{getattr(call.value, 'id', '?')}.{call.attr}"
                            in LAUNCH_APIS):
                        found.append(child.name)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                value = getattr(child.func.value, "id", "?")
                if f"{value}.{child.func.attr}" in LAUNCH_APIS:
                    body = child.args[0] if child.args else None
                    if body is not None:
                        found.append(_body_name(body, stack))
            walk(child, deeper)

    walk(ast.parse(text), ())
    return found


def _body_name(node, stack) -> str:
    """Best-effort name of the kernel body passed to a launch call."""
    while isinstance(node, ast.Call):  # functools.partial(body, ...)
        node = node.args[0] if node.args else None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return f"<anonymous in {'.'.join(stack) or 'module'}>"


def implementations() -> list[dict]:
    inventory = json.loads((ROOT / "inventory.json").read_text())
    out = []
    for family in inventory["families"].values():
        for impl in family.get("corpus_implementations", []):
            if impl["migrated_launch_points"]:
                out.append(impl)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="tests", help="pytest target")
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="report instead of exiting non-zero",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp)
        (plugin_dir / "_launch_plugin.py").write_text(PLUGIN)
        log = plugin_dir / "launches.txt"
        env = {**os.environ, "LAUNCH_LOG": str(log),
               "PYTHONPATH": f"{plugin_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "_launch_plugin",
             args.tests, "-q"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        print(result.stdout.strip().splitlines()[-1] if result.stdout else "")
        fired = set(log.read_text().splitlines()) if log.exists() else set()

    by_file: dict[str, set[str]] = {}
    for entry in fired:
        if "::" not in entry:
            continue
        filename, _, name = entry.partition("::")
        by_file.setdefault(Path(filename).resolve().as_posix(), set()).add(name)

    missing: list = []
    unknown: list = []
    for impl in sorted(implementations(), key=lambda i: (i["family"], i["name"])):
        path = (ROOT / impl["directory"] / impl["file"]).resolve()
        sites = launch_sites(path)
        seen = by_file.get(path.as_posix(), set())
        claimed = impl["migrated_launch_points"]
        # Compare COUNTS, not names.  A kernel body's static name is usually a
        # local ("kernel") while its runtime __qualname__ is qualified
        # ("f.<locals>.kernel"), so matching them pairwise produces noise.  The
        # signal that caught both real gaps is simply: this file contains N
        # launch sites and only M of them ever fired.
        #
        # The comparison is against sites in the file, not against the ledger's
        # claim: one flattened site can cover several audited launch points
        # (sglang-jax's kv-cache update is two upstream launches over one
        # pallas_call), so the claim is reported but is not the criterion.
        if not sites:
            # The scanner recognises pl.pallas_call and @pl.kernel decorators.
            # The SparseCore gathers launch through an inlined `pl.core_map`
            # helper it cannot see, so it has no basis to judge those files --
            # say so instead of reporting a gap that is not one.
            status = "?  "
            unknown.append(impl)
        elif len(seen) < len(sites):
            # A launch is identified by its kernel body, so two sites passing
            # the SAME body collapse to one.  sglang-jax's kv-cache update does
            # exactly that -- `kv_cache_update_impl` and the shard_mapped
            # wrapper both launch `kv_cache_update_kernel` -- so a shortfall
            # here means "look", not "broken".  That pair is covered by
            # test_sglang_both_launch_points_reach_pallas, which asserts the
            # launch count on each entry point directly.
            status = "GAP"
            missing.append((impl, sites, seen))
        else:
            status = "ok "
        print(f"  {status} {impl['family']:24s} {impl['name']:26s} "
              f"claimed {claimed}, fired {len(seen)} of {len(sites)} sites")

    if unknown:
        print("\nNo launch site recognised -- not judged (these launch through "
              "`pl.core_map`, which the scanner does not model; their tests "
              "assert the launch count directly instead):")
        for impl in unknown:
            print(f"  {impl['family']}/{impl['name']} ({impl['file']})")

    if missing:
        print("\nFewer distinct kernel bodies fired than the file launches. "
              "Two sites sharing one body collapse here, so check whether a "
              "test reaches each entry point before concluding it is a gap:")
        for impl, sites, seen in missing:
            print(f"  {impl['family']}/{impl['name']} ({impl['file']}): "
                  f"{len(seen)} of {len(sites)} launch sites fired"
                  + (f"; fired: {sorted(seen)}" if seen else ""))
        if not args.allow_missing:
            raise SystemExit(1)
    else:
        print("\nEvery migrated launch point fires at least once.")


if __name__ == "__main__":
    main()
