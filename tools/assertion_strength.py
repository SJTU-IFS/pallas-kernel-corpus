"""Find test comparisons that a wrong kernel would still pass.

A green assertion is not evidence if the thing being compared is degenerate.
The DeepSeek-V4 sliding-window MLA test is the worked example: its attention
sinks are drawn from [200, 500] while the logits reach ~135, so ``exp(sink - m)``
swamps the softmax denominator and **every output on both sides collapses to
about 1e-26**.  ``assert_allclose(rtol=0.1, atol=0.1)`` then compares two arrays
of zeros and passes however wrong the kernel is.

The criterion here is mechanical rather than a judgement call:

    a comparison is VACUOUS if a kernel that returned all zeros --
    or a kernel that returned one constant everywhere -- would pass it.

For ``assert_allclose(actual, desired, rtol, atol)`` that is just
``allclose(zeros, desired)`` and ``allclose(full(mean), desired)``.  For the
cosine-threshold assertions the corpus uses, magnitude does not matter (cosine
is scale-invariant) but *direction* does: a nearly-constant reference is matched
by any constant vector, so the reported signal there is the reference's relative
spread.

Scalar references are reported apart from both.  A 0-d value *is* its own mean,
so the constant test is true of every scalar comparison and would flood the
WEAK list with a category error.  The question worth asking about a scalar is a
different one -- a loss reduced to one number is a thin check on a kernel that
computed B x V logits to produce it -- so scalars are only reported when no
other comparison in the same test covers an array.

Runs the suite once and reports; it does not fail the build.  Requires a TPU::

    uv run --frozen --with pytest python tools/assertion_strength.py
    uv run --frozen --with pytest python tools/assertion_strength.py --tests tests/test_mla_attention_tpu.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).parents[1]

PLUGIN = r'''
import atexit, json, os
import numpy as np
import numpy.testing as npt

RECORDS = []
LOG = os.environ["ASSERTION_LOG"]
CURRENT = {"test": "<setup>"}


def _stats(desired, rtol, atol, kind):
    a = np.asarray(desired)
    if a.dtype == np.bool_:
        return None
    # NOT `np.issubdtype(a.dtype, np.number)`.  ml_dtypes' bfloat16 and the fp8
    # types -- which is what half the corpus's kernels actually return -- are
    # registered with kind "V", so that test is False for them and every
    # comparison made directly on a bf16 or fp8 array was silently dropped.
    # The sweep reported "clean" while not looking at them.  Converting is the
    # real question: anything that becomes float64 can be analysed.
    try:
        a = np.asarray(a, np.float64)
    except (TypeError, ValueError):
        return None
    if a.size == 0:
        return None
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    peak = float(np.max(np.abs(finite)))
    mean = float(np.mean(finite))
    # Would a kernel that returned zeros pass?  Would one returning a single
    # constant pass?  Those are the two degenerate outputs worth ruling out.
    zeros_pass = bool(np.allclose(np.zeros_like(a), a, rtol=rtol, atol=atol,
                                  equal_nan=True))
    const_pass = bool(np.allclose(np.full_like(a, mean), a, rtol=rtol,
                                  atol=atol, equal_nan=True))
    spread = float(np.std(finite) / (abs(mean) + 1e-30))
    return {
        "test": CURRENT["test"], "kind": kind, "shape": list(a.shape),
        "peak": peak, "mean_abs": float(np.mean(np.abs(finite))),
        "frac_zero": float(np.mean(finite == 0)),
        "rtol": rtol, "atol": atol,
        "zeros_would_pass": zeros_pass, "constant_would_pass": const_pass,
        "relative_spread": spread,
    }


_allclose = npt.assert_allclose
_array_equal = npt.assert_array_equal


def assert_allclose(actual, desired, rtol=1e-7, atol=0, *a, **k):
    rec = _stats(desired, rtol, atol, "assert_allclose")
    if rec:
        RECORDS.append(rec)
    return _allclose(actual, desired, rtol, atol, *a, **k)


def assert_array_equal(x, y, *a, **k):
    rec = _stats(y, 0.0, 0.0, "assert_array_equal")
    if rec:
        RECORDS.append(rec)
    return _array_equal(x, y, *a, **k)


npt.assert_allclose = assert_allclose
npt.assert_array_equal = assert_array_equal
np.testing.assert_allclose = assert_allclose
np.testing.assert_array_equal = assert_array_equal


def pytest_runtest_setup(item):
    CURRENT["test"] = item.nodeid


def pytest_collection_modifyitems(items):
    """Wrap each test module's own `cosine` helper.

    The corpus asserts `cosine(actual, expected) > 0.9999` far more often than
    it calls assert_allclose, and those calls are module-local functions rather
    than a shared library entry point.
    """
    seen = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        for helper in ("cosine", "_close"):
            _wrap_helper(module, helper)


def _wrap_helper(module, helper):
        original = getattr(module, helper, None)
        if not callable(original):
            return

        def wrapper(actual, expected, *rest, _orig=original, _kind=helper,
                    **kw):
            value = _orig(actual, expected, *rest, **kw)
            e = np.asarray(expected, np.float64).ravel()
            finite = e[np.isfinite(e)]
            if finite.size:
                mean = float(np.mean(finite))
                RECORDS.append({
                    "test": CURRENT["test"], "kind": _kind,
                    "shape": list(np.asarray(expected).shape),
                    "peak": float(np.max(np.abs(finite))),
                    "mean_abs": float(np.mean(np.abs(finite))),
                    "frac_zero": float(np.mean(finite == 0)),
                    "rtol": None, "atol": None,
                    "zeros_would_pass": bool(np.max(np.abs(finite)) == 0),
                    "constant_would_pass": bool(
                        np.std(finite) <= 1e-6 * (abs(mean) + 1e-30)),
                    "relative_spread": float(
                        np.std(finite) / (abs(mean) + 1e-30)),
                    "value": float(value),
                })
            return value

        setattr(module, helper, wrapper)


@atexit.register
def _dump():
    with open(LOG, "w") as handle:
        json.dump(RECORDS, handle)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", default="tests")
    parser.add_argument(
        "--spread-threshold", type=float, default=1e-3,
        help="flag cosine comparisons whose reference varies less than this, "
             "relative to its mean",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp)
        (plugin_dir / "_assert_plugin.py").write_text(PLUGIN)
        log = plugin_dir / "records.json"
        env = {
            **os.environ,
            "ASSERTION_LOG": str(log),
            "PYTHONPATH": f"{plugin_dir}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "_assert_plugin",
             args.tests, "-q"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        tail = result.stdout.strip().splitlines()
        print(tail[-1] if tail else "(no pytest output)")
        records = json.loads(log.read_text()) if log.exists() else []

    print(f"\n{len(records)} numeric comparisons recorded\n")

    for row in records:
        if row["kind"] == "_close":
            # `_close(got, want, tol)` passes when max|got - want| <= tol *
            # max(1, max|want|); a zeroed output passes exactly when the
            # reference is itself within that band of zero.
            row["zeros_would_pass"] = (
                row["peak"] <= 2e-2 * max(1.0, row["peak"]))

    # A 0-d reference is its own mean, so "a constant output would pass" is
    # true of every scalar comparison and says nothing about its strength.
    # Reporting those as WEAK would bury the real findings under a category
    # error, so they are counted separately -- but they are still shown,
    # because collapsing a large computation to one number *is* worth seeing.
    # What redeems such a test is another comparison beside it over an array,
    # so that is what the report says about each one.
    def is_scalar(row: dict) -> bool:
        return not row["shape"]

    array_comparisons = {r["test"] for r in records if not is_scalar(r)}

    vacuous = [r for r in records if r["zeros_would_pass"]]
    constant = [r for r in records
                if r["constant_would_pass"] and not r["zeros_would_pass"]
                and not is_scalar(r)]
    scalar = [r for r in records if is_scalar(r) and not r["zeros_would_pass"]]
    unbacked = [r for r in scalar if r["test"] not in array_comparisons]
    flat = [r for r in records
            if r["kind"] == "cosine"
            and r["relative_spread"] < args.spread_threshold
            and not r["constant_would_pass"]]

    def show(title: str, rows: list[dict], note: str) -> None:
        print(f"## {title} — {len(rows)}")
        if note:
            print(f"   {note}")
        for row in sorted(rows, key=lambda r: r["test"]):
            print(f"   {row['test']}")
            print(f"      {row['kind']} shape={row['shape']} "
                  f"peak={row['peak']:.4g} mean_abs={row['mean_abs']:.4g} "
                  f"frac_zero={row['frac_zero']:.3f} "
                  f"rtol={row['rtol']} atol={row['atol']}")
        print()

    show("VACUOUS: a kernel returning all zeros would pass", vacuous,
         "the reference is within tolerance of zero everywhere")
    show("WEAK: a kernel returning one constant would pass", constant,
         "the reference is within tolerance of its own mean everywhere")
    show("UNBACKED SCALAR: the whole test collapses to one number", unbacked,
         "a 0-d reference is trivially its own mean, so the constant test says "
         "nothing here -- what matters is that no comparison in the same test "
         "covers an array")
    show("cosine over a nearly-flat reference", flat,
         "cosine ignores scale, so a flat reference is matched by any constant")

    backed = len(scalar) - len(unbacked)
    if backed:
        print(f"{backed} further scalar comparison(s) sit in tests that also "
              f"compare an array, so the array carries the check.")
        print()

    if not (vacuous or constant or unbacked or flat):
        print("No comparison would be satisfied by a zero or constant output, "
              "and no test rests on a scalar alone.")


if __name__ == "__main__":
    main()
