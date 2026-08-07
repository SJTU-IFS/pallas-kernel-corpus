"""JAXBench-style correctness, timing, and FLOP analysis for corpus kernels.

Run one implementation per process so profiler traces and JAX compilation
caches remain isolated:

  python tools/profile_kernel.py --implementation jaxbench --native
  python tools/profile_kernel.py --implementation gmm-tpu-inference --native

Timing follows JAXBench: bf16 inputs, JIT compilation, five warmups, then 50
iterations under ``jax.profiler.trace``. Device time is extracted from complete
``jit_*()`` events in the Perfetto trace. Wall-clock timing is retained only as
a diagnostic.

Each iteration is traced separately by default. The trace-viewer converter that
writes ``perfetto_trace.json.gz`` drops events past a fixed cap, and under the
corpus's instruction-level tracing flags one iteration of a dense Pallas kernel
costs a large fraction of that cap, so batching iterations into one trace
truncates it and loses the very ``jit_*`` events being measured.

Everything family-specific -- input generation, reference pairing, native
shapes, FLOP conventions -- lives in ``tools/kernel_registry.py``. This file
owns only the measurement protocol.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gzip
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kernel_registry import REGISTRY, add_arguments  # noqa: E402


ROOT = Path(__file__).parents[1]
V6E_BF16_PEAK_FLOPS = 918e12


def block_ready(value: Any) -> None:
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def stats(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "min_ms": float(data.min()),
        "median_ms": float(np.median(data)),
        "mean_ms": float(data.mean()),
        "std_ms": float(data.std()),
        "p5_ms": float(np.percentile(data, 5)),
        "p95_ms": float(np.percentile(data, 95)),
    }


#: xprof's trace-viewer JSON converter keeps at most this many duration events
#: and silently drops the rest -- `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS`
#: overrides it, and 1,000,000 is the observed built-in default.  The drop is
#: not selective: the `jit_*` XLA module events this file measures are thrown
#: out alongside the instruction-level noise, so a truncated chunk yields fewer
#: timings than iterations.  See `trace_shortfall`.
TRACE_VIEWER_MAX_EVENTS = int(
    os.environ.get("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS", 1_000_000)
)


class DeviceEvents(NamedTuple):
    """What one trace chunk yielded, plus what it cost against the cap."""

    times_ms: list[float]
    names: list[str]
    #: Events carrying a duration -- the population the converter caps.  The
    #: corpus profiles with `--xla_xprof_register_llo_debug_info=true`, which
    #: turns every VLIW bundle inside a Pallas kernel into one of these, so a
    #: single iteration of a dense kernel can cost several hundred thousand.
    duration_events: int


def extract_device_events(trace_dir: Path) -> DeviceEvents:
    traces = list(trace_dir.glob("**/perfetto_trace.json.gz"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one Perfetto trace, found {traces}")
    with gzip.open(traces[0], "rt") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload)
    with_duration = 0
    selected = []
    for event in events:
        if not isinstance(event, dict) or "dur" not in event:
            continue
        with_duration += 1
        name = str(event.get("name", ""))
        if event["dur"] > 0 and name.startswith("jit_") and "(" in name:
            selected.append(event)
    return DeviceEvents(
        [float(event["dur"]) / 1000.0 for event in selected],
        [str(event["name"]) for event in selected],
        with_duration,
    )


def trace_shortfall(
    chunk_index: int, iterations: int, events: DeviceEvents
) -> RuntimeError:
    """Explain a chunk that produced fewer device timings than iterations.

    The usual cause is trace-viewer truncation rather than anything about the
    kernel: at `TRACE_VIEWER_MAX_EVENTS` the converter drops events, and the
    per-iteration event volume of an instruction-dense Pallas kernel is high
    enough that a handful of iterations exhausts the budget.  Say which case
    this is, because the two have different fixes.
    """
    header = (
        f"trace chunk {chunk_index} yielded {len(events.times_ms)} jit device "
        f"events for {iterations} iterations "
        f"(names: {sorted(set(events.names))})"
    )
    if events.duration_events >= TRACE_VIEWER_MAX_EVENTS:
        # The retained count *is* the cap, so it says nothing about how many
        # events the run really produced -- dividing it by `iterations` would
        # recommend the chunk size that just failed.  The surviving module
        # events are the honest measure: each one is an iteration that fit.
        survived = len(events.times_ms)
        return RuntimeError(
            f"{header}.  The trace hit the trace-viewer event cap: it holds "
            f"{events.duration_events} duration events against a limit of "
            f"{TRACE_VIEWER_MAX_EVENTS}, so events past the limit -- including "
            "the jit module events measured here -- were dropped.  Under the "
            "corpus's instruction-level tracing flags this kernel spends the "
            f"whole budget on about {survived} iterations, so re-run with "
            f"--trace-chunk-size {max(1, survived // 2)} (1 is always safe), or "
            "raise TF_PROFILER_TRACE_VIEWER_MAX_EVENTS -- which costs memory, "
            "since the trace is loaded whole to be rewritten."
        )
    return RuntimeError(
        f"{header}, and the trace was not truncated "
        f"({events.duration_events} duration events, cap "
        f"{TRACE_VIEWER_MAX_EVENTS}), so the executions themselves are missing "
        "from the trace rather than dropped from it."
    )


def correctness_status(cosine: float) -> str:
    if cosine > 0.9999:
        return "PASS"
    if cosine > 0.99:
        return "MARGINAL"
    return "FAIL"


def compare(actual: Any, expected: Any) -> dict[str, Any]:
    """Compare two outputs, which may be arrays or tuples of arrays.

    Integer leaves -- expert ids, page indices, anything used as an index --
    are compared for **exact** equality, because a nearly-right index is a
    wrong index and a cosine similarity over indices is meaningless.  Float
    leaves get the usual cosine/max/mean treatment, and the headline cosine is
    the worst across them.
    """
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    if len(actual_leaves) != len(expected_leaves):
        raise ValueError(
            f"output structures differ: {len(actual_leaves)} leaves vs "
            f"{len(expected_leaves)}"
        )

    per_leaf: list[dict[str, Any]] = []
    cosines: list[float] = []
    exact_failures = 0
    max_abs = 0.0
    mean_abs = 0.0
    for index, (got, want) in enumerate(zip(actual_leaves, expected_leaves)):
        if jnp.issubdtype(jnp.asarray(want).dtype, jnp.integer):
            equal = bool(jnp.all(got == want))
            exact_failures += int(not equal)
            per_leaf.append(
                {"leaf": index, "dtype": str(want.dtype), "exact_match": equal}
            )
            continue
        got_f32 = jnp.asarray(got).astype(jnp.float32)
        want_f32 = jnp.asarray(want).astype(jnp.float32)
        difference = jnp.abs(got_f32 - want_f32)
        cosine = float(
            jnp.vdot(got_f32, want_f32)
            / (jnp.linalg.norm(got_f32) * jnp.linalg.norm(want_f32) + 1e-12)
        )
        leaf_max = float(jnp.max(difference))
        leaf_mean = float(jnp.mean(difference))
        cosines.append(cosine)
        max_abs = max(max_abs, leaf_max)
        mean_abs = max(mean_abs, leaf_mean)
        per_leaf.append(
            {
                "leaf": index,
                "dtype": str(want.dtype),
                "cosine_similarity": cosine,
                "max_abs_diff": leaf_max,
                "mean_abs_diff": leaf_mean,
            }
        )

    cosine = min(cosines) if cosines else 1.0
    status = correctness_status(cosine)
    if exact_failures:
        status = "FAIL"
    return {
        "cosine_similarity": cosine,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "status": status,
        "exact_match_failures": exact_failures,
        "per_leaf": per_leaf if len(per_leaf) > 1 else None,
    }


def cost_analysis(
    function: Callable[..., Any], inputs: tuple[Any, ...]
) -> dict[str, float]:
    try:
        analysis = jax.jit(function).lower(*inputs).compile().cost_analysis()
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        return {
            str(key): float(value)
            for key, value in analysis.items()
            if isinstance(value, (int, float))
        }
    except Exception as error:  # Keep profiling useful if an estimate is unavailable.
        return {"error": str(error)}


#: HLO ops that XLA itself rewrites into `tpu_custom_call` on TPU, with no
#: Pallas kernel involved.  `jnp.einsum` + `jax.nn.softmax` written in plain JAX
#: lowers to `%online-softmax`, so "contains a tpu_custom_call" does **not**
#: mean "is a Pallas kernel" -- a pure-JAX attention reference trips it.
XLA_AUTOMATIC_CUSTOM_CALLS = ("online-softmax",)


def count_pallas_launches(
    function: Callable[..., Any], inputs: tuple[Any, ...]
) -> int:
    """Count Pallas launches in the lowered HLO.

    Pallas lowers to a `tpu_custom_call` on TPU, so this distinguishes a real
    kernel launch from a fallback that quietly ran plain XLA instead.  Returns
    -1 if the module cannot be compiled for inspection, which is not treated as
    a failure -- the profiling run that follows will surface any real problem.

    XLA's own automatic rewrites use the same custom-call target, so they are
    excluded by op name: counting them would make the pure-JAX flash-attention
    reference look like a Pallas kernel and get it rejected as a baseline.
    """
    try:
        hlo = jax.jit(function).lower(*inputs).compile().as_text()
    except Exception:
        return -1
    return sum(
        1
        for line in hlo.splitlines()
        if "tpu_custom_call" in line
        and not any(name in line for name in XLA_AUTOMATIC_CUSTOM_CALLS)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", required=True, choices=sorted(REGISTRY)
    )
    parser.add_argument(
        "--allow-xla-fallback",
        action="store_true",
        help=(
            "profile an implementation even though it lowered to no Pallas "
            "launch.  Use only to measure a fallback path deliberately; the "
            "result records pallas_launches=0 either way."
        ),
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="use the source benchmark's original configuration",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--trace-chunk-size",
        type=int,
        default=1,
        help=(
            "iterations per profiler trace.  One is the default because the "
            "trace-viewer converter drops events past "
            f"{TRACE_VIEWER_MAX_EVENTS}, and an instruction-dense Pallas kernel "
            "can spend that budget in a couple of iterations; raising it is "
            "faster but silently unsafe for kernels that have never been "
            "measured (see trace_shortfall)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles")
    parser.add_argument(
        "--run-label",
        default="",
        help="suffix the run directory, to keep variants of one implementation "
        "(e.g. balanced vs unbalanced routing) side by side",
    )
    add_arguments(parser)
    args = parser.parse_args()

    case = REGISTRY[args.implementation]
    if args.native:
        if case.native_shape is None:
            raise SystemExit(f"{args.implementation} has no native shape registered")
        case.native_shape(args)

    built = case.build(args)
    if built.contract != case.contract:
        raise RuntimeError(
            f"{args.implementation}: registry declares {case.contract!r} but "
            f"built {built.contract!r}"
        )
    compiled = jax.jit(built.function)

    # Kernels that donate an argument consume it, so every call needs its own
    # copy.  Pre-build them here, outside the timed region.
    donated = built.donated_argnum
    total_calls = args.warmup + 2 * args.iterations + 4
    if donated is None:
        def call_inputs(_index: int) -> tuple[Any, ...]:
            return built.inputs
    else:
        spares = [
            jnp.array(built.inputs[donated], copy=True) for _ in range(total_calls)
        ]
        block_ready(spares)

        def call_inputs(index: int) -> tuple[Any, ...]:
            arguments = list(built.inputs)
            arguments[donated] = spares[index]
            return tuple(arguments)

    counter = iter(range(total_calls))
    for _ in range(args.warmup):
        output = compiled(*call_inputs(next(counter)))
        block_ready(output)

    # ------------------------------------------------------------------
    # Traced measurement.  This runs FIRST, with nothing between process
    # start and here but the build, the spare copies and the warmups.
    #
    # It is not stylistic.  Under the corpus's instruction-level tracing
    # flags, a process that has already run the correctness check, the cost
    # analysis and the wall-clock loop can open exactly one profiler
    # session: the second `jax.profiler.trace` blocks for sixty seconds and
    # then returns a trace with no device data in it at all (~1,600 host
    # events, a 300 KB XPlane against the usual 18 MB), and every session
    # after that returns empty immediately.  Measured on v6e: fourteen
    # consecutive traced chunks all capture when the trace loop comes first,
    # and chunk 1 onwards capture nothing when it comes last, with the same
    # kernel and the same flags.  No single step is responsible -- each one
    # alone is harmless -- so the ordering, not a smaller prelude, is the
    # fix.  Everything the run needs that is not the measurement therefore
    # happens after it.  See profiles/native/mla_attention/report.md #7b.
    # ------------------------------------------------------------------
    run_name = args.implementation.replace("-", "_")
    if args.run_label:
        run_name = f"{run_name}__{args.run_label}"
    run_dir = args.output_dir / case.family / run_name
    trace_dir = run_dir / "trace"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    device_times: list[float] = []
    event_names: list[str] = []
    chunk_event_counts: list[int] = []
    trace_directories = []
    remaining = args.iterations
    chunk_index = 0
    while remaining:
        chunk_iterations = min(args.trace_chunk_size, remaining)
        chunk_dir = trace_dir / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True)
        with jax.profiler.trace(
            str(chunk_dir),
            create_perfetto_link=False,
            create_perfetto_trace=True,
        ):
            for _ in range(chunk_iterations):
                arguments = call_inputs(next(counter))
                with jax.named_scope("bench_kernel"):
                    output = compiled(*arguments)
                block_ready(output)
        chunk = extract_device_events(chunk_dir)
        if len(chunk.times_ms) < chunk_iterations:
            raise trace_shortfall(chunk_index, chunk_iterations, chunk)
        # Match JAXBench: take the complete jit wrapper event per iteration.
        device_times.extend(chunk.times_ms[:chunk_iterations])
        event_names.extend(chunk.names)
        chunk_event_counts.append(chunk.duration_events)
        trace_directories.append(str(chunk_dir.resolve()))
        remaining -= chunk_iterations
        chunk_index += 1
    device_median_s = statistics.median(device_times) / 1000.0

    # ------------------------------------------------------------------
    # Everything below is analysis, and runs only after the trace loop is
    # finished, for the reason given above.  A run that fails a check here
    # has already paid for its profiling, which is the price of the
    # ordering; nothing is written to disk until every check has passed.
    # ------------------------------------------------------------------
    wall_times = []
    for _ in range(args.iterations):
        arguments = call_inputs(next(counter))
        start = time.perf_counter()
        output = compiled(*arguments)
        block_ready(output)
        wall_times.append((time.perf_counter() - start) * 1e3)

    # Several upstream kernels fall back to plain XLA at runtime -- when there
    # is no SparseCore, when an input fits in VMEM, when a shape is unsupported
    # -- and do it silently.  Profiling one of those would report XLA's time
    # under a Pallas label.  The lowered HLO settles it: a Pallas launch emits a
    # `tpu_custom_call`.  Recorded for every run, and checked here.
    pallas_launches = count_pallas_launches(built.function, built.inputs)
    if case.is_reference:
        if pallas_launches:
            raise RuntimeError(
                f"{args.implementation}: the reference baseline lowered to "
                f"{pallas_launches} Pallas tpu_custom_call(s); a Pallas "
                "implementation is not a valid JAX baseline.  (XLA's own "
                f"rewrites -- {', '.join(XLA_AUTOMATIC_CUSTOM_CALLS)} -- are "
                "already excluded from this count.)"
            )
    elif pallas_launches == 0 and not args.allow_xla_fallback:
        raise RuntimeError(
            f"{args.implementation}: lowered to 0 tpu_custom_call(s) -- the "
            "kernel took a non-Pallas fallback at this shape, so the run "
            "measured XLA, not the kernel.  Change the shape, or profile it "
            "deliberately with --allow-xla-fallback."
        )

    # A reference that validates traced values cannot be jitted; it is still a
    # valid correctness check, just not a timing denominator.
    compiled_reference = (
        jax.jit(built.reference) if built.jit_reference else built.reference
    )
    expected = compiled_reference(*call_inputs(next(counter)))
    actual = compiled(*call_inputs(next(counter)))
    block_ready((expected, actual))
    correctness = compare(actual, expected)
    correctness["reference"] = getattr(
        built.reference, "__name__", str(built.reference)
    )

    reference_cost = (
        cost_analysis(built.reference, built.inputs)
        if built.jit_reference
        else {"error": "reference is not jittable; correctness-only"}
    )
    implementation_cost = cost_analysis(built.function, built.inputs)

    achieved = {
        convention: {
            "flops": count,
            "achieved_tflops": count / device_median_s / 1e12,
            "mxu_utilization_pct": count
            / device_median_s
            / V6E_BF16_PEAK_FLOPS
            * 100,
        }
        for convention, count in built.flops.items()
    }
    # A pure data-movement kernel has no FLOPs, so MXU utilization would be a
    # meaningless zero.  Report achieved bandwidth instead and leave the
    # compute fields null rather than filling them with a fake number.
    if built.primary_flops is None:
        primary_flops = None
        achieved_tflops = None
        utilization = None
    else:
        primary_flops = built.flops[built.primary_flops]
        achieved_tflops = achieved[built.primary_flops]["achieved_tflops"]
        utilization = achieved[built.primary_flops]["mxu_utilization_pct"]
    achieved_bytes = {
        convention: {
            "bytes": count,
            "achieved_gbytes_per_second": count / device_median_s / 1e9,
        }
        for convention, count in (built.bytes_moved or {}).items()
    }
    bandwidth = (
        achieved_bytes[built.primary_bytes]["achieved_gbytes_per_second"]
        if built.primary_bytes
        else None
    )

    result = {
        "implementation": args.implementation,
        "run_label": args.run_label,
        "family": case.family,
        "contract": built.contract,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "dtype": "bfloat16",
        "shape": built.shape,
        "warmup_iterations": args.warmup,
        "profiled_iterations": args.iterations,
        "correctness": correctness,
        "pallas_launches": pallas_launches,
        "measures_xla_fallback": bool(
            pallas_launches == 0 and not case.is_reference
        ),
        "reference_is_jittable": built.jit_reference,
        "donated_argnum": built.donated_argnum,
        "flops": achieved,
        "primary_flops_convention": built.primary_flops,
        "logical_flops": primary_flops,
        "flop_accounting_notes": list(built.notes),
        "reference_cost_analysis": reference_cost,
        "implementation_cost_analysis": implementation_cost,
        "device_timing": stats(device_times),
        "wall_timing": stats(wall_times),
        "trace_event_names": sorted(set(event_names)),
        "native_source_shape": args.native,
        "kernel_config": built.config,
        "bytes": achieved_bytes,
        "primary_bytes_convention": built.primary_bytes,
        "bytes_moved": (
            built.bytes_moved[built.primary_bytes] if built.primary_bytes else None
        ),
        "achieved_gbytes_per_second": bandwidth,
        "achieved_tflops": achieved_tflops,
        "mxu_utilization_pct": utilization,
        "v6e_peak_tflops_bf16": V6E_BF16_PEAK_FLOPS / 1e12,
        "trace_directory": str(trace_dir.resolve()),
        "trace_chunks": trace_directories,
        "trace_chunk_size": args.trace_chunk_size,
        # How close each chunk came to the converter's drop threshold.  A run
        # whose chunks sit near the cap measured a trace that was nearly
        # truncated, and its timings are worth re-taking at a smaller chunk.
        "trace_viewer_max_events": TRACE_VIEWER_MAX_EVENTS,
        "trace_duration_events_per_chunk": chunk_event_counts,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
