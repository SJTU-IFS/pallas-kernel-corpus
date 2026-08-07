"""Measure what a SparseCore gather costs when a TensorCore matmul is running.

`tools/profile_kernel.py` times one kernel with the rest of the chip idle. For
the SparseCore ragged gathers that is the one regime where they cannot win:
XLA's `x[indices]` already runs at roughly HBM bandwidth, so a standalone
comparison can only show the SparseCore kernel losing. Their actual claim is
different -- a SparseCore gather runs on hardware the TensorCore is not using,
so in an MoE layer it should disappear behind the matmuls.

This harness measures that claim directly. It times five configurations and
takes **differences** between them:

    matmul                      T_mm
    gather-sparsecore           T_sc      (Pallas, plsc.VectorSubcoreMesh)
    gather-xla                  T_xla     (x[indices])
    matmul+gather-sparsecore    T_mm_sc
    matmul+gather-xla           T_mm_xla

and derives, for each gather:

    marginal cost  = T_both - T_mm       what the gather actually adds
    hidden         = 1 - marginal / T_alone
                     1.0 -> free, fully overlapped
                     0.0 -> fully serialized, as expensive as running it alone
                     < 0 -> worse than serial: it is contending for something

The comparison that matters is `hidden` for SparseCore **against** `hidden` for
XLA at the same matmul size. Both gathers move the same bytes through HBM; the
question is only whether issuing them from the SparseCore keeps the TensorCore
fed. A SparseCore gather is not free just because it runs elsewhere -- if the
matmul is HBM-bound there is no spare bandwidth to hide it in. That is why the
matmul size is swept: a square NxNxN bf16 matmul has arithmetic intensity N/3,
so N=1024 sits below the v6e's ridge point (577.96 FLOP/byte) and N>=2048 sits
above it. If the SparseCore story is right, `hidden` should rise with N.

The two configurations in a pair are compiled as one jitted function with **no
data dependency** between them, returning both outputs so neither is dead code.
Whether XLA then overlaps them is exactly what is being measured -- it is not
assumed anywhere, and a flat `hidden` near zero would be a real result, not a
harness bug.

Protocol matches `profile_kernel.py`: 5 warmups, 50 wall-clock diagnostics, 50
device-profiled iterations, device time from complete Perfetto `jit_*()` events.
Run **one configuration per process** so compilation caches and profiler state
stay isolated:

    python tools/overlap_harness.py --config matmul --matmul-n 4096
    python tools/overlap_harness.py --config matmul+gather-sparsecore --matmul-n 4096

Then summarize, which needs no TPU:

    python tools/overlap_harness.py --summarize --output-dir profiles/native
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
from profile_kernel import (  # noqa: E402
    V6E_BF16_PEAK_FLOPS,
    block_ready,
    count_pallas_launches,
    extract_device_events,
    stats,
    trace_shortfall,
)

ROOT = Path(__file__).parents[1]
SPARSECORE = ROOT / "kernels" / "memory" / "sparsecore_ragged_gather"
GROUPED_MATMUL = ROOT / "kernels" / "moe" / "grouped_matmul"
FAMILY = "sparsecore_ragged_gather"
V6E_HBM_BYTES_PER_SECOND = 1640e9

# Two TensorCore workloads, because which one you hide behind changes the
# answer.  A square dense matmul has arithmetic intensity N/3, so at N>=2048 it
# is compute-bound and leaves HBM bandwidth spare for a concurrent gather.  The
# grouped matmul an MoE layer actually runs is **memory**-bound at the corpus's
# native shape (~208 FLOP/byte against a 578 ridge point), so there is far less
# spare bandwidth -- and a gather is pure bandwidth.  The dense number alone
# would flatter these kernels.
CONFIGS = (
    "matmul",
    "gmm",
    "gather-sparsecore",
    "gather-xla",
    "matmul+gather-sparsecore",
    "matmul+gather-xla",
    "gmm+gather-sparsecore",
    "gmm+gather-xla",
)


def load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_tracks(trace_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Group a Perfetto trace's timed events by XPlane track name.

    The TPU trace keeps the TensorCore and the SparseCores on separate tracks,
    which is what makes overlap measurable rather than inferable:

        "XLA Ops"          TensorCore HLO ops
        "Sparse Core Ops"  the SparseCore program (async call-start/call-done)
        "TEC 0".."TEC 15"  the 16 subcores of each SparseCore
        "XLA Modules"      the enclosing jit_* module

    Summing durations across every track triple-counts the same work -- the
    same op appears under Modules, Ops and per-subcore views -- so a metric
    built that way reports concurrency even for a lone matmul.  Track names are
    resolved from the trace's own metadata events rather than guessed.
    """
    import gzip

    traces = list(trace_dir.glob("**/perfetto_trace.json.gz"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one Perfetto trace, found {traces}")
    with gzip.open(traces[0], "rt") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload)

    thread_names: dict[tuple[Any, Any], str] = {}
    for event in events:
        if isinstance(event, dict) and event.get("ph") == "M":
            if event.get("name") == "thread_name":
                label = (event.get("args") or {}).get("name")
                if label:
                    thread_names[(event.get("pid"), event.get("tid"))] = label

    tracks: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event, dict) or not event.get("dur"):
            continue
        if not isinstance(event.get("ts"), (int, float)):
            continue
        label = thread_names.get((event.get("pid"), event.get("tid")))
        if label is None:
            continue
        tracks.setdefault(label, []).append(
            {
                "name": str(event.get("name", "")),
                "ts": float(event["ts"]),
                "dur": float(event["dur"]),
                "pid": event.get("pid"),
            }
        )
    return tracks


def union_intervals(
    events: list[dict[str, Any]]
) -> list[tuple[float, float]]:
    """Merge event intervals so a busy period is counted once, not per event."""
    if not events:
        return []
    intervals = sorted(
        (event["ts"], event["ts"] + event["dur"]) for event in events
    )
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def total_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def intersect(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def is_offload_stub(name: str) -> bool:
    """Is this TensorCore-track op just bookkeeping for an offloaded kernel?

    XLA lowers a SparseCore launch as an async pair and leaves `call-start`,
    `call-done` and `prepare_*` ops on the TensorCore track. The `call-done` in
    particular spans the entire SparseCore execution, because it *is* the
    TensorCore waiting -- counting it as TensorCore work makes even a
    standalone gather look 100% overlapped.

    The rule is structural rather than name-based on purpose. Matching by
    kernel-name root would drop real TensorCore ops: the grouped matmul's
    SparseCore offload is named `copy.23.cloned.1.call-done`, whose root
    `copy` also matches the ordinary TensorCore op `copy.3`.

    (`.call-start`/`.call-done` is the async *custom-call* pattern. Async
    collectives use a similar suffix, but these are single-device runs with no
    collectives.)
    """
    return (
        name.startswith("prepare_start_")
        or name.startswith("prepare_done_")
        or name.endswith(".call-start")
        or name.endswith(".call-done")
    )


def measure_overlap(trace_dir: Path, kernel_hint: str | None) -> dict[str, Any]:
    """How much of the gather's SparseCore work ran while the TensorCore computed.

    ``sparsecore_overlapped_fraction`` is the number that matters: 1.0 means
    every microsecond the gather spent on the SparseCore was hidden underneath
    TensorCore work; 0.0 means the two units took strict turns.

    ``kernel_hint`` separates *our* SparseCore work from the workload's own.
    That distinction is not hypothetical: XLA offloads scatters inside the
    grouped matmul to the SparseCore (`scatter_offload_custom_fusion`), so a
    grouped matmul occupies the SparseCore for ~10 us before any gather is
    added. Attributing that to the gather would understate its cost and
    overstate how idle the SparseCore is. The harness passes the hint because
    it knows which kernel it launched; nothing is inferred from trace layout.

    Both windows are reported raw so the shared-clock assumption between the
    TensorCore plane and the SparseCore planes stays checkable rather than
    taken on faith.
    """
    tracks = read_tracks(trace_dir)
    all_sparsecore = tracks.get("Sparse Core Ops", [])
    if kernel_hint is None:
        measured_events, workload_events = [], all_sparsecore
    else:
        measured_events = [
            event for event in all_sparsecore if kernel_hint in event["name"]
        ]
        workload_events = [
            event for event in all_sparsecore if kernel_hint not in event["name"]
        ]

    measured = union_intervals(measured_events)
    tensorcore_events = [
        event
        for event in tracks.get("XLA Ops", [])
        if not is_offload_stub(event["name"])
    ]
    tensorcore = union_intervals(tensorcore_events)

    detail: dict[str, Any] = {
        "tensorcore_busy_us": total_length(tensorcore),
        "tensorcore_ops_counted": len(tensorcore_events),
        "tensorcore_ops_excluded_as_offload_stubs": len(
            tracks.get("XLA Ops", [])
        )
        - len(tensorcore_events),
        # SparseCore time the TensorCore workload spends on its own account,
        # before any gather is added.
        "workload_sparsecore_busy_us": total_length(
            union_intervals(workload_events)
        ),
        "workload_sparsecore_op_names": sorted(
            {event["name"].split(".")[0] for event in workload_events}
        )[:6],
    }
    if not measured:
        detail.update(
            {
                "sparsecore_busy_us": 0.0,
                "overlap_us": 0.0,
                "sparsecore_overlapped_fraction": None,
                "reason": (
                    "no SparseCore ops matching the measured kernel"
                    if kernel_hint
                    else "no SparseCore kernel in this configuration"
                ),
            }
        )
        return detail

    both = intersect(tensorcore, measured)
    detail.update(
        {
            "sparsecore_busy_us": total_length(measured),
            "overlap_us": total_length(both),
            "sparsecore_overlapped_fraction": total_length(both)
            / total_length(measured),
            "tensorcore_window": [tensorcore[0][0], tensorcore[-1][1]]
            if tensorcore
            else None,
            "sparsecore_window": [measured[0][0], measured[-1][1]],
            "sparsecore_op_names": [event["name"] for event in measured_events][:6],
        }
    )
    return detail


def workload_bytes(spec: dict[str, Any]) -> int | None:
    """Bytes the TensorCore workload itself moves through HBM, bf16 operands.

    For a grouped matmul the **expert weights dominate** -- at 128 experts they
    are 81% of the traffic -- so arithmetic intensity here is essentially tokens
    per expert, which is the knob an MoE design actually turns.
    """
    shape = spec["shape"]
    if shape.get("gmm"):
        gmm = shape["gmm"]
        return 2 * (
            gmm["rows"] * gmm["k"]
            + gmm["groups"] * gmm["k"] * gmm["n"]
            + gmm["rows"] * gmm["n"]
        )
    if shape.get("matmul_n"):
        return 3 * shape["matmul_n"] ** 2 * 2
    return None


def workload_arithmetic_intensity(spec: dict[str, Any]) -> float | None:
    """FLOP per byte of the TensorCore workload, against a 578 ridge point.

    Intended as a proxy for whether there is spare HBM bandwidth for a
    concurrent gather to use.  It is only a proxy: what actually matters is the
    achieved bandwidth, which `summarize` reports alongside it -- a kernel can
    sit at high arithmetic intensity and still not saturate the bus.
    """
    moved = workload_bytes(spec)
    if not spec["matmul_flops"] or not moved:
        return None
    return spec["matmul_flops"] / moved


def build(args) -> dict[str, Any]:
    """Build the callable for one configuration, plus its cost accounting."""
    baseline = load_module("scg_baseline", SPARSECORE / "baseline.py")
    workload = args.config.split("+")[0]
    wants_matmul = workload == "matmul"
    wants_gmm = workload == "gmm"
    wants_gather = "gather" in args.config

    inputs: list[Any] = []
    matmul_flops = 0
    gather_bytes = 0
    gmm_shape: dict[str, int] | None = None

    if wants_gmm:
        gmm_baseline = load_module(
            "gmm_baseline", GROUPED_MATMUL / "baseline.py"
        )
        gmm_module = load_module(
            "gmm_jaxbench", GROUPED_MATMUL / "jaxbench_optimized.py"
        )
        lhs, rhs, group_sizes = gmm_baseline.create_inputs(
            rows=args.gmm_rows,
            num_groups=args.gmm_groups,
            k=args.gmm_k,
            n=args.gmm_n,
            dtype=jnp.bfloat16,
            balanced=True,
        )
        # Same balance check the grouped_matmul family makes: inside jit
        # group_sizes is a tracer and nothing can verify itself.
        gmm_baseline.check_uniform_groups(
            group_sizes, rows=args.gmm_rows, num_groups=args.gmm_groups
        )
        inputs += [lhs, rhs, group_sizes]
        # `workload` is JAXBench's upstream-tuned tiling, matching the shape
        # profiled in profiles/native/grouped_matmul/.
        grouped_matmul = gmm_module.workload
        matmul_flops = 2 * args.gmm_rows * args.gmm_k * args.gmm_n
        gmm_shape = {
            "rows": args.gmm_rows,
            "groups": args.gmm_groups,
            "k": args.gmm_k,
            "n": args.gmm_n,
        }

    if wants_matmul:
        size = args.matmul_n
        keys = jax.random.split(jax.random.key(0), 2)
        left = jax.random.normal(keys[0], (size, size), jnp.bfloat16)
        right = jax.random.normal(keys[1], (size, size), jnp.bfloat16)
        inputs += [left, right]
        matmul_flops = 2 * size**3

    gather_dtype = jnp.dtype(args.gather_dtype)
    if wants_gather:
        built = baseline.create_inputs(
            num_rows=args.scg_rows,
            hidden=args.scg_hidden,
            out_rows=args.scg_out_rows,
            dtype=gather_dtype,
        )
        inputs += [built["x"], built["indices"], built["start"], built["end"]]
        # A gather is pure bandwidth, so halving the element size halves the
        # work.  Whether that changes the *answer* depends on whether the
        # TensorCore workload left any bandwidth spare -- which is the point of
        # sweeping this.
        gather_bytes = baseline.bytes_moved(
            out_rows=args.scg_out_rows,
            hidden=args.scg_hidden,
            itemsize=gather_dtype.itemsize,
        )
        if args.config.endswith("sparsecore"):
            module = load_module("scg_tokamax", SPARSECORE / "tokamax_optimized.py")
            gather = module.ragged_gather_pallas
        else:
            gather = baseline.ragged_gather

    # No data dependency between the two halves: whether they overlap is the
    # measurement, not an assumption.  Both outputs are returned so neither can
    # be eliminated as dead code.
    if wants_matmul and wants_gather:
        def function(left, right, x, indices, start, end):
            return jnp.matmul(left, right), gather(x, indices, start, end)
    elif wants_gmm and wants_gather:
        def function(lhs, rhs, group_sizes, x, indices, start, end):
            return (
                grouped_matmul(lhs, rhs, group_sizes),
                gather(x, indices, start, end),
            )
    elif wants_matmul:
        def function(left, right):
            return jnp.matmul(left, right)
    elif wants_gmm:
        def function(lhs, rhs, group_sizes):
            return grouped_matmul(lhs, rhs, group_sizes)
    else:
        def function(x, indices, start, end):
            return gather(x, indices, start, end)

    return {
        "function": function,
        "inputs": tuple(inputs),
        "matmul_flops": matmul_flops,
        "gather_bytes": gather_bytes,
        "shape": {
            "workload": workload if (wants_matmul or wants_gmm) else None,
            "matmul_n": args.matmul_n if wants_matmul else None,
            "gmm": gmm_shape,
            "gather": (
                {
                    "num_rows": args.scg_rows,
                    "hidden": args.scg_hidden,
                    "out_rows": args.scg_out_rows,
                    "dtype": gather_dtype.name,
                }
                if wants_gather
                else None
            ),
        },
    }


def run_one(args) -> dict[str, Any]:
    spec = build(args)
    function, inputs = spec["function"], spec["inputs"]
    compiled = jax.jit(function)

    # Exactly how many Pallas launches this configuration should contain: the
    # grouped matmul is itself a Mosaic kernel, the dense matmul is not, and the
    # SparseCore gather adds one on top.  Checking the exact count catches a
    # silent fallback in either direction -- a SparseCore gather that quietly
    # ran XLA, or an "XLA" arm that is not actually XLA.
    launches = count_pallas_launches(function, inputs)
    expected = int(args.config.split("+")[0] == "gmm") + int(
        args.config.endswith("sparsecore")
    )
    if launches != expected:
        raise RuntimeError(
            f"{args.config}: expected {expected} tpu_custom_call(s), found "
            f"{launches}.  A SparseCore gather that lowers to 0 took its XLA "
            "fallback and would not measure overlap at all."
        )

    # Which SparseCore ops in the trace are ours, as opposed to ones the
    # TensorCore workload offloads on its own account.
    kernel_hint = (
        "ragged_gather" if args.config.endswith("gather-sparsecore") else None
    )

    for _ in range(args.warmup):
        block_ready(compiled(*inputs))

    wall_times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        block_ready(compiled(*inputs))
        wall_times.append((time.perf_counter() - start) * 1e3)

    run_name = args.config.replace("+", "_plus_").replace("-", "_")
    if args.config.startswith("matmul"):
        run_name = f"{run_name}__n{args.matmul_n}"
    elif args.config.startswith("gmm"):
        run_name = (
            f"{run_name}__m{args.gmm_rows}x{args.gmm_groups}"
            f"x{args.gmm_k}x{args.gmm_n}"
        )
    if "gather" in args.config:
        run_name = f"{run_name}__g{args.scg_out_rows}"
        if args.gather_dtype != "float32":
            run_name = f"{run_name}__{args.gather_dtype}"
    run_dir = args.output_dir / FAMILY / "overlap" / run_name
    trace_dir = run_dir / "trace"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    device_times: list[float] = []
    overlap_samples: list[float] = []
    overlap_detail: dict[str, Any] = {}
    remaining = args.iterations
    chunk_index = 0
    while remaining:
        chunk_iterations = min(args.trace_chunk_size, remaining)
        chunk_dir = trace_dir / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True)
        with jax.profiler.trace(
            str(chunk_dir), create_perfetto_link=False, create_perfetto_trace=True
        ):
            for _ in range(chunk_iterations):
                with jax.named_scope("bench_overlap"):
                    output = compiled(*inputs)
                block_ready(output)
        chunk = extract_device_events(chunk_dir)
        if len(chunk.times_ms) < chunk_iterations:
            raise trace_shortfall(chunk_index, chunk_iterations, chunk)
        device_times.extend(chunk.times_ms[:chunk_iterations])
        measured = measure_overlap(chunk_dir, kernel_hint)
        if measured.get("sparsecore_overlapped_fraction") is not None:
            overlap_samples.append(measured["sparsecore_overlapped_fraction"])
        if not overlap_detail:
            overlap_detail = measured
        remaining -= chunk_iterations
        chunk_index += 1

    if not args.keep_traces:
        # Traces for a 20-run sweep are tens of GB; the derived numbers are the
        # deliverable.  --keep-traces retains them for inspecting one run.
        shutil.rmtree(trace_dir)

    device = stats(device_times)
    result = {
        "config": args.config,
        "family": FAMILY,
        "jax_version": jax.__version__,
        "devices": [str(device_) for device_ in jax.devices()],
        "shape": spec["shape"],
        "matmul_dtype": "bfloat16",
        "gather_dtype": (
            spec["shape"]["gather"]["dtype"] if spec["shape"]["gather"] else None
        ),
        "warmup_iterations": args.warmup,
        "profiled_iterations": args.iterations,
        "pallas_launches": launches,
        "device_timing": device,
        "wall_timing": stats(wall_times),
        "matmul_flops": spec["matmul_flops"],
        "gather_bytes_moved": spec["gather_bytes"],
        "matmul_tflops": (
            spec["matmul_flops"] / (device["median_ms"] * 1e-3) / 1e12
            if spec["matmul_flops"]
            else None
        ),
        "workload_arithmetic_intensity": workload_arithmetic_intensity(spec),
        "v6e_ridge_point_flop_per_byte": V6E_BF16_PEAK_FLOPS
        / V6E_HBM_BYTES_PER_SECOND,
        "sparsecore_overlapped_fraction": (
            statistics.median(overlap_samples) if overlap_samples else None
        ),
        "trace_overlap_detail": overlap_detail,
        "traces_kept": bool(args.keep_traces),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def workload_label(payload: dict[str, Any]) -> str | None:
    """Name the TensorCore workload a run was paired with, or None if none."""
    base = payload["config"].split("+")[0]
    shape = payload["shape"]
    if base == "matmul":
        return f"dense-{shape['matmul_n']}"
    if base == "gmm":
        gmm = shape["gmm"]
        return f"gmm-{gmm['rows']}x{gmm['groups']}x{gmm['k']}x{gmm['n']}"
    return None


def summarize(args) -> None:
    """Derive marginal cost and hidden fraction. Needs no TPU."""
    root = args.output_dir / FAMILY / "overlap"
    results: dict[tuple[str, Any, Any], dict[str, Any]] = {}
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text())
        gather = payload["shape"]["gather"]
        results[
            (
                payload["config"],
                workload_label(payload),
                gather["out_rows"] if gather else None,
                # Runs recorded before dtype was an axis were all float32.
                gather.get("dtype", "float32") if gather else None,
            )
        ] = payload

    def entry(config: str, label, out_rows, dtype) -> dict[str, Any] | None:
        return results.get((config, label, out_rows, dtype))

    def median(config: str, label, out_rows, dtype) -> float | None:
        found = entry(config, label, out_rows, dtype)
        return found["device_timing"]["median_ms"] if found else None

    # Three axes: which TensorCore workload, how big it is, and how big the
    # gather is.  A combined run only means something next to the same
    # workload run alone and the same gather run alone.
    combinations = sorted(
        {
            (
                payload["config"].split("+")[0],
                workload_label(payload),
                payload["shape"]["gather"].get("dtype", "float32"),
                payload["shape"]["gather"]["out_rows"],
            )
            for payload in results.values()
            if "+" in payload["config"]
        }
    )

    rows = []
    for base, label, dtype, out_rows in combinations:
        alone = {
            "sparsecore": median("gather-sparsecore", None, out_rows, dtype),
            "xla": median("gather-xla", None, out_rows, dtype),
        }
        # The TensorCore workload alone does not involve the gather at all.
        workload_entry = entry(base, label, None, None)
        if workload_entry is None:
            continue
        workload_ms = workload_entry["device_timing"]["median_ms"]
        workload_busy = workload_entry["trace_overlap_detail"].get(
            "tensorcore_busy_us"
        )
        row: dict[str, Any] = {
            "workload": base,
            "workload_label": label,
            "gather_dtype": dtype,
            "gather_out_rows": out_rows,
            # Recomputed from the shape rather than read back, so runs
            # recorded before this field existed are treated the same.
            "arithmetic_intensity": workload_arithmetic_intensity(
                {
                    "matmul_flops": workload_entry["matmul_flops"],
                    "shape": workload_entry["shape"],
                }
            ),
            "workload_ms": workload_ms,
            "workload_tensorcore_busy_us": workload_busy,
            "workload_tflops": workload_entry["matmul_tflops"],
            "workload_own_sparsecore_us": workload_entry[
                "trace_overlap_detail"
            ].get("workload_sparsecore_busy_us"),
            # The variable the headroom explanation actually rests on: how much
            # of HBM the workload is already using before a gather is added.
            "workload_hbm_pct": (
                100.0
                * (moved / (workload_ms * 1e-3))
                / V6E_HBM_BYTES_PER_SECOND
                if (moved := workload_bytes(workload_entry)) and workload_ms
                else None
            ),
            "gather_alone_ms": dict(alone),
        }
        for tag, suffix in (("sparsecore", "gather-sparsecore"), ("xla", "gather-xla")):
            both = median(f"{base}+{suffix}", label, out_rows, dtype)
            if both is None or alone[tag] is None:
                continue
            marginal = both - workload_ms
            row[f"{tag}_both_ms"] = both
            row[f"{tag}_marginal_ms"] = marginal
            row[f"{tag}_alone_ms"] = alone[tag]
            row[f"{tag}_hidden"] = 1.0 - marginal / alone[tag]
            combined = entry(f"{base}+{suffix}", label, out_rows, dtype)
            row[f"{tag}_trace_overlap"] = combined["sparsecore_overlapped_fraction"]
            busy = combined["trace_overlap_detail"].get("tensorcore_busy_us")
            row[f"{tag}_tensorcore_busy_us"] = busy
            # Decompose the marginal cost: part of it is the TensorCore itself
            # running slower because the gather competes for memory, the rest
            # is launch and drain at the edges of the module.
            if busy is not None and workload_busy is not None:
                slowdown_ms = (busy - workload_busy) / 1e3
                row[f"{tag}_tensorcore_slowdown_ms"] = slowdown_ms
                row[f"{tag}_edge_cost_ms"] = marginal - slowdown_ms
        if "sparsecore_marginal_ms" in row and "xla_marginal_ms" in row:
            row["xla_over_sparsecore"] = (
                row["xla_marginal_ms"] / row["sparsecore_marginal_ms"]
            )
        # A marginal cost at or below zero means adding the gather did not make
        # the module slower, which the hidden/ratio columns cannot express --
        # `hidden` goes above 1.0 and the ratio's sign flips.  Flag it rather
        # than letting a meaningless ratio be read as a very good result.
        for tag in ("sparsecore", "xla"):
            marginal = row.get(f"{tag}_marginal_ms")
            if marginal is not None and marginal <= 0:
                row.setdefault("anomalies", []).append(
                    f"{tag}_marginal_ms <= 0 ({marginal:.5f}): adding the "
                    "gather did not slow the workload down; hidden and "
                    "xla_over_sparsecore are not meaningful for this row"
                )
        rows.append(row)

    summary = {
        "notes": [
            "tensorcore_slowdown_ms = how much longer the TensorCore itself was "
            "busy in the combined run than in the workload-alone run; "
            "edge_cost_ms is the remainder of the marginal cost (launch and "
            "drain).",
            "trace_overlap is the fraction of SparseCore busy time that fell "
            "inside TensorCore busy time.  It can be 1.0 while hidden is near "
            "0.0: running concurrently is not the same as running for free.",
            "xla_over_sparsecore > 1 means the SparseCore kernel is the cheaper "
            "way to gather alongside this workload.",
        ],
        "note": (
            "hidden = 1 - (T_both - T_workload) / T_gather_alone.  1.0 means the "
            "gather was fully overlapped and cost nothing; 0.0 means it was "
            "fully serialized; negative means it contended with the workload "
            "and cost more than running it alone."
        ),
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    header = (
        f"{'workload':>22} {'AI':>6} {'HBM%':>6} {'ownSC':>6} {'dtype':>9} "
        f"{'gRows':>7} {'wkld ms':>9} "
        f"{'sc marg':>9} {'sc hid':>7} {'sc ovl':>7} {'sc slow':>8} "
        f"{'xla marg':>9} {'xla hid':>8} {'xla/sc':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        def show(key: str, fmt: str) -> str:
            value = row.get(key)
            return format(value, fmt) if isinstance(value, (int, float)) else "-"

        print(
            f"{row['workload_label']:>22} {show('arithmetic_intensity', '6.0f')} "
            f"{show('workload_hbm_pct', '6.1f')} "
            f"{show('workload_own_sparsecore_us', '6.1f')} "
            f"{row['gather_dtype']:>9} {row['gather_out_rows']:>7} "
            f"{show('workload_ms', '9.5f')} "
            f"{show('sparsecore_marginal_ms', '9.5f')} "
            f"{show('sparsecore_hidden', '7.2f')} "
            f"{show('sparsecore_trace_overlap', '7.2f')} "
            f"{show('sparsecore_tensorcore_slowdown_ms', '8.5f')} "
            f"{show('xla_marginal_ms', '9.5f')} {show('xla_hidden', '8.2f')} "
            f"{show('xla_over_sparsecore', '7.3f')}"
            f"{'  <- ANOMALY' if row.get('anomalies') else ''}"
        )
    print(f"\nwrote {root / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument(
        "--summarize", action="store_true", help="derive the table; needs no TPU"
    )
    parser.add_argument("--matmul-n", type=int, default=4096)
    # Defaults are the shape profiled in profiles/native/grouped_matmul/, so
    # the grouped matmul here is the same workload the corpus already reports.
    parser.add_argument("--gmm-rows", type=int, default=32768)
    parser.add_argument("--gmm-groups", type=int, default=128)
    parser.add_argument("--gmm-k", type=int, default=4096)
    parser.add_argument("--gmm-n", type=int, default=1536)
    parser.add_argument("--scg-rows", type=int, default=4096)
    parser.add_argument("--scg-hidden", type=int, default=1024)
    parser.add_argument("--scg-out-rows", type=int, default=2048)
    parser.add_argument(
        "--gather-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help=(
            "element type of the gathered rows.  float32 matches what "
            "upstream's own tests use; bfloat16 is what an MoE layer actually "
            "carries and moves half the bytes."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trace-chunk-size", type=int, default=1)
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="retain raw traces (tens of GB for a full sweep)",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profiles" / "native")
    args = parser.parse_args()

    if args.summarize:
        summarize(args)
        return
    if not args.config:
        parser.error("--config is required unless --summarize is given")
    run_one(args)


if __name__ == "__main__":
    main()
