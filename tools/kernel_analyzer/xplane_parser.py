"""Kernel-agnostic XPlane trace parser for TPU profiling."""
import os
import glob


def find_xplane(trace_dir):
    """Find the first .xplane.pb file in a trace directory."""
    pattern = os.path.join(trace_dir, "**", "*.xplane.pb")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No .xplane.pb found in {trace_dir}")
    return matches[0]


def _parse_xplane(xplane_path):
    """Parse an xplane.pb file. Returns list of (op_name, duration_ms)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    xspace = xplane_pb2.XSpace()
    with open(xplane_path, "rb") as f:
        xspace.ParseFromString(f.read())

    ops = []
    for plane in xspace.planes:
        if "TPU" not in plane.name:
            continue

        em = plane.event_metadata
        sm = plane.stat_metadata

        dur_stat_id = None
        for sid, smd in sm.items():
            if smd.name == "device_duration_ps":
                dur_stat_id = sid
                break

        if dur_stat_id is None:
            continue

        for line in plane.lines:
            for ev in line.events:
                ev_name = em[ev.metadata_id].name
                for s in ev.stats:
                    if s.metadata_id == dur_stat_id:
                        dur_ps = s.int64_value or s.uint64_value
                        ops.append((ev_name, dur_ps / 1e9))

    return ops


def get_per_op_breakdown(trace_dir, top_n=20):
    """Get per-op device time breakdown.

    Returns list of (op_name, duration_ms, pct_of_total), sorted by duration descending.
    """
    xplane_path = find_xplane(trace_dir)
    raw = _parse_xplane(xplane_path)

    if not raw:
        return []

    from collections import defaultdict
    totals = defaultdict(list)
    for name, dur_ms in raw:
        totals[name].append(dur_ms)

    agg = []
    for name, durs in totals.items():
        avg_dur = sum(durs) / len(durs)
        agg.append((name, avg_dur))

    agg.sort(key=lambda x: x[1], reverse=True)
    total_ms = sum(d for _, d in agg)

    result = []
    for name, dur in agg[:top_n]:
        pct = dur / total_ms * 100 if total_ms > 0 else 0
        result.append((name, dur, pct))

    return result


def get_total_program_time(trace_dir):
    """Get total device time in ms across all ops."""
    breakdown = get_per_op_breakdown(trace_dir, top_n=10000)
    return sum(d for _, d, _ in breakdown)


def get_op_time(trace_dir, op_name):
    """Get average device time in ms for a specific op name."""
    xplane_path = find_xplane(trace_dir)
    raw = _parse_xplane(xplane_path)
    durs = [d for n, d in raw if n == op_name]
    if not durs:
        raise ValueError(f"Op '{op_name}' not found in trace. Available: {sorted(set(n for n, _ in raw))}")
    return sum(durs) / len(durs)
