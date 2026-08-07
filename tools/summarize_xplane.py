"""Convert corpus XPlane traces into compact per-op and roofline summaries."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from xprof.convert.raw_to_tool_data import xspace_to_tool_data


def table_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = [column["id"] for column in table.get("cols", [])]
    result = []
    for row in table.get("rows", []):
        values = [
            cell.get("v") if cell else None
            for cell in row.get("c", [])
        ]
        result.append(dict(zip(columns, values)))
    return result


def load_tool(xplanes: list[str], tool: str) -> Any:
    data, _ = xspace_to_tool_data(
        xplanes, tool, {"use_saved_result": False}
    )
    if isinstance(data, bytes):
        data = data.decode()
    return json.loads(data)


def summarize_chunk(chunk: Path) -> dict[str, Any]:
    xplanes = glob.glob(str(chunk / "**" / "*.xplane.pb"), recursive=True)
    if not xplanes:
        raise FileNotFoundError(chunk)

    framework_tables = load_tool(xplanes, "framework_op_stats")
    device_ops = []
    for table in framework_tables:
        for row in table_rows(table):
            if (
                row.get("host_or_device") == "Device"
                and row.get("operation") != "IDLE"
            ):
                device_ops.append(
                    {
                        "operation": row.get("operation"),
                        "type": row.get("type"),
                        "occurrences": row.get("occurrences"),
                        "total_time_us": row.get("total_time"),
                        "average_time_us": row.get("avg_time"),
                        "device_self_time_pct": (
                            100 * row["device_total_self_time_percent"]
                            if row.get("device_total_self_time_percent") is not None
                            else None
                        ),
                        "model_gflops_per_second": row.get("model_flop_rate"),
                        "memory_gbytes_per_second": row.get("measured_memory_bw"),
                        "operational_intensity": row.get("operational_intensity"),
                        "bound_by": row.get("bound_by"),
                    }
                )

    roofline_tables = load_tool(xplanes, "roofline_model")
    roofline_ops = []
    hardware = {}
    for table in roofline_tables:
        if table.get("p", {}).get("device_type"):
            hardware = table["p"]
        for row in table_rows(table):
            if (
                row.get("step") == "Total"
                and row.get("category") not in (None, "Program")
            ):
                roofline_ops.append(
                    {
                        key: row.get(key)
                        for key in (
                            "category",
                            "operation",
                            "occurrences",
                            "total_time",
                            "avg_time",
                            "measured_flop_rate",
                            "model_flop_rate",
                            "measured_memory_bw",
                            "operational_intensity",
                            "bound_by",
                            "roofline_efficiency",
                            "compute_efficiency",
                            "max_mem_bw_utilization",
                            "source_info",
                        )
                    }
                )

    return {
        "xplane_files": xplanes,
        "device_ops": device_ops,
        "roofline_hardware": hardware,
        "roofline_ops": roofline_ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).parents[1] / "profiles",
    )
    args = parser.parse_args()
    results = {}
    for result_path in sorted(args.profiles.glob("**/result.json")):
        run = json.loads(result_path.read_text())
        # Key by implementation *and* run label: one implementation can be
        # profiled in several regimes (e.g. balanced vs unbalanced routing),
        # and keying by name alone silently keeps only the last one.
        key = run["implementation"]
        if run.get("run_label"):
            key = f"{key}__{run['run_label']}"
        if key in results:
            raise KeyError(f"duplicate summary key {key!r} from {result_path}")
        chunks = run.get("trace_chunks")
        chunk = Path(chunks[0] if chunks else run["trace_directory"])
        results[key] = summarize_chunk(chunk)
    output = args.profiles / "xplane_summary.json"
    output.write_text(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
