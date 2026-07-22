#!/usr/bin/env python3
"""Summarize end-to-end delay and energy by time slot from scenario routes."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.link_model import LinkModelConfig, SPEED_OF_LIGHT_KM_PER_MS  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "remote_sensing_scenario"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
METRIC_MODE_RECORDED = "recorded"
METRIC_MODE_SINGLE_PACKET = "single-packet"
METRIC_SUITE_ALL = "all"
METRIC_SUITE_SELECTED = "selected"
LINK_OBSERVATION = "observation"

REQUIRED_ROUTE_COLUMNS = {
    "time_slot",
    "access_success",
    "route_success",
    "total_delay_ms",
    "total_energy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze end-to-end delay and energy variation across time slots."
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Scenario output directory containing routes.csv. Defaults to the latest output directory.",
    )
    parser.add_argument(
        "--routes",
        default="",
        help="Path to routes.csv. Overrides --output-dir.",
    )
    parser.add_argument(
        "--metric-mode",
        choices=[METRIC_MODE_RECORDED, METRIC_MODE_SINGLE_PACKET],
        default=METRIC_MODE_RECORDED,
        help=(
            "Metric source for delay plots when --metric-suite selected is used. "
            "'recorded' uses total_delay_ms from routes.csv. 'single-packet' recomputes one-packet "
            "delay from the route path and topology_edges.csv."
        ),
    )
    parser.add_argument(
        "--energy-metric-mode",
        choices=[METRIC_MODE_RECORDED, METRIC_MODE_SINGLE_PACKET],
        default=METRIC_MODE_RECORDED,
        help=(
            "Metric source for energy plots when --metric-suite selected is used. Defaults to 'recorded', "
            "meaning total_energy from routes.csv. Use 'single-packet' to recompute one-packet energy from "
            "topology_edges.csv."
        ),
    )
    parser.add_argument(
        "--metric-suite",
        choices=[METRIC_SUITE_ALL, METRIC_SUITE_SELECTED],
        default=METRIC_SUITE_ALL,
        help=(
            "'all' writes total recorded delay/energy and, when topology_edges.csv is available, "
            "single-packet delay/energy. 'selected' writes only the modes chosen by --metric-mode and "
            "--energy-metric-mode."
        ),
    )
    parser.add_argument(
        "--topology-edges",
        default="",
        help=(
            "Path to topology_edges.csv used by --metric-mode single-packet. "
            "Defaults to topology_edges.csv next to routes.csv."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory used when locating the latest scenario output.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Scenario config used only to infer time_slot_seconds for the CSV.",
    )
    parser.add_argument(
        "--time-slot-seconds",
        type=float,
        default=None,
        help="Override time slot length in seconds for the time_sec column.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed route rows in delay/energy statistics. By default only successful routes are measured.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path for the per-slot summary CSV. Defaults to figures_Nslots/end_to_end_delay_energy_by_slot_Nslots.csv.",
    )
    parser.add_argument(
        "--plot-path",
        default="",
        help=(
            "Path for the delay PNG plot. Defaults to figures_Nslots/end_to_end_delay_by_slot_Nslots.png. "
            "When --energy-plot-path is omitted, the energy plot path is derived from this path."
        ),
    )
    parser.add_argument(
        "--energy-plot-path",
        default="",
        help="Path for the energy PNG plot. Defaults to figures_Nslots/end_to_end_energy_by_slot_Nslots.png.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation and only write the CSV summary.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value == "":
        return default
    return float(value)


def find_latest_output_dir(output_root: Path) -> Path:
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir() and (path / "routes.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No routes.csv found under {output_root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_routes_path(args: argparse.Namespace) -> Path:
    if args.routes:
        routes_path = Path(args.routes)
    elif args.output_dir:
        routes_path = Path(args.output_dir) / "routes.csv"
    else:
        routes_path = find_latest_output_dir(Path(args.output_root)) / "routes.csv"

    if not routes_path.exists():
        raise FileNotFoundError(f"routes.csv not found: {routes_path}")
    return routes_path


def read_routes(routes_path: Path) -> list[dict[str, str]]:
    with routes_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_ROUTE_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{routes_path} is missing required columns: {sorted(missing)}")
        return list(reader)


def resolve_topology_edges_path(args: argparse.Namespace, routes_path: Path) -> Path:
    topology_edges_path = Path(args.topology_edges) if args.topology_edges else routes_path.parent / "topology_edges.csv"
    if not topology_edges_path.exists():
        raise FileNotFoundError(
            f"{METRIC_MODE_SINGLE_PACKET} metrics require topology_edges.csv: {topology_edges_path}"
        )
    return topology_edges_path


def find_optional_topology_edges_path(args: argparse.Namespace, routes_path: Path) -> Path | None:
    topology_edges_path = Path(args.topology_edges) if args.topology_edges else routes_path.parent / "topology_edges.csv"
    if topology_edges_path.exists():
        return topology_edges_path
    if args.topology_edges:
        raise FileNotFoundError(f"topology_edges.csv not found: {topology_edges_path}")
    return None


def read_topology_edges(topology_edges_path: Path) -> dict[tuple[int, str, str], dict[str, str]]:
    required_columns = {
        "time_slot",
        "src",
        "dst",
        "link_type",
        "distance_km",
        "capacity_bps",
        "delay_ms",
        "energy_cost",
    }
    edges: dict[tuple[int, str, str], dict[str, str]] = {}
    with topology_edges_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(f"{topology_edges_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            edges[(int(row["time_slot"]), row["src"], row["dst"])] = row
    return edges


def infer_time_slot_seconds(config_path: Path) -> float | None:
    if not config_path.exists():
        return None

    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("time_slot_seconds:"):
            continue
        value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
        try:
            return float(value)
        except ValueError:
            return None
    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * p / 100.0
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return ordered[int(rank)]

    lower = ordered[lower_idx]
    upper = ordered[upper_idx]
    return lower + (upper - lower) * (rank - lower_idx)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def parse_route_path(path: str) -> list[str]:
    return [node.strip() for node in path.split("->") if node.strip()]


def single_packet_edge_metrics(edge: dict[str, str], link_config: LinkModelConfig) -> tuple[float, float] | None:
    if edge["link_type"] == LINK_OBSERVATION:
        return float(edge["delay_ms"]), float(edge["energy_cost"])

    capacity_bps = float(edge["capacity_bps"])
    if capacity_bps <= 0.0:
        return None

    distance_km = float(edge["distance_km"])
    tx_duration_s = float(link_config.packet_size_bits) / capacity_bps
    delay_ms = distance_km / SPEED_OF_LIGHT_KM_PER_MS + tx_duration_s * 1000.0
    energy_j = (float(link_config.tx_power_w) + float(link_config.rx_power_w)) * tx_duration_s
    return delay_ms, energy_j


def single_packet_route_metrics(
    row: dict[str, str],
    topology_edges: dict[tuple[int, str, str], dict[str, str]],
    link_config: LinkModelConfig,
) -> tuple[float, float] | None:
    nodes = parse_route_path(row.get("path", ""))
    if len(nodes) < 2:
        return None

    time_slot = int(row["time_slot"])
    total_delay_ms = 0.0
    total_energy_j = 0.0
    for src, dst in zip(nodes, nodes[1:]):
        edge = topology_edges.get((time_slot, src, dst))
        if edge is None:
            return None
        metrics = single_packet_edge_metrics(edge, link_config)
        if metrics is None:
            return None
        edge_delay_ms, edge_energy_j = metrics
        total_delay_ms += edge_delay_ms
        total_energy_j += edge_energy_j
    return total_delay_ms, total_energy_j


def measured_delay_energy(
    rows: list[dict[str, str]],
    delay_metric_mode: str,
    energy_metric_mode: str,
    topology_edges: dict[tuple[int, str, str], dict[str, str]] | None,
    link_config: LinkModelConfig,
) -> tuple[list[float], list[float]]:
    delays: list[float] = []
    energies: list[float] = []
    for row in rows:
        single_packet_metrics = None
        if delay_metric_mode == METRIC_MODE_SINGLE_PACKET or energy_metric_mode == METRIC_MODE_SINGLE_PACKET:
            if topology_edges is None:
                raise ValueError(f"single-packet metrics require topology_edges")
            single_packet_metrics = single_packet_route_metrics(row, topology_edges, link_config)
            if single_packet_metrics is None:
                continue

        if delay_metric_mode == METRIC_MODE_RECORDED:
            delays.append(as_float(row, "total_delay_ms"))
        else:
            assert single_packet_metrics is not None
            delays.append(single_packet_metrics[0])

        if energy_metric_mode == METRIC_MODE_RECORDED:
            energies.append(as_float(row, "total_energy"))
        else:
            assert single_packet_metrics is not None
            energies.append(single_packet_metrics[1])
    return delays, energies


def build_slot_summary(
    routes: list[dict[str, str]],
    time_slot_seconds: float | None,
    include_failed: bool,
    delay_metric_mode: str,
    energy_metric_mode: str,
    topology_edges: dict[tuple[int, str, str], dict[str, str]] | None = None,
    link_config: LinkModelConfig | None = None,
) -> list[dict[str, Any]]:
    link_config = link_config or LinkModelConfig()
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in routes:
        grouped[int(row["time_slot"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for time_slot in sorted(grouped):
        rows = grouped[time_slot]
        success_rows = [
            row for row in rows
            if as_bool(row.get("access_success")) and as_bool(row.get("route_success"))
        ]
        access_failed = [row for row in rows if not as_bool(row.get("access_success"))]
        route_failed = [
            row for row in rows
            if as_bool(row.get("access_success")) and not as_bool(row.get("route_success"))
        ]
        measured_rows = rows if include_failed else success_rows

        delays, energies = measured_delay_energy(
            measured_rows,
            delay_metric_mode=delay_metric_mode,
            energy_metric_mode=energy_metric_mode,
            topology_edges=topology_edges,
            link_config=link_config,
        )
        hop_counts = [as_float(row, "hop_count") for row in measured_rows if row.get("hop_count", "") != ""]
        cls_relays = [
            as_float(row, "cls_relay_hops")
            for row in measured_rows
            if row.get("cls_relay_hops", "") != ""
        ]
        queue_costs = [
            as_float(row, "total_queue_cost")
            for row in measured_rows
            if row.get("total_queue_cost", "") != ""
        ]
        loss_risks = [
            as_float(row, "total_loss_risk")
            for row in measured_rows
            if row.get("total_loss_risk", "") != ""
        ]

        success_count = len(success_rows)
        total_count = len(rows)
        summary_rows.append(
            {
                "time_slot": time_slot,
                "time_sec": None if time_slot_seconds is None else time_slot * time_slot_seconds,
                "total_tasks": total_count,
                "success_tasks": success_count,
                "access_failed_tasks": len(access_failed),
                "route_failed_tasks": len(route_failed),
                "success_rate": success_count / total_count if total_count else None,
                "measured_tasks": len(delays),
                "avg_delay_ms": average(delays),
                "min_delay_ms": min(delays) if delays else None,
                "p50_delay_ms": percentile(delays, 50.0),
                "p90_delay_ms": percentile(delays, 90.0),
                "p95_delay_ms": percentile(delays, 95.0),
                "max_delay_ms": max(delays) if delays else None,
                "avg_energy_j": average(energies),
                "min_energy_j": min(energies) if energies else None,
                "p50_energy_j": percentile(energies, 50.0),
                "p90_energy_j": percentile(energies, 90.0),
                "p95_energy_j": percentile(energies, 95.0),
                "max_energy_j": max(energies) if energies else None,
                "total_energy_j": sum(energies) if energies else None,
                "avg_hop_count": average(hop_counts),
                "avg_cls_relay_hops": average(cls_relays),
                "avg_queue_cost": average(queue_costs),
                "avg_loss_risk": average(loss_risks),
            }
        )
    return summary_rows


def format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return value


def write_summary_csv(summary_rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_slot",
        "time_sec",
        "total_tasks",
        "success_tasks",
        "access_failed_tasks",
        "route_failed_tasks",
        "success_rate",
        "measured_tasks",
        "avg_delay_ms",
        "min_delay_ms",
        "p50_delay_ms",
        "p90_delay_ms",
        "p95_delay_ms",
        "max_delay_ms",
        "avg_energy_j",
        "min_energy_j",
        "p50_energy_j",
        "p90_energy_j",
        "p95_energy_j",
        "max_energy_j",
        "total_energy_j",
        "avg_hop_count",
        "avg_cls_relay_hops",
        "avg_queue_cost",
        "avg_loss_risk",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: format_csv_value(row.get(field)) for field in fieldnames})


def default_artifact_paths(
    output_dir: Path,
    slot_count: int,
    delay_metric_mode: str,
    energy_metric_mode: str,
) -> tuple[Path, Path, Path]:
    artifact_dir = output_dir / f"figures_{slot_count}slots"
    if delay_metric_mode == METRIC_MODE_RECORDED and energy_metric_mode == METRIC_MODE_RECORDED:
        csv_name = f"end_to_end_delay_energy_by_slot_{slot_count}slots.csv"
        delay_name = f"end_to_end_delay_by_slot_{slot_count}slots.png"
        energy_name = f"end_to_end_energy_by_slot_{slot_count}slots.png"
    elif delay_metric_mode == METRIC_MODE_SINGLE_PACKET and energy_metric_mode == METRIC_MODE_RECORDED:
        csv_name = f"single_packet_delay_total_energy_by_slot_{slot_count}slots.csv"
        delay_name = f"single_packet_end_to_end_delay_by_slot_{slot_count}slots.png"
        energy_name = f"total_end_to_end_energy_by_slot_{slot_count}slots.png"
    elif delay_metric_mode == METRIC_MODE_SINGLE_PACKET and energy_metric_mode == METRIC_MODE_SINGLE_PACKET:
        csv_name = f"single_packet_end_to_end_delay_energy_by_slot_{slot_count}slots.csv"
        delay_name = f"single_packet_end_to_end_delay_by_slot_{slot_count}slots.png"
        energy_name = f"single_packet_end_to_end_energy_by_slot_{slot_count}slots.png"
    else:
        csv_name = f"recorded_delay_single_packet_energy_by_slot_{slot_count}slots.csv"
        delay_name = f"end_to_end_delay_by_slot_{slot_count}slots.png"
        energy_name = f"single_packet_end_to_end_energy_by_slot_{slot_count}slots.png"
    return artifact_dir / csv_name, artifact_dir / delay_name, artifact_dir / energy_name


def derive_energy_plot_path(delay_plot_path: Path) -> Path:
    return delay_plot_path.with_name(f"{delay_plot_path.stem}_energy{delay_plot_path.suffix}")


def plot_summary(
    summary_rows: list[dict[str, Any]],
    delay_plot_path: Path,
    energy_plot_path: Path,
    time_slot_seconds: float | None,
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return f"Skipped plot: {exc}"

    delay_plot_path.parent.mkdir(parents=True, exist_ok=True)
    energy_plot_path.parent.mkdir(parents=True, exist_ok=True)

    slots = [int(row["time_slot"]) for row in summary_rows]
    avg_delays = [float_or_nan(row["avg_delay_ms"]) for row in summary_rows]
    p90_delays = [float_or_nan(row["p90_delay_ms"]) for row in summary_rows]
    avg_energies = [float_or_nan(row["avg_energy_j"]) for row in summary_rows]
    p90_energies = [float_or_nan(row["p90_energy_j"]) for row in summary_rows]
    success_rates = [float_or_nan(row["success_rate"]) * 100.0 for row in summary_rows]
    x_label = "Time slot"
    if time_slot_seconds is not None:
        x_label += f" ({time_slot_seconds:g} s/slot)"

    fig, (delay_ax, success_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    delay_ax.plot(slots, avg_delays, color="#1f77b4", linewidth=1.8, label="Avg delay")
    delay_ax.plot(slots, p90_delays, color="#1f77b4", linestyle="--", linewidth=1.2, label="P90 delay")
    delay_ax.set_ylabel("Delay (ms)")
    delay_ax.grid(True, axis="both", linestyle="--", alpha=0.3)
    delay_ax.legend(loc="upper right")

    success_ax.plot(slots, success_rates, color="#444444", linewidth=1.4, label="Success rate")
    success_ax.set_ylim(0, 105)
    success_ax.set_ylabel("Success (%)")
    success_ax.grid(True, axis="both", linestyle="--", alpha=0.3)
    success_ax.set_xlabel(x_label)
    fig.suptitle("End-to-end delay by time slot")
    fig.tight_layout()
    fig.savefig(delay_plot_path, dpi=180)
    plt.close(fig)

    fig, (energy_ax, success_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    energy_ax.plot(slots, avg_energies, color="#2ca02c", linewidth=1.8, label="Avg energy")
    energy_ax.plot(slots, p90_energies, color="#2ca02c", linestyle="--", linewidth=1.2, label="P90 energy")
    energy_ax.set_ylabel("Energy (J)")
    energy_ax.grid(True, axis="both", linestyle="--", alpha=0.3)
    energy_ax.legend(loc="upper right")

    success_ax.plot(slots, success_rates, color="#444444", linewidth=1.4, label="Success rate")
    success_ax.set_ylim(0, 105)
    success_ax.set_ylabel("Success (%)")
    success_ax.grid(True, axis="both", linestyle="--", alpha=0.3)
    success_ax.set_xlabel(x_label)
    fig.suptitle("End-to-end energy by time slot")
    fig.tight_layout()
    fig.savefig(energy_plot_path, dpi=180)
    plt.close(fig)
    return None


def float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def summarize_overall(
    routes: list[dict[str, str]],
    include_failed: bool,
    delay_metric_mode: str,
    energy_metric_mode: str,
    topology_edges: dict[tuple[int, str, str], dict[str, str]] | None = None,
    link_config: LinkModelConfig | None = None,
) -> dict[str, Any]:
    link_config = link_config or LinkModelConfig()
    success_rows = [
        row for row in routes
        if as_bool(row.get("access_success")) and as_bool(row.get("route_success"))
    ]
    measured_rows = routes if include_failed else success_rows
    delays, energies = measured_delay_energy(
        measured_rows,
        delay_metric_mode=delay_metric_mode,
        energy_metric_mode=energy_metric_mode,
        topology_edges=topology_edges,
        link_config=link_config,
    )
    return {
        "total_tasks": len(routes),
        "success_tasks": len(success_rows),
        "measured_tasks": len(delays),
        "avg_delay_ms": average(delays),
        "avg_energy_j": average(energies),
    }


def run_analysis(
    *,
    routes: list[dict[str, str]],
    routes_path: Path,
    output_dir: Path,
    time_slot_seconds: float | None,
    include_failed: bool,
    delay_metric_mode: str,
    energy_metric_mode: str,
    topology_edges: dict[tuple[int, str, str], dict[str, str]] | None,
    output_csv: str = "",
    plot_path: str = "",
    energy_plot_path: str = "",
    no_plot: bool = False,
) -> dict[str, Any]:
    summary_rows = build_slot_summary(
        routes,
        time_slot_seconds,
        include_failed=include_failed,
        delay_metric_mode=delay_metric_mode,
        energy_metric_mode=energy_metric_mode,
        topology_edges=topology_edges,
    )
    if not summary_rows:
        raise SystemExit("No time slots found in routes.csv")

    default_csv_path, default_delay_plot_path, default_energy_plot_path = default_artifact_paths(
        output_dir,
        len(summary_rows),
        delay_metric_mode,
        energy_metric_mode,
    )
    csv_path = Path(output_csv) if output_csv else default_csv_path
    delay_plot_path = Path(plot_path) if plot_path else default_delay_plot_path
    if energy_plot_path:
        resolved_energy_plot_path = Path(energy_plot_path)
    elif plot_path:
        resolved_energy_plot_path = derive_energy_plot_path(delay_plot_path)
    else:
        resolved_energy_plot_path = default_energy_plot_path

    write_summary_csv(summary_rows, csv_path)

    plot_message = None
    if not no_plot:
        plot_message = plot_summary(
            summary_rows,
            delay_plot_path,
            resolved_energy_plot_path,
            time_slot_seconds,
        )

    overall = summarize_overall(
        routes,
        include_failed=include_failed,
        delay_metric_mode=delay_metric_mode,
        energy_metric_mode=energy_metric_mode,
        topology_edges=topology_edges,
    )
    print(f"Routes: {routes_path}")
    print(f"Delay metric mode: {delay_metric_mode}")
    print(f"Energy metric mode: {energy_metric_mode}")
    print(f"Summary CSV: {csv_path}")
    if no_plot:
        print("Plot: skipped by --no-plot")
    elif plot_message:
        print(f"Plot: {plot_message}")
    else:
        print(f"Delay plot: {delay_plot_path}")
        print(f"Energy plot: {resolved_energy_plot_path}")
    print(
        "Overall: "
        f"slots={len(summary_rows)} "
        f"tasks={overall['total_tasks']} "
        f"success={overall['success_tasks']} "
        f"measured={overall['measured_tasks']} "
        f"avg_delay_ms={format_csv_value(overall['avg_delay_ms'])} "
        f"avg_energy_j={format_csv_value(overall['avg_energy_j'])}"
    )
    return {
        "summary_rows": summary_rows,
        "overall": overall,
        "csv_path": csv_path,
        "delay_plot_path": delay_plot_path,
        "energy_plot_path": resolved_energy_plot_path,
    }


def main() -> None:
    args = parse_args()
    routes_path = resolve_routes_path(args)
    output_dir = routes_path.parent
    routes = read_routes(routes_path)
    if not routes:
        raise SystemExit(f"No route rows found in {routes_path}")

    time_slot_seconds = args.time_slot_seconds
    if time_slot_seconds is None:
        time_slot_seconds = infer_time_slot_seconds(Path(args.config))

    if args.metric_suite == METRIC_SUITE_SELECTED:
        topology_edges = None
        if args.metric_mode == METRIC_MODE_SINGLE_PACKET or args.energy_metric_mode == METRIC_MODE_SINGLE_PACKET:
            topology_edges = read_topology_edges(resolve_topology_edges_path(args, routes_path))
        run_analysis(
            routes=routes,
            routes_path=routes_path,
            output_dir=output_dir,
            time_slot_seconds=time_slot_seconds,
            include_failed=args.include_failed,
            delay_metric_mode=args.metric_mode,
            energy_metric_mode=args.energy_metric_mode,
            topology_edges=topology_edges,
            output_csv=args.output_csv,
            plot_path=args.plot_path,
            energy_plot_path=args.energy_plot_path,
            no_plot=args.no_plot,
        )
        return

    if args.output_csv or args.plot_path or args.energy_plot_path:
        raise SystemExit(
            "Custom --output-csv/--plot-path/--energy-plot-path requires --metric-suite selected."
        )

    print("Metric suite: all")
    run_analysis(
        routes=routes,
        routes_path=routes_path,
        output_dir=output_dir,
        time_slot_seconds=time_slot_seconds,
        include_failed=args.include_failed,
        delay_metric_mode=METRIC_MODE_RECORDED,
        energy_metric_mode=METRIC_MODE_RECORDED,
        topology_edges=None,
        no_plot=args.no_plot,
    )

    topology_edges_path = find_optional_topology_edges_path(args, routes_path)
    if topology_edges_path is None:
        print("Single-packet metrics: skipped because topology_edges.csv was not found.")
        return

    topology_edges = read_topology_edges(topology_edges_path)
    print("")
    run_analysis(
        routes=routes,
        routes_path=routes_path,
        output_dir=output_dir,
        time_slot_seconds=time_slot_seconds,
        include_failed=args.include_failed,
        delay_metric_mode=METRIC_MODE_SINGLE_PACKET,
        energy_metric_mode=METRIC_MODE_SINGLE_PACKET,
        topology_edges=topology_edges,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
