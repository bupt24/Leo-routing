#!/usr/bin/env python3
"""Validate remote-sensing beam-model outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_scenario import LINK_OBSERVATION, LINK_RLS_BACKUP, load_scenario_config  # noqa: E402


REQUIRED_OUTPUTS = ("routes.csv", "topology_edges.csv", "nodes.csv", "metrics_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check remote-sensing beam-model output invariants.")
    parser.add_argument("--output-dir", required=True, help="Directory containing scenario output CSV/JSON files.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"),
        help="Scenario YAML config used to generate the output.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Floating-point tolerance for inequality checks.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_scenario_config(args.config)

    failures: list[str] = []
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        if not path.exists():
            failures.append(f"missing required output file: {path}")

    if failures:
        report(failures)

    edges = read_csv(output_dir / "topology_edges.csv")
    routes = read_csv(output_dir / "routes.csv")
    nodes = read_csv(output_dir / "nodes.csv")

    check_observation_edges(edges, config, args.tolerance, failures)
    check_observation_resource_limits(edges, config, failures)
    check_nodes(nodes, failures)
    check_rls_backup_edges(edges, failures)
    check_routes(routes, failures)

    if failures:
        report(failures)

    print("CHECK PASSED")


def report(failures: list[str]) -> None:
    print("CHECK FAILED")
    for failure in failures:
        print(f"- {failure}")
    raise SystemExit(1)


def check_observation_edges(edges: list[dict[str, str]], config: Any, tolerance: float, failures: list[str]) -> None:
    obs_edges = [edge for edge in edges if edge.get("link_type") == LINK_OBSERVATION]
    if not obs_edges:
        return

    required_fields = (
        "elevation_deg",
        "off_nadir_angle_deg",
        "slant_range_km",
        "beam_gain_db",
        "observation_quality",
        "beam_available",
        "beam_reason",
    )
    for field in required_fields:
        if field not in obs_edges[0]:
            failures.append(f"topology_edges.csv is missing beam field {field!r}")
            return

    for idx, edge in enumerate(obs_edges, start=1):
        elevation = float(edge["elevation_deg"])
        off_nadir = float(edge["off_nadir_angle_deg"])
        slant_range = float(edge["slant_range_km"])
        if not as_bool(edge["beam_available"]):
            failures.append(f"observation edge #{idx} has beam_available=False")
        if edge.get("beam_reason") not in {"available", "legacy_elevation_only"}:
            failures.append(f"observation edge #{idx} has unexpected beam_reason={edge.get('beam_reason')!r}")
        if elevation + tolerance < config.min_elevation_deg:
            failures.append(
                f"observation edge #{idx} elevation {elevation:.6f} < min_elevation_deg {config.min_elevation_deg:.6f}"
            )
        if config.beam.enabled:
            if config.beam.pointing_mode.lower() == "nadir" and off_nadir - tolerance > config.beam.half_angle_deg:
                failures.append(
                    f"observation edge #{idx} off_nadir {off_nadir:.6f} > beam.half_angle_deg "
                    f"{config.beam.half_angle_deg:.6f}"
                )
            if off_nadir - tolerance > config.beam.max_off_nadir_deg:
                failures.append(
                    f"observation edge #{idx} off_nadir {off_nadir:.6f} > beam.max_off_nadir_deg "
                    f"{config.beam.max_off_nadir_deg:.6f}"
                )
            if slant_range - tolerance > config.beam.max_observation_range_km:
                failures.append(
                    f"observation edge #{idx} slant_range {slant_range:.6f} > beam.max_observation_range_km "
                    f"{config.beam.max_observation_range_km:.6f}"
                )


def check_observation_resource_limits(edges: list[dict[str, str]], config: Any, failures: list[str]) -> None:
    per_rls: dict[tuple[str, str], int] = {}
    per_target: dict[tuple[str, str], int] = {}
    for edge in edges:
        if edge.get("link_type") != LINK_OBSERVATION:
            continue
        time_slot = edge["time_slot"]
        per_rls[(time_slot, edge["dst"])] = per_rls.get((time_slot, edge["dst"]), 0) + 1
        per_target[(time_slot, edge["src"])] = per_target.get((time_slot, edge["src"]), 0) + 1

    for (time_slot, rls), count in sorted(per_rls.items()):
        if count > config.beam.max_observation_beams_per_rls:
            failures.append(
                f"time_slot {time_slot} RLS {rls} has {count} observation edges, "
                f"limit={config.beam.max_observation_beams_per_rls}"
            )
    for (time_slot, target), count in sorted(per_target.items()):
        if count > config.beam.max_rls_per_target:
            failures.append(
                f"time_slot {time_slot} target {target} has {count} selected RLS edges, "
                f"limit={config.beam.max_rls_per_target}"
            )


def check_nodes(nodes: list[dict[str, str]], failures: list[str]) -> None:
    orbital = [node for node in nodes if node.get("node_type") in {"rls", "cls"}]
    if not orbital:
        failures.append("nodes.csv has no RLS/CLS nodes")
        return
    for node in orbital:
        if node.get("frame") != "ECEF":
            failures.append(f"{node.get('node')} frame={node.get('frame')!r}, expected ECEF")
        if node.get("motion_type") != "orbital":
            failures.append(f"{node.get('node')} motion_type={node.get('motion_type')!r}, expected orbital")


def check_rls_backup_edges(edges: list[dict[str, str]], failures: list[str]) -> None:
    backup_edges = [edge for edge in edges if edge.get("link_type") == LINK_RLS_BACKUP]
    if not backup_edges:
        failures.append("no RLS-RLS backup edges were produced")
        return
    for edge in backup_edges:
        if edge.get("src_type") != "rls" or edge.get("dst_type") != "rls":
            failures.append(f"rls_backup edge {edge.get('src')}->{edge.get('dst')} is not RLS->RLS")
        if as_bool(edge.get("is_data_flow_allowed")):
            failures.append(f"rls_backup edge {edge.get('src')}->{edge.get('dst')} allows data flow")


def check_routes(routes: list[dict[str, str]], failures: list[str]) -> None:
    if not routes:
        failures.append("routes.csv is empty")
        return
    for route in routes:
        nodes = [part.strip() for part in route.get("path", "").split("->") if part.strip()]
        for src, dst in zip(nodes, nodes[1:]):
            if src.startswith("RLS") and dst.startswith("RLS"):
                failures.append(
                    f"route {route.get('task_id')} at slot {route.get('time_slot')} uses RLS->RLS data path: "
                    f"{route.get('path')}"
                )


if __name__ == "__main__":
    main()
