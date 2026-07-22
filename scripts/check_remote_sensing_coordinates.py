#!/usr/bin/env python3
"""Check coordinate-frame and ground-motion invariants for the remote-sensing scenario."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_scenario import (  # noqa: E402
    RemoteSensingScenario,
    ScenarioEdge,
    ScenarioNodeRecord,
    load_scenario_config,
    write_scenario_outputs,
)


GROUND_NODE_TYPES = {"target", "es"}
ORBITAL_NODE_TYPES = {"rls", "cls"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ECEF coordinate and motion invariants for the remote-sensing scenario."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"),
        help="Path to the remote sensing scenario YAML config.",
    )
    parser.add_argument(
        "--time-slots",
        type=int,
        default=3,
        help="Number of deterministic time slots to simulate for the check.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory where routes.csv, topology_edges.csv, nodes.csv, and metrics are written.",
    )
    parser.add_argument(
        "--position-tolerance-km",
        type=float,
        default=1e-3,
        help="Distance tolerance for equality checks in kilometers.",
    )
    parser.add_argument(
        "--latlon-tolerance-deg",
        type=float,
        default=1e-9,
        help="Latitude/longitude tolerance for fixed ground points in degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_slots < 2:
        raise SystemExit("FAILED: --time-slots must be at least 2 to check motion.")

    config = load_scenario_config(args.config)
    scenario = RemoteSensingScenario(config)
    result = scenario.run(num_time_slots=args.time_slots)

    if args.output_dir:
        write_scenario_outputs(result, args.output_dir)

    failures = check_result(
        result.topology_nodes,
        result.topology_edges,
        position_tolerance_km=args.position_tolerance_km,
        latlon_tolerance_deg=args.latlon_tolerance_deg,
    )
    if failures:
        print("FAILED remote-sensing coordinate checks:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Remote-sensing coordinate checks passed:")
    print(f"- RLS/CLS positions changed across {args.time_slots} time slots.")
    print("- Fixed target/ES latitude and longitude stayed constant.")
    print("- Moving target/user latitude or longitude changed across time slots.")
    print("- Edge distances match ECEF node coordinates.")


def check_result(
    node_records: list[ScenarioNodeRecord],
    edges: list[ScenarioEdge],
    position_tolerance_km: float,
    latlon_tolerance_deg: float,
) -> list[str]:
    failures: list[str] = []
    records_by_node: dict[str, list[ScenarioNodeRecord]] = defaultdict(list)
    for record in node_records:
        records_by_node[record.node].append(record)
        if record.frame != "ECEF":
            failures.append(f"{record.node} at slot {record.time_slot} has frame={record.frame!r}, expected 'ECEF'.")

    for records in records_by_node.values():
        records.sort(key=lambda record: record.time_slot)

    _check_orbital_nodes_move(records_by_node, failures, position_tolerance_km)
    _check_fixed_ground_latlon(records_by_node, failures, latlon_tolerance_deg)
    _check_moving_ground_latlon(records_by_node, failures, latlon_tolerance_deg)
    _check_edge_distances(node_records, edges, failures, position_tolerance_km)
    return failures


def _check_orbital_nodes_move(
    records_by_node: dict[str, list[ScenarioNodeRecord]],
    failures: list[str],
    tolerance_km: float,
) -> None:
    orbital_nodes = [records for records in records_by_node.values() if records[0].node_type in ORBITAL_NODE_TYPES]
    if not orbital_nodes:
        failures.append("No RLS/CLS nodes were produced.")
        return

    for records in orbital_nodes:
        for prev, current in zip(records, records[1:]):
            delta_km = _record_distance_km(prev, current)
            if delta_km <= tolerance_km:
                failures.append(
                    f"{current.node} did not move between slots {prev.time_slot} and {current.time_slot} "
                    f"(delta={delta_km:.6g} km)."
                )


def _check_fixed_ground_latlon(
    records_by_node: dict[str, list[ScenarioNodeRecord]],
    failures: list[str],
    tolerance_deg: float,
) -> None:
    fixed_nodes = [
        records
        for records in records_by_node.values()
        if records[0].node_type in GROUND_NODE_TYPES and records[0].motion_type == "fixed"
    ]
    if not fixed_nodes:
        failures.append("No fixed target/ES nodes were produced.")
        return

    for records in fixed_nodes:
        first = records[0]
        for current in records[1:]:
            lat_delta = abs(current.lat_deg - first.lat_deg)
            lon_delta = abs(current.lon_deg - first.lon_deg)
            if lat_delta > tolerance_deg or lon_delta > tolerance_deg:
                failures.append(
                    f"Fixed node {current.node} changed lat/lon by "
                    f"({lat_delta:.6g}, {lon_delta:.6g}) deg."
                )


def _check_moving_ground_latlon(
    records_by_node: dict[str, list[ScenarioNodeRecord]],
    failures: list[str],
    tolerance_deg: float,
) -> None:
    moving_nodes = [
        records
        for records in records_by_node.values()
        if records[0].node_type in GROUND_NODE_TYPES and records[0].motion_type == "moving"
    ]
    if not moving_nodes:
        failures.append("No moving target/user nodes were produced.")
        return

    for records in moving_nodes:
        changed = any(
            abs(current.lat_deg - records[0].lat_deg) > tolerance_deg
            or abs(current.lon_deg - records[0].lon_deg) > tolerance_deg
            for current in records[1:]
        )
        if not changed:
            failures.append(f"Moving node {records[0].node} did not change lat/lon across time slots.")


def _check_edge_distances(
    node_records: list[ScenarioNodeRecord],
    edges: list[ScenarioEdge],
    failures: list[str],
    tolerance_km: float,
) -> None:
    if not edges:
        failures.append("No topology edges were produced, so link distance coordinates could not be checked.")
        return

    positions = {
        (record.time_slot, record.node): (record.x_km, record.y_km, record.z_km)
        for record in node_records
    }
    for edge in edges:
        src_position = positions.get((edge.time_slot, edge.src))
        dst_position = positions.get((edge.time_slot, edge.dst))
        if src_position is None or dst_position is None:
            failures.append(f"Missing node position for edge {edge.src}->{edge.dst} at slot {edge.time_slot}.")
            continue
        distance_km = math.dist(src_position, dst_position)
        if abs(distance_km - edge.distance_km) > tolerance_km:
            failures.append(
                f"Edge {edge.src}->{edge.dst} at slot {edge.time_slot} has distance {edge.distance_km:.6f} km, "
                f"but ECEF node positions give {distance_km:.6f} km."
            )


def _record_distance_km(first: ScenarioNodeRecord, second: ScenarioNodeRecord) -> float:
    return math.dist(
        (first.x_km, first.y_km, first.z_km),
        (second.x_km, second.y_km, second.z_km),
    )


if __name__ == "__main__":
    main()
