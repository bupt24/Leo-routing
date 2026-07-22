"""Shared routing metrics for remote-sensing CLS routing experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from env.link_model import LinkModelConfig, SPEED_OF_LIGHT_KM_PER_MS
from env.remote_sensing_scenario import (
    LINK_DOWNLINK,
    LINK_ISL,
    LINK_OBSERVATION,
    NODE_CLS,
    NODE_ES,
    RemoteSensingScenario,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioSnapshot,
)


CLS_ROUTING_LINK_TYPES = {LINK_ISL, LINK_DOWNLINK}


def eligible_cls_edges(
    snapshot: ScenarioSnapshot,
    src_id: int,
    visited_cls_ids: set[int] | None = None,
) -> list[ScenarioEdge]:
    """Return currently legal MAPPO/Random-style CLS forwarding edges.

    This deliberately performs no future reachability or BFS look-ahead check.
    """

    visited_cls_ids = visited_cls_ids or set()
    edges = [
        edge
        for edge in snapshot.edges
        if edge.src_id == src_id
        and edge.available
        and edge.is_data_flow_allowed
        and edge.link_type in CLS_ROUTING_LINK_TYPES
        and edge.dst_type in {NODE_CLS, NODE_ES}
        and not (edge.dst_type == NODE_CLS and edge.dst_id in visited_cls_ids)
    ]
    edges.sort(key=lambda edge: (edge.dst, edge.dst_id, edge.link_type))
    return edges


def find_edge(
    snapshot: ScenarioSnapshot,
    src_id: int,
    dst_id: int,
    link_type: str | None = None,
) -> ScenarioEdge | None:
    for edge in snapshot.edges:
        if edge.src_id != src_id or edge.dst_id != dst_id:
            continue
        if link_type is not None and edge.link_type != link_type:
            continue
        return edge
    return None


def find_edge_by_label(
    snapshot: ScenarioSnapshot,
    src_label: str,
    dst_label: str,
    link_type: str | None = None,
) -> ScenarioEdge | None:
    for edge in snapshot.edges:
        if edge.src != src_label or edge.dst != dst_label:
            continue
        if link_type is not None and edge.link_type != link_type:
            continue
        return edge
    return None


def node_id_by_label(scenario: RemoteSensingScenario, label: str) -> int | None:
    for node_id, node in scenario.node_by_id.items():
        if node.label == label:
            return node_id
    return None


def single_packet_edge_delay_ms(
    edge: ScenarioEdge,
    link_config: LinkModelConfig | None = None,
) -> float:
    if edge.link_type == LINK_OBSERVATION:
        return float(edge.delay_ms)
    if edge.capacity_bps <= 0.0:
        return float("inf")
    config = link_config or LinkModelConfig()
    prop_delay_ms = float(edge.distance_km) / SPEED_OF_LIGHT_KM_PER_MS
    tx_delay_ms = float(config.packet_size_bits) / float(edge.capacity_bps) * 1000.0
    return prop_delay_ms + tx_delay_ms


def task_total_edge_energy_j(
    edge: ScenarioEdge,
    scenario_config: RemoteSensingScenarioConfig,
    link_config: LinkModelConfig | None = None,
) -> float:
    if edge.link_type == LINK_OBSERVATION:
        return float(edge.energy_cost)
    if edge.capacity_bps <= 0.0:
        return 0.0
    config = link_config or LinkModelConfig()
    duration_s = (
        float(scenario_config.packet_count_per_task)
        * float(config.packet_size_bits)
        / float(edge.capacity_bps)
    )
    return (float(config.tx_power_w) + float(config.rx_power_w)) * duration_s


def edge_loss(edge: ScenarioEdge) -> float:
    return max(float(edge.loss_risk), 0.0)


def path_delay_ms(edges: list[ScenarioEdge], link_config: LinkModelConfig) -> float:
    return float(sum(single_packet_edge_delay_ms(edge, link_config) for edge in edges))


def path_energy_j(
    edges: list[ScenarioEdge],
    scenario_config: RemoteSensingScenarioConfig,
    link_config: LinkModelConfig,
) -> float:
    return float(sum(task_total_edge_energy_j(edge, scenario_config, link_config) for edge in edges))


def path_loss(edges: list[ScenarioEdge]) -> float:
    return float(sum(edge_loss(edge) for edge in edges))


def resolve_route_edges(snapshot: ScenarioSnapshot, node_ids: list[int]) -> list[ScenarioEdge]:
    resolved: list[ScenarioEdge] = []
    for src_id, dst_id in zip(node_ids, node_ids[1:]):
        edge = find_edge(snapshot, src_id, dst_id)
        if edge is not None:
            resolved.append(edge)
    return resolved


def format_path(scenario: RemoteSensingScenario, node_ids: list[int]) -> str:
    labels: list[str] = []
    for node_id in node_ids:
        node = scenario.node_by_id.get(node_id)
        labels.append(node.label if node is not None else str(node_id))
    return " -> ".join(labels)


def count_cls_relay_hops(scenario: RemoteSensingScenario, node_ids: list[int]) -> int:
    relay_hops = 0
    for src_id, dst_id in zip(node_ids, node_ids[1:]):
        src = scenario.node_by_id.get(src_id)
        dst = scenario.node_by_id.get(dst_id)
        if src is not None and dst is not None and src.node_type == NODE_CLS and dst.node_type == NODE_CLS:
            relay_hops += 1
    return relay_hops


def build_route_metric_record(
    *,
    scenario: RemoteSensingScenario,
    snapshot: ScenarioSnapshot,
    front_edges: list[ScenarioEdge],
    cls_edges: list[ScenarioEdge],
) -> dict[str, float]:
    all_edges = list(front_edges) + list(cls_edges)
    return {
        "delay_ms_total": path_delay_ms(all_edges, scenario.link_config),
        "delay_ms_cls_route": path_delay_ms(cls_edges, scenario.link_config),
        "energy_j_total": path_energy_j(all_edges, scenario.config, scenario.link_config),
        "loss_total": path_loss(all_edges),
    }


def summarize_route_records(routes: list[dict[str, Any]]) -> dict[str, Any]:
    success_routes = [route for route in routes if bool(route.get("success"))]
    failed_routes = [route for route in routes if not bool(route.get("success"))]
    fail_reason_counts = Counter(
        str(route.get("fail_reason"))
        for route in failed_routes
        if route.get("fail_reason") not in {None, ""}
    )

    def avg(key: str, rows: list[dict[str, Any]]) -> float:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) not in {None, ""}
        ]
        return float(sum(values) / len(values)) if values else 0.0

    total = len(routes)
    return {
        "total_tasks": total,
        "success_count": len(success_routes),
        "failed_count": len(failed_routes),
        "success_rate": len(success_routes) / total if total else 0.0,
        "avg_delay_success_ms": avg("delay_ms_total", success_routes),
        "avg_delay_actual_all_ms": avg("delay_ms_total", routes),
        "avg_cls_delay_success_ms": avg("delay_ms_cls_route", success_routes),
        "avg_energy_success_j": avg("energy_j_total", success_routes),
        "avg_loss_success": avg("loss_total", success_routes),
        "avg_reward_all": avg("reward_total", routes),
        "fail_reason_counts": dict(sorted(fail_reason_counts.items())),
    }
