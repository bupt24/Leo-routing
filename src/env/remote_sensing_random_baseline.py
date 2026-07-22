"""Shared concurrent simulator for the Random01-04 routing baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from env.remote_sensing_scenario import (
    LINK_DOWNLINK,
    NODE_CLS,
    NODE_ES,
    RemoteSensingScenario,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioNode,
    ScenarioSnapshot,
)
from env.remote_sensing_task_core import (
    DownlinkQueueManager,
    MultiSourceTaskFactory,
    RemoteSensingTask,
    current_isl_edges,
    shortest_isl_path,
    task_edge_transmission_delay_ms,
)


RANDOM_VARIANTS = {"random01", "random02", "random03", "random04", "dijkstra"}
RANDOM_FAIL_REASONS = (
    "no_reachable_gs",
    "no_access_edge",
    "no_path_to_gs",
    "no_available_next_hop",
    "loop",
    "ttl_exceeded",
    "deadline_exceeded",
    "cls_queue_contention",
    "downlink_queue_overflow",
    "episode_horizon_exceeded",
)


@dataclass(frozen=True)
class RandomBaselineConfig:
    seed: int = 42
    ttl_margin: int = 3
    ttl_cap: int = 10
    max_attempts: int = 1
    drain_slots: int | None = None
    variant: str = "random01"


@dataclass(frozen=True)
class RetryRandomBaselineConfig(RandomBaselineConfig):
    max_attempts: int = 8
    variant: str = "random03"


@dataclass
class PlannedRoute:
    task: RemoteSensingTask
    node_ids: list[int]
    selected_cls: str
    exit_cls: str
    route_delay_ms: float
    energy_j: float
    loss: float
    attempt_count: int
    failed_attempts: int
    attempted_delay_ms: float
    attempted_energy_j: float


def _task_edge_energy(edge: ScenarioEdge, task: RemoteSensingTask, scenario: RemoteSensingScenario) -> float:
    if edge.capacity_bps <= 0.0:
        return 0.0
    duration_s = task.packet_count * scenario.link_config.packet_size_bits / float(edge.capacity_bps)
    return (scenario.link_config.tx_power_w + scenario.link_config.rx_power_w) * duration_s


def _future_exit_ids(task: RemoteSensingTask, future: list[ScenarioSnapshot]) -> set[int]:
    if task.destination_gs is None:
        return set()
    return {
        edge.src_id
        for snapshot in future
        for edge in snapshot.edges
        if edge.link_type == LINK_DOWNLINK
        and edge.dst_id == task.destination_gs.node_id
        and edge.available
        and edge.is_data_flow_allowed
    }


def _shortest_hops(snapshot: ScenarioSnapshot, start: int, exits: set[int], blocked: set[int]) -> int | None:
    if start in exits:
        return 0
    frontier = [(start, 0)]
    visited = set(blocked) | {start}
    for node_id, hops in frontier:
        for edge in current_isl_edges(snapshot, node_id):
            if edge.dst_id in exits:
                return hops + 1
            if edge.dst_id in visited:
                continue
            visited.add(edge.dst_id)
            frontier.append((edge.dst_id, hops + 1))
    return None


class RandomBaselineSimulator:
    def __init__(
        self,
        scenario_config: RemoteSensingScenarioConfig,
        num_time_slots: int,
        random_config: RandomBaselineConfig,
    ) -> None:
        if random_config.variant not in RANDOM_VARIANTS:
            raise ValueError(f"Unsupported random variant {random_config.variant!r}.")
        self.config = scenario_config
        self.num_time_slots = max(1, int(num_time_slots))
        self.random_config = random_config
        self.drain_slots = int(
            scenario_config.drain_slots
            if random_config.drain_slots is None
            else random_config.drain_slots
        )
        self.scenario = RemoteSensingScenario(scenario_config)
        self.factory = MultiSourceTaskFactory(
            scenario_config, self.scenario.link_config.packet_size_bits
        )
        self.downlink = DownlinkQueueManager(
            scenario_config, self.scenario.link_config.packet_size_bits
        )
        self.forecaster = RemoteSensingScenario(scenario_config)
        total = self.num_time_slots + self.drain_slots + scenario_config.gs_lookahead_slots
        self.snapshots = [self.forecaster.build_snapshot(slot) for slot in range(total)]
        self.pending: dict[str, PlannedRoute] = {}
        self.routes: list[dict[str, Any]] = []
        self.topology_edges: list[ScenarioEdge] = []
        self.queue_rows: list[dict[str, Any]] = []

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any], list[ScenarioEdge]]:
        total_slots = self.num_time_slots + self.drain_slots
        for slot in range(total_slots):
            snapshot = self._activate_snapshot(slot)
            injected: dict[tuple[int, int], float] = defaultdict(float)
            if slot < self.num_time_slots:
                future = self.snapshots[slot : slot + self.config.gs_lookahead_slots]
                tasks = self.factory.build_tasks(
                    snapshot,
                    future,
                    episode=0,
                    seed=self.random_config.seed,
                    queued_packets_by_gs=self.downlink.queued_packets_by_gs(),
                )
                for task_index, task in enumerate(
                    sorted(tasks, key=lambda item: (item.priority, item.deadline_s, item.task_id))
                ):
                    rng = random.Random(
                        self.random_config.seed + slot * 100_003 + task_index * 1_009
                    )
                    planned, fail_reason, attempt_stats = self._plan_with_retries(
                        task, snapshot, future, injected, tasks, rng
                    )
                    if planned is None:
                        self.routes.append(
                            self._failed_record(task, fail_reason, snapshot, attempt_stats)
                        )
                        continue
                    if not self._reserve_cls_path(planned, injected):
                        self.routes.append(
                            self._failed_record(
                                task,
                                "cls_queue_contention",
                                snapshot,
                                attempt_stats,
                                planned=planned,
                            )
                        )
                        continue
                    exit_cls_id = planned.node_ids[-1]
                    if not self.downlink.admit(
                        task,
                        exit_cls_id,
                        slot,
                        upstream_delay_ms=planned.route_delay_ms,
                    ):
                        self.routes.append(
                            self._failed_record(
                                task,
                                "downlink_queue_overflow",
                                snapshot,
                                attempt_stats,
                                planned=planned,
                            )
                        )
                        continue
                    self.pending[task.task_id] = planned
            self._apply_injected(injected)
            self.queue_rows.extend(
                self.downlink.queue_rows(snapshot.time_slot, self.scenario.node_by_id)
            )
            self._handle_delivery_events(self.downlink.service(snapshot))
            self.scenario._service_cls_queues(snapshot)

        final_time_s = total_slots * float(self.config.time_slot_seconds)
        self._handle_delivery_events(self.downlink.fail_all(final_time_s))
        self.routes.sort(key=lambda route: (route["time_slot"], route["task_id"]))
        metrics = build_random_metrics_summary(self.routes, self.random_config)
        metrics["_downlink_queue_rows"] = self.queue_rows
        queued_values = [float(row.get("queued_packets", 0.0)) for row in self.queue_rows]
        metrics["avg_downlink_queue_packets"] = (
            sum(queued_values) / len(queued_values) if queued_values else 0.0
        )
        metrics["peak_downlink_queue_packets"] = max(queued_values, default=0.0)
        return self.routes, metrics, self.topology_edges

    def _activate_snapshot(self, slot: int) -> ScenarioSnapshot:
        snapshot = self.snapshots[slot]
        self.scenario.node_by_id = {node.node_id: node for node in snapshot.nodes}
        self.scenario.cls_node_ids = [
            node.node_id for node in snapshot.nodes if node.node_type == NODE_CLS
        ]
        self.scenario._refresh_cls_queue_metrics(snapshot.edges)
        self.topology_edges.extend(snapshot.edges)
        return snapshot

    def _plan_with_retries(
        self,
        task: RemoteSensingTask,
        snapshot: ScenarioSnapshot,
        future: list[ScenarioSnapshot],
        injected: dict[tuple[int, int], float],
        all_tasks: list[RemoteSensingTask],
        rng: random.Random,
    ) -> tuple[PlannedRoute | None, str, dict[str, Any]]:
        variant = self.random_config.variant
        attempts = max(1, self.random_config.max_attempts if variant in {"random03", "random04"} else 1)
        attempted_delay = 0.0
        attempted_energy = 0.0
        reasons: list[str] = []
        final_plan: PlannedRoute | None = None
        attempts_made = 0
        for attempt in range(1, attempts + 1):
            attempts_made = attempt
            plan, reason, attempt_delay, attempt_energy = self._plan_once(
                task, snapshot, future, injected, all_tasks, rng
            )
            attempted_delay += attempt_delay
            attempted_energy += attempt_energy
            if plan is not None:
                if attempted_delay > task.deadline_s * 1000.0:
                    reasons.append("deadline_exceeded")
                    break
                plan.attempt_count = attempt
                plan.failed_attempts = attempt - 1
                plan.attempted_delay_ms = attempted_delay
                plan.attempted_energy_j = attempted_energy
                final_plan = plan
                break
            reasons.append(reason)
            if attempted_delay > task.deadline_s * 1000.0:
                reasons.append("deadline_exceeded")
                break
        stats = {
            "attempt_count": final_plan.attempt_count if final_plan is not None else attempts_made,
            "failed_attempts": final_plan.failed_attempts if final_plan is not None else attempts_made,
            "attempted_delay_ms": attempted_delay,
            "attempted_energy_j": attempted_energy,
            "attempt_fail_reasons": "|".join(reasons),
        }
        return final_plan, reasons[-1] if reasons else "no_path_to_gs", stats

    def _plan_once(
        self,
        task: RemoteSensingTask,
        snapshot: ScenarioSnapshot,
        future: list[ScenarioSnapshot],
        injected: dict[tuple[int, int], float],
        all_tasks: list[RemoteSensingTask],
        rng: random.Random,
    ) -> tuple[PlannedRoute | None, str, float, float]:
        if task.destination_gs is None:
            return (
                None,
                "no_reachable_gs",
                float(task.observation_edge.delay_ms),
                float(task.observation_edge.energy_cost),
            )
        access_edges = list(task.candidate_access_edges)
        if not access_edges:
            return (
                None,
                "no_access_edge",
                float(task.observation_edge.delay_ms),
                float(task.observation_edge.energy_cost),
            )
        exits = _future_exit_ids(task, future)
        if not exits:
            return (
                None,
                "no_path_to_gs",
                float(task.observation_edge.delay_ms),
                float(task.observation_edge.energy_cost),
            )
        if self.random_config.variant == "dijkstra":
            joint_candidates: list[tuple[float, str, ScenarioEdge, list[int], float]] = []
            for candidate_access in access_edges:
                candidate_path, _path_weight = shortest_isl_path(
                    snapshot, candidate_access.dst_id, exits
                )
                if not candidate_path:
                    continue
                candidate_edges = [task.observation_edge, candidate_access]
                candidate_edges.extend(self._resolve_cls_edges(snapshot, candidate_path))
                candidate_delay, candidate_energy = self._route_totals(task, candidate_edges)
                joint_candidates.append(
                    (
                        candidate_delay,
                        candidate_access.dst,
                        candidate_access,
                        candidate_path,
                        candidate_energy,
                    )
                )
            if not joint_candidates:
                return (
                    None,
                    "no_path_to_gs",
                    float(task.observation_edge.delay_ms),
                    float(task.observation_edge.energy_cost),
                )
            delay, _label, access, cls_path, energy = min(
                joint_candidates, key=lambda item: (item[0], item[1])
            )
        elif self.random_config.variant == "random04":
            access = min(
                access_edges,
                key=lambda edge: self._cooperative_access_score(
                    task, edge, snapshot, future, injected, all_tasks
                ),
            )
        else:
            access = rng.choice(access_edges)
        if self.random_config.variant != "dijkstra":
            cls_path, reason = self._random_cls_path(
                snapshot,
                access.dst_id,
                exits,
                rng,
                informed=self.random_config.variant == "random01",
            )
            partial_edges = [task.observation_edge, access]
            partial_edges.extend(self._resolve_cls_edges(snapshot, cls_path))
            delay, energy = self._route_totals(task, partial_edges)
            if reason:
                return None, reason, delay, energy
        node_ids = [task.target.node_id, task.source_rls.node_id] + cls_path
        selected_cls = self.scenario.node_by_id[access.dst_id].label
        exit_cls = self.scenario.node_by_id[cls_path[-1]].label
        edges = [task.observation_edge, access] + self._resolve_cls_edges(snapshot, cls_path)
        delay, energy = self._route_totals(task, edges)
        loss = sum(float(edge.loss_risk) for edge in edges)
        if delay > task.deadline_s * 1000.0:
            return None, "deadline_exceeded", delay, energy
        return (
            PlannedRoute(
                task=task,
                node_ids=node_ids,
                selected_cls=selected_cls,
                exit_cls=exit_cls,
                route_delay_ms=delay,
                energy_j=energy,
                loss=loss,
                attempt_count=1,
                failed_attempts=0,
                attempted_delay_ms=delay,
                attempted_energy_j=energy,
            ),
            "",
            delay,
            energy,
        )

    def _route_totals(
        self,
        task: RemoteSensingTask,
        edges: list[ScenarioEdge],
    ) -> tuple[float, float]:
        delay = sum(
            float(edge.delay_ms)
            if edge is task.observation_edge
            else task_edge_transmission_delay_ms(
                edge, task, self.scenario.link_config.packet_size_bits
            )
            for edge in edges
        )
        energy = float(task.observation_edge.energy_cost)
        energy += sum(
            _task_edge_energy(edge, task, self.scenario)
            for edge in edges
            if edge is not task.observation_edge
        )
        return delay, energy

    def _random_cls_path(
        self,
        snapshot: ScenarioSnapshot,
        start_cls_id: int,
        exits: set[int],
        rng: random.Random,
        *,
        informed: bool,
    ) -> tuple[list[int], str]:
        if informed:
            shortest = _shortest_hops(snapshot, start_cls_id, exits, set())
            if shortest is None:
                return [start_cls_id], "no_path_to_gs"
            ttl = min(self.random_config.ttl_cap, shortest + self.random_config.ttl_margin)
        else:
            ttl = self.random_config.ttl_cap
        current = start_cls_id
        path = [current]
        visited = {current}
        for hop in range(ttl + 1):
            if hop >= ttl:
                return (path, "") if current in exits else (path, "ttl_exceeded")
            choices: list[int | None] = [None] if current in exits else []
            remaining = ttl - hop - 1
            for edge in current_isl_edges(snapshot, current):
                if edge.dst_id in visited:
                    continue
                if informed:
                    reachable = _shortest_hops(snapshot, edge.dst_id, exits, visited)
                    if reachable is None or reachable > remaining:
                        continue
                choices.append(edge.dst_id)
            if not choices:
                return path, "no_available_next_hop"
            selected = rng.choice(choices)
            if selected is None:
                return path, ""
            path.append(selected)
            visited.add(selected)
            current = selected
        return path, "ttl_exceeded"

    def _cooperative_access_score(
        self,
        task: RemoteSensingTask,
        edge: ScenarioEdge,
        snapshot: ScenarioSnapshot,
        future: list[ScenarioSnapshot],
        injected: dict[tuple[int, int], float],
        all_tasks: list[RemoteSensingTask],
    ) -> float:
        destination = self.scenario.node_by_id[edge.dst_id]
        class_idx = task.traffic_class
        capacity = float(self.config.queue_capacities[class_idx])
        queue = float(self.scenario.queue_lengths[destination.local_id, class_idx].item())
        already_injected = injected.get((edge.dst_id, class_idx), 0.0)
        other_rls_pressure = sum(
            other.packet_count
            for other in all_tasks
            if other.task_id != task.task_id
            and other.source_rls.node_id != task.source_rls.node_id
            and any(candidate.dst_id == edge.dst_id for candidate in other.candidate_access_edges)
        )
        queue_ratio = min((queue + already_injected) / max(capacity, 1.0), 1.0)
        residual_ratio = max(capacity - queue - already_injected, 0.0) / max(capacity, 1.0)
        injected_ratio = min((already_injected + other_rls_pressure) / max(capacity, 1.0), 1.0)
        exits = _future_exit_ids(task, future)
        path, path_delay = shortest_isl_path(snapshot, edge.dst_id, exits)
        downstream = 1.0 if not path else min(path_delay / max(self.config.normalizer_delay_ms, 1.0), 1.0)
        distance = min(edge.distance_km / max(self.config.max_cross_layer_distance_km, 1.0), 1.0)
        return (
            0.25 * queue_ratio
            + 0.25 * (1.0 - residual_ratio)
            + 0.25 * injected_ratio
            + 0.20 * downstream
            + 0.05 * distance
        )

    def _reserve_cls_path(
        self,
        planned: PlannedRoute,
        injected: dict[tuple[int, int], float],
    ) -> bool:
        class_idx = planned.task.traffic_class
        capacity = float(self.config.queue_capacities[class_idx])
        cls_ids = [
            node_id
            for node_id in planned.node_ids
            if self.scenario.node_by_id[node_id].node_type == NODE_CLS
        ]
        for cls_id in cls_ids:
            node = self.scenario.node_by_id[cls_id]
            base = float(self.scenario.queue_lengths[node.local_id, class_idx].item())
            if base + injected[(cls_id, class_idx)] + planned.task.packet_count > capacity + 1e-9:
                return False
        for cls_id in cls_ids:
            injected[(cls_id, class_idx)] += planned.task.packet_count
        return True

    def _apply_injected(self, injected: dict[tuple[int, int], float]) -> None:
        for (cls_id, class_idx), packets in injected.items():
            node = self.scenario.node_by_id[cls_id]
            capacity = float(self.config.queue_capacities[class_idx])
            self.scenario.queue_lengths[node.local_id, class_idx] = min(
                float(self.scenario.queue_lengths[node.local_id, class_idx].item()) + packets,
                capacity,
            )

    @staticmethod
    def _resolve_cls_edges(snapshot: ScenarioSnapshot, path: list[int]) -> list[ScenarioEdge]:
        by_pair = {
            (edge.src_id, edge.dst_id): edge
            for edge in snapshot.edges
            if edge.src_type == NODE_CLS and edge.dst_type == NODE_CLS
        }
        return [by_pair[(src, dst)] for src, dst in zip(path, path[1:])]

    def _handle_delivery_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            planned = self.pending.pop(str(event["task_id"]), None)
            if planned is None:
                continue
            self.routes.append(self._completed_record(planned, event))

    def _completed_record(self, planned: PlannedRoute, event: dict[str, Any]) -> dict[str, Any]:
        task = planned.task
        completion_time_s = float(event["completion_time_s"])
        elapsed_ms = max(completion_time_s - task.generation_time_s, 0.0) * 1000.0
        delay_ms = elapsed_ms + planned.route_delay_ms
        success = bool(event["success"])
        return self._record_base(
            task,
            planned,
            success=success,
            fail_reason=event.get("fail_reason"),
            delay_ms=delay_ms,
            completion_time_s=completion_time_s,
            delivered_packets=float(event.get("delivered_packets", 0.0)),
        )

    def _failed_record(
        self,
        task: RemoteSensingTask,
        fail_reason: str,
        snapshot: ScenarioSnapshot,
        attempt_stats: dict[str, Any],
        *,
        planned: PlannedRoute | None = None,
    ) -> dict[str, Any]:
        if planned is None:
            planned = PlannedRoute(
                task=task,
                node_ids=[task.target.node_id, task.source_rls.node_id],
                selected_cls="",
                exit_cls="",
                route_delay_ms=float(attempt_stats.get("attempted_delay_ms", task.observation_edge.delay_ms)),
                energy_j=float(attempt_stats.get("attempted_energy_j", task.observation_edge.energy_cost)),
                loss=0.0,
                attempt_count=int(attempt_stats.get("attempt_count", 1)),
                failed_attempts=int(attempt_stats.get("failed_attempts", 1)),
                attempted_delay_ms=float(attempt_stats.get("attempted_delay_ms", 0.0)),
                attempted_energy_j=float(attempt_stats.get("attempted_energy_j", 0.0)),
            )
        return self._record_base(
            task,
            planned,
            success=False,
            fail_reason=fail_reason,
            delay_ms=planned.attempted_delay_ms,
            completion_time_s=float(snapshot.time_sec),
            delivered_packets=0.0,
            attempt_fail_reasons=str(attempt_stats.get("attempt_fail_reasons", "")),
        )

    def _record_base(
        self,
        task: RemoteSensingTask,
        planned: PlannedRoute,
        *,
        success: bool,
        fail_reason: str | None,
        delay_ms: float,
        completion_time_s: float,
        delivered_packets: float,
        attempt_fail_reasons: str = "",
    ) -> dict[str, Any]:
        gs_label = task.destination_gs.label if task.destination_gs is not None else ""
        node_ids = list(planned.node_ids)
        if success and task.destination_gs is not None:
            node_ids.append(task.destination_gs.node_id)
        return {
            "time_slot": task.generation_slot,
            "generation_slot": task.generation_slot,
            "completion_slot": int(completion_time_s // max(self.config.time_slot_seconds, 1e-9)),
            "completion_time_s": completion_time_s,
            "task_id": task.task_id,
            "target_name": task.target.label,
            "profile_name": task.profile_name,
            "data_size_mb": task.data_size_mb,
            "packet_count": task.packet_count,
            "traffic_class": task.traffic_class,
            "priority": task.priority,
            "deadline_s": task.deadline_s,
            "rls_node": task.source_rls.label,
            "rls_reused": task.rls_reused,
            "candidate_rls_count": task.candidate_rls_count,
            "rls_reuse_count": task.rls_reuse_count,
            "candidate_cls_count": len(task.candidate_access_edges),
            "selected_cls": planned.selected_cls,
            "selected_es": gs_label,
            "destination_gs": gs_label,
            "exit_cls": planned.exit_cls,
            "success": success,
            "deadline_met": (
                success
                and completion_time_s + planned.route_delay_ms / 1000.0
                <= task.absolute_deadline_s + 1e-9
            ),
            "access_success": bool(planned.selected_cls),
            "route_success": success,
            "fail_reason": fail_reason,
            "path": " -> ".join(self.scenario.node_by_id[node_id].label for node_id in node_ids),
            "hop_count": max(len(node_ids) - 1, 0),
            "cls_relay_hops": max(sum(1 for node_id in planned.node_ids if self.scenario.node_by_id[node_id].node_type == NODE_CLS) - 1, 0),
            "total_delay_ms": delay_ms,
            "delay_ms_total": delay_ms,
            "delay_with_timeout_ms": delay_ms if success else task.deadline_s * 1000.0,
            "delay_ms_cls_route": max(planned.route_delay_ms - task.observation_edge.delay_ms, 0.0),
            "total_energy": planned.energy_j,
            "energy_j_total": planned.energy_j,
            "loss_total": planned.loss,
            "total_loss_risk": planned.loss,
            "delivered_packets": delivered_packets,
            "attempt_count": planned.attempt_count,
            "failed_attempts": planned.failed_attempts,
            "attempt_fail_reasons": attempt_fail_reasons,
            "attempted_delay_ms": planned.attempted_delay_ms,
            "attempted_energy": planned.attempted_energy_j,
            "max_attempts": self.random_config.max_attempts,
            "gs_score": task.gs_score,
        }


def run_random_baseline(
    scenario_config: RemoteSensingScenarioConfig,
    num_time_slots: int,
    random_config: RandomBaselineConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ScenarioEdge]]:
    random_config = random_config or RandomBaselineConfig()
    return RandomBaselineSimulator(scenario_config, num_time_slots, random_config).run()


def build_random_metrics_summary(
    routes: list[dict[str, Any]],
    random_config: RandomBaselineConfig | None = None,
) -> dict[str, Any]:
    random_config = random_config or RandomBaselineConfig()
    successes = [route for route in routes if route.get("success")]
    failures = [route for route in routes if not route.get("success")]

    def average(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0

    reasons = Counter(str(route.get("fail_reason")) for route in failures)
    for reason in RANDOM_FAIL_REASONS:
        reasons.setdefault(reason, 0)
    total = len(routes)
    delivered_mb = sum(float(route.get("data_size_mb", 0.0)) for route in successes)
    elapsed_s = max((float(route.get("completion_time_s", 0.0)) for route in routes), default=0.0)
    gs_counts = Counter(
        str(route.get("destination_gs"))
        for route in routes
        if route.get("destination_gs")
    )
    profile_counts = Counter(str(route.get("profile_name")) for route in routes)
    return {
        "variant": random_config.variant,
        "task_count": total,
        "total_tasks": total,
        "success_count": len(successes),
        "deadline_met_count": len(successes),
        "route_failed_count": len(failures),
        "failed_count": len(failures),
        "success_rate": len(successes) / total if total else 0.0,
        "deadline_meeting_rate": len(successes) / total if total else 0.0,
        "throughput_mbps": delivered_mb * 8.0 / elapsed_s if elapsed_s > 0.0 else 0.0,
        "avg_delay_success_ms": average("total_delay_ms", successes),
        "avg_delay_with_timeout_ms": average("delay_with_timeout_ms", routes),
        "avg_cls_delay_success_ms": average("delay_ms_cls_route", successes),
        "avg_energy_success_j": average("energy_j_total", successes),
        "avg_attempt_count_success": average("attempt_count", successes),
        "avg_attempted_delay_ms": average("attempted_delay_ms", routes),
        "avg_attempted_energy_j": average("attempted_energy", routes),
        "max_attempts": random_config.max_attempts,
        "rls_reuse_count": sum(1 for route in routes if route.get("rls_reused")),
        "gs_assignment_counts": dict(sorted(gs_counts.items())),
        "task_profile_counts": dict(sorted(profile_counts.items())),
        "fail_reason_counts": dict(sorted(reasons.items())),
    }


def write_random_outputs(
    output_dir: str | Path,
    routes: list[dict[str, Any]],
    metrics: dict[str, Any],
    topology_edges: list[ScenarioEdge],
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    route_fields = list(dict.fromkeys(key for route in routes for key in route)) if routes else []
    for filename in ("tasks.csv", "random_routes.csv"):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            if route_fields:
                writer = csv.DictWriter(handle, fieldnames=route_fields)
                writer.writeheader()
                writer.writerows({key: route.get(key, "") for key in route_fields} for route in routes)
    edge_fields = list(topology_edges[0].csv_row()) if topology_edges else []
    for filename in ("topology.csv", "topology_edges.csv"):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            if edge_fields:
                writer = csv.DictWriter(handle, fieldnames=edge_fields)
                writer.writeheader()
                writer.writerows(edge.csv_row() for edge in topology_edges)
    queue_rows = list(metrics.get("_downlink_queue_rows", []))
    queue_fields = list(dict.fromkeys(key for row in queue_rows for key in row)) if queue_rows else []
    for filename in ("slot_queues.csv", "downlink_queues.csv"):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            if queue_fields:
                writer = csv.DictWriter(handle, fieldnames=queue_fields)
                writer.writeheader()
                writer.writerows({key: row.get(key, "") for key in queue_fields} for row in queue_rows)
    json_metrics = {key: value for key, value in metrics.items() if not key.startswith("_")}
    payload = json.dumps(json_metrics, indent=2)
    for filename in ("summary.json", "random_metrics_summary.json"):
        (out_dir / filename).write_text(payload, encoding="utf-8")


__all__ = [
    "RANDOM_FAIL_REASONS",
    "RANDOM_VARIANTS",
    "RandomBaselineConfig",
    "RandomBaselineSimulator",
    "RetryRandomBaselineConfig",
    "build_random_metrics_summary",
    "run_random_baseline",
    "write_random_outputs",
]
