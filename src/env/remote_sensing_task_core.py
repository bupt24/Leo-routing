"""Shared multi-source task generation and CLS downlink queue mechanics."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Any, Iterable

from env.remote_sensing_scenario import (
    LINK_ACCESS,
    LINK_DOWNLINK,
    LINK_ISL,
    LINK_OBSERVATION,
    NODE_CLS,
    NODE_ES,
    NODE_TARGET,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioNode,
    ScenarioSnapshot,
    TaskProfile,
)


@dataclass
class RemoteSensingTask:
    task_id: str
    episode: int
    generation_slot: int
    generation_time_s: float
    target: ScenarioNode
    source_rls: ScenarioNode
    observation_edge: ScenarioEdge
    candidate_access_edges: list[ScenarioEdge]
    profile_name: str
    data_size_mb: float
    packet_count: float
    traffic_class: int
    priority: int
    deadline_s: float
    destination_gs: ScenarioNode | None = None
    rls_reused: bool = False
    candidate_rls_count: int = 0
    rls_reuse_count: int = 0
    gs_score: float = 0.0
    gs_score_components: dict[str, float] = field(default_factory=dict)

    @property
    def absolute_deadline_s(self) -> float:
        return self.generation_time_s + self.deadline_s


@dataclass(frozen=True)
class RoutingCandidate:
    kind: str
    src_id: int
    dst_id: int
    label: str
    edge: ScenarioEdge | None = None
    forecast_wait_slots: int = 0
    forecast_visible_slots: int = 0
    forecast_service_packets: float = 0.0


@dataclass
class DownlinkQueueItem:
    task: RemoteSensingTask
    exit_cls_id: int
    destination_gs_id: int
    remaining_packets: float
    admitted_slot: int
    upstream_delay_ms: float = 0.0


def task_edge_transmission_delay_ms(edge: ScenarioEdge, task: RemoteSensingTask, packet_size_bits: float) -> float:
    if edge.capacity_bps <= 0.0:
        return float(edge.delay_ms)
    transmission_ms = task.packet_count * float(packet_size_bits) / float(edge.capacity_bps) * 1000.0
    propagation_ms = max(float(edge.delay_ms), 0.0)
    return propagation_ms + transmission_ms


def current_isl_edges(snapshot: ScenarioSnapshot, src_id: int) -> list[ScenarioEdge]:
    edges = [
        edge
        for edge in snapshot.edges
        if edge.src_id == src_id
        and edge.src_type == NODE_CLS
        and edge.dst_type == NODE_CLS
        and edge.link_type == LINK_ISL
        and edge.available
        and edge.is_data_flow_allowed
    ]
    edges.sort(key=lambda edge: (edge.dst, edge.dst_id))
    return edges


def shortest_isl_path(
    snapshot: ScenarioSnapshot,
    start_cls_id: int,
    exit_cls_ids: set[int],
    *,
    blocked: set[int] | None = None,
) -> tuple[list[int], float]:
    if start_cls_id in exit_cls_ids:
        return [start_cls_id], 0.0
    blocked = set(blocked or ())
    distances: dict[int, float] = {start_cls_id: 0.0}
    previous: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, start_cls_id)]
    reached: int | None = None
    while heap:
        distance, node_id = heapq.heappop(heap)
        if distance > distances.get(node_id, float("inf")):
            continue
        if node_id in exit_cls_ids:
            reached = node_id
            break
        for edge in current_isl_edges(snapshot, node_id):
            if edge.dst_id in blocked and edge.dst_id not in exit_cls_ids:
                continue
            candidate = distance + max(float(edge.delay_ms), 0.0)
            if candidate < distances.get(edge.dst_id, float("inf")):
                distances[edge.dst_id] = candidate
                previous[edge.dst_id] = node_id
                heapq.heappush(heap, (candidate, edge.dst_id))
    if reached is None:
        return [], float("inf")
    path = [reached]
    while path[-1] != start_cls_id:
        path.append(previous[path[-1]])
    path.reverse()
    return path, distances[reached]


class MultiSourceTaskFactory:
    """Generate one task per observed target and assign one fixed destination GS."""

    def __init__(self, config: RemoteSensingScenarioConfig, packet_size_bits: float) -> None:
        self.config = config
        self.packet_size_bits = float(packet_size_bits)

    def build_tasks(
        self,
        snapshot: ScenarioSnapshot,
        future_snapshots: list[ScenarioSnapshot],
        *,
        episode: int,
        seed: int,
        queued_packets_by_gs: dict[int, float] | None = None,
    ) -> list[RemoteSensingTask]:
        queued_packets_by_gs = dict(queued_packets_by_gs or {})
        observation_by_target: dict[int, list[ScenarioEdge]] = defaultdict(list)
        for edge in snapshot.edges:
            if (
                edge.link_type == LINK_OBSERVATION
                and edge.src_type == NODE_TARGET
                and edge.available
                and edge.is_data_flow_allowed
            ):
                observation_by_target[edge.src_id].append(edge)
        for edges in observation_by_target.values():
            edges.sort(
                key=lambda edge: (
                    -float(edge.observation_quality),
                    float(edge.off_nadir_angle_deg),
                    float(edge.slant_range_km),
                    edge.dst,
                )
            )

        targets = sorted(
            (node for node in snapshot.nodes if node.node_type == NODE_TARGET),
            key=lambda node: (len(observation_by_target.get(node.node_id, [])), node.node_id),
        )[: self.config.max_concurrent_tasks]
        selected_observations: dict[int, tuple[ScenarioEdge, bool]] = {}
        used_rls: set[int] = set()
        deferred: list[ScenarioNode] = []
        for target in targets:
            candidate_limit = min(3, max(int(self.config.beam.max_rls_per_target), 0))
            candidates = observation_by_target.get(target.node_id, [])[:candidate_limit]
            selected = next((edge for edge in candidates if edge.dst_id not in used_rls), None)
            if selected is None:
                deferred.append(target)
                continue
            selected_observations[target.node_id] = (selected, False)
            used_rls.add(selected.dst_id)
        for target in deferred:
            candidate_limit = min(3, max(int(self.config.beam.max_rls_per_target), 0))
            candidates = observation_by_target.get(target.node_id, [])[:candidate_limit]
            if candidates:
                selected_observations[target.node_id] = (candidates[0], candidates[0].dst_id in used_rls)
                used_rls.add(candidates[0].dst_id)

        assignment_counts = Counter(edge.dst_id for edge, _reused in selected_observations.values())
        node_by_id = {node.node_id: node for node in snapshot.nodes}
        drafts: list[RemoteSensingTask] = []
        for target in sorted(targets, key=lambda node: node.node_id):
            selected = selected_observations.get(target.node_id)
            if selected is None:
                continue
            observation_edge, reused = selected
            source_rls = node_by_id[observation_edge.dst_id]
            access_edges = [
                edge
                for edge in snapshot.edges
                if edge.src_id == source_rls.node_id
                and edge.link_type == LINK_ACCESS
                and edge.available
                and edge.is_data_flow_allowed
            ]
            access_edges.sort(key=lambda edge: (edge.delay_ms, edge.distance_km, edge.dst))
            if self.config.max_access_cls_per_rls > 0:
                access_edges = access_edges[: self.config.max_access_cls_per_rls]
            rng = random.Random(
                int(seed) * 1_000_003
                + int(snapshot.time_slot) * 1_009
                + int(target.local_id)
            )
            profile = self._sample_profile(rng)
            data_size_mb = rng.uniform(profile.min_data_mb, profile.max_data_mb)
            packet_count = math.ceil(data_size_mb * 8_000_000.0 / self.packet_size_bits)
            drafts.append(
                RemoteSensingTask(
                    task_id=f"task_{snapshot.time_slot}_{target.local_id}",
                    episode=int(episode),
                    generation_slot=int(snapshot.time_slot),
                    generation_time_s=float(snapshot.time_sec),
                    target=target,
                    source_rls=source_rls,
                    observation_edge=observation_edge,
                    candidate_access_edges=access_edges,
                    profile_name=profile.name,
                    data_size_mb=data_size_mb,
                    packet_count=float(packet_count),
                    traffic_class=profile.traffic_class,
                    priority=profile.priority,
                    deadline_s=profile.deadline_s,
                    rls_reused=reused,
                    candidate_rls_count=min(
                        len(observation_by_target.get(target.node_id, [])),
                        min(3, max(int(self.config.beam.max_rls_per_target), 0)),
                    ),
                    rls_reuse_count=max(assignment_counts[observation_edge.dst_id] - 1, 0),
                )
            )

        reservations: dict[int, float] = defaultdict(float)
        for task in sorted(drafts, key=lambda item: (item.priority, item.deadline_s, item.task_id)):
            gs, score, components = self._select_gs(
                task,
                snapshot,
                future_snapshots,
                queued_packets_by_gs,
                reservations,
            )
            task.destination_gs = gs
            task.gs_score = score
            task.gs_score_components = components
            if gs is not None:
                reservations[gs.node_id] += task.packet_count
        return sorted(drafts, key=lambda task: task.task_id)

    def _sample_profile(self, rng: random.Random) -> TaskProfile:
        draw = rng.random()
        cumulative = 0.0
        for profile in self.config.task_profiles:
            cumulative += profile.probability
            if draw <= cumulative:
                return profile
        return self.config.task_profiles[-1]

    def _select_gs(
        self,
        task: RemoteSensingTask,
        snapshot: ScenarioSnapshot,
        future_snapshots: list[ScenarioSnapshot],
        queued: dict[int, float],
        reservations: dict[int, float],
    ) -> tuple[ScenarioNode | None, float, dict[str, float]]:
        gs_nodes = sorted(
            (node for node in snapshot.nodes if node.node_type == NODE_ES),
            key=lambda node: node.label,
        )
        scored: list[tuple[float, str, ScenarioNode, dict[str, float]]] = []
        horizon = max(len(future_snapshots), 1)
        for gs in gs_nodes:
            contacts = self._contacts_for_gs(gs.node_id, future_snapshots)
            if not contacts:
                continue
            exit_ids = {edge.src_id for _, edge in contacts}
            reachable = 0
            path_delays: list[float] = []
            for access in task.candidate_access_edges:
                path, delay = shortest_isl_path(snapshot, access.dst_id, exit_ids)
                if path:
                    reachable += 1
                    path_delays.append(float(access.delay_ms) + delay)
            if reachable == 0:
                continue
            first_slot = min(offset for offset, _edge in contacts)
            visible_slots = len({offset for offset, _edge in contacts})
            service_packets = sum(
                float(edge.capacity_bps) * float(self.config.time_slot_seconds) / self.packet_size_bits
                for _offset, edge in contacts
            )
            delay_norm = min(min(path_delays) / max(self.config.normalizer_delay_ms, 1.0), 1.0)
            reachability_penalty = 1.0 - reachable / max(len(task.candidate_access_edges), 1)
            window_penalty = 0.5 * min(first_slot / horizon, 1.0) + 0.5 * (1.0 - visible_slots / horizon)
            assigned = queued.get(gs.node_id, 0.0) + reservations.get(gs.node_id, 0.0) + task.packet_count
            load_penalty = min(assigned / max(service_packets, 1.0), 1.0)
            weights = self.config.gs_selection_weights
            score = (
                weights.path_delay * delay_norm
                + weights.exit_reachability * reachability_penalty
                + weights.visibility_window * window_penalty
                + weights.assigned_load * load_penalty
            )
            components = {
                "path_delay": delay_norm,
                "exit_reachability": reachability_penalty,
                "visibility_window": window_penalty,
                "assigned_load": load_penalty,
            }
            scored.append((score, gs.label, gs, components))
        if not scored:
            return None, float("inf"), {}
        score, _label, gs, components = min(scored, key=lambda item: (item[0], item[1]))
        return gs, float(score), components

    @staticmethod
    def _contacts_for_gs(gs_id: int, snapshots: Iterable[ScenarioSnapshot]) -> list[tuple[int, ScenarioEdge]]:
        contacts: list[tuple[int, ScenarioEdge]] = []
        for offset, snapshot in enumerate(snapshots):
            contacts.extend(
                (offset, edge)
                for edge in snapshot.edges
                if edge.link_type == LINK_DOWNLINK
                and edge.dst_id == gs_id
                and edge.available
                and edge.is_data_flow_allowed
            )
        return contacts

    def access_can_reach_gs(
        self,
        access_cls_id: int,
        gs_id: int,
        snapshot: ScenarioSnapshot,
        future_snapshots: list[ScenarioSnapshot],
    ) -> bool:
        exit_ids = {edge.src_id for _offset, edge in self._contacts_for_gs(gs_id, future_snapshots)}
        path, _delay = shortest_isl_path(snapshot, access_cls_id, exit_ids)
        return bool(path)

    def exit_forecast(
        self,
        cls_id: int,
        gs_id: int,
        future_snapshots: list[ScenarioSnapshot],
    ) -> tuple[int, int, float] | None:
        contacts = [
            (offset, edge)
            for offset, edge in self._contacts_for_gs(gs_id, future_snapshots)
            if edge.src_id == cls_id
        ]
        if not contacts:
            return None
        wait_slots = min(offset for offset, _edge in contacts)
        visible_slots = len({offset for offset, _edge in contacts})
        service_packets = sum(
            float(edge.capacity_bps) * float(self.config.time_slot_seconds) / self.packet_size_bits
            for _offset, edge in contacts
        )
        return wait_slots, visible_slots, service_packets


class DownlinkQueueManager:
    """Destination-aware queues located at exit CLS nodes, never inside a GS."""

    def __init__(self, config: RemoteSensingScenarioConfig, packet_size_bits: float) -> None:
        self.config = config
        self.packet_size_bits = float(packet_size_bits)
        self.queues: dict[tuple[int, int, int], deque[DownlinkQueueItem]] = defaultdict(deque)

    def queue_length(self, cls_id: int, gs_id: int, traffic_class: int) -> float:
        return sum(item.remaining_packets for item in self.queues[(cls_id, gs_id, traffic_class)])

    def can_admit(self, task: RemoteSensingTask, exit_cls_id: int) -> bool:
        if task.destination_gs is None:
            return False
        queued = self.queue_length(exit_cls_id, task.destination_gs.node_id, task.traffic_class)
        capacity = float(self.config.downlink_queue_capacities[task.traffic_class])
        return queued + task.packet_count <= capacity + 1e-9

    def admit(
        self,
        task: RemoteSensingTask,
        exit_cls_id: int,
        time_slot: int,
        *,
        upstream_delay_ms: float = 0.0,
    ) -> bool:
        if not self.can_admit(task, exit_cls_id) or task.destination_gs is None:
            return False
        key = (exit_cls_id, task.destination_gs.node_id, task.traffic_class)
        self.queues[key].append(
            DownlinkQueueItem(
                task=task,
                exit_cls_id=exit_cls_id,
                destination_gs_id=task.destination_gs.node_id,
                remaining_packets=task.packet_count,
                admitted_slot=int(time_slot),
                upstream_delay_ms=max(float(upstream_delay_ms), 0.0),
            )
        )
        return True

    def queued_packets_by_gs(self) -> dict[int, float]:
        totals: dict[int, float] = defaultdict(float)
        for (_cls_id, gs_id, _class_idx), queue in self.queues.items():
            totals[gs_id] += sum(item.remaining_packets for item in queue)
        return dict(totals)

    def total_packets(self) -> float:
        return sum(item.remaining_packets for queue in self.queues.values() for item in queue)

    def service(self, snapshot: ScenarioSnapshot) -> list[dict[str, Any]]:
        events = self.expire(snapshot.time_sec)
        visible_edges: dict[tuple[int, int], ScenarioEdge] = {
            (edge.src_id, edge.dst_id): edge
            for edge in snapshot.edges
            if edge.link_type == LINK_DOWNLINK and edge.available and edge.is_data_flow_allowed
        }
        cls_ids = sorted({cls_id for cls_id, _gs_id, _class_idx in self.queues})
        for cls_id in cls_ids:
            eligible_keys = [
                key
                for key, queue in self.queues.items()
                if key[0] == cls_id and queue and (key[0], key[1]) in visible_edges
            ]
            if not eligible_keys:
                continue
            active_priority = min(self.config.priority_levels[key[2]] for key in eligible_keys)
            active_keys = [
                key for key in eligible_keys if self.config.priority_levels[key[2]] == active_priority
            ]
            total_weight = sum(self.config.downlink_wpq_weights[key[2]] for key in active_keys)
            for key in active_keys:
                edge = visible_edges[(key[0], key[1])]
                time_share = self.config.downlink_wpq_weights[key[2]] / max(total_weight, 1e-9)
                capacity_packets = (
                    float(edge.capacity_bps)
                    / self.packet_size_bits
                    * float(self.config.time_slot_seconds)
                    * time_share
                )
                events.extend(self._serve_queue(key, capacity_packets, edge, snapshot))
        self._remove_empty()
        return events

    def _serve_queue(
        self,
        key: tuple[int, int, int],
        capacity_packets: float,
        edge: ScenarioEdge,
        snapshot: ScenarioSnapshot,
    ) -> list[dict[str, Any]]:
        queue = self.queues[key]
        events: list[dict[str, Any]] = []
        used_packets = 0.0
        rate_pps = float(edge.capacity_bps) / self.packet_size_bits
        while queue and capacity_packets > 1e-9:
            item = queue[0]
            served = min(item.remaining_packets, capacity_packets)
            item.remaining_packets -= served
            capacity_packets -= served
            used_packets += served
            if item.remaining_packets > 1e-9:
                break
            queue.popleft()
            completion_s = float(snapshot.time_sec) + used_packets / max(rate_pps, 1e-9)
            effective_completion_s = completion_s + item.upstream_delay_ms / 1000.0
            success = effective_completion_s <= item.task.absolute_deadline_s + 1e-9
            events.append(
                {
                    "task_id": item.task.task_id,
                    "success": success,
                    "fail_reason": None if success else "deadline_exceeded",
                    "completion_time_s": completion_s,
                    "delivered_packets": item.task.packet_count,
                    "exit_cls_id": item.exit_cls_id,
                    "destination_gs_id": item.destination_gs_id,
                }
            )
        return events

    def expire(self, current_time_s: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for key, queue in list(self.queues.items()):
            kept: deque[DownlinkQueueItem] = deque()
            while queue:
                item = queue.popleft()
                if (
                    float(current_time_s) + item.upstream_delay_ms / 1000.0
                    >= item.task.absolute_deadline_s
                ):
                    delivered_packets = max(
                        item.task.packet_count - item.remaining_packets,
                        0.0,
                    )
                    events.append(
                        {
                            "task_id": item.task.task_id,
                            "success": False,
                            "fail_reason": "deadline_exceeded",
                            "completion_time_s": float(current_time_s),
                            "delivered_packets": delivered_packets,
                            "exit_cls_id": item.exit_cls_id,
                            "destination_gs_id": item.destination_gs_id,
                        }
                    )
                else:
                    kept.append(item)
            self.queues[key] = kept
        self._remove_empty()
        return events

    def fail_all(self, current_time_s: float, reason: str = "episode_horizon_exceeded") -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for queue in self.queues.values():
            while queue:
                item = queue.popleft()
                delivered_packets = max(
                    item.task.packet_count - item.remaining_packets,
                    0.0,
                )
                events.append(
                    {
                        "task_id": item.task.task_id,
                        "success": False,
                        "fail_reason": reason,
                        "completion_time_s": float(current_time_s),
                        "delivered_packets": delivered_packets,
                        "exit_cls_id": item.exit_cls_id,
                        "destination_gs_id": item.destination_gs_id,
                    }
                )
        self._remove_empty()
        return events

    def queue_rows(self, time_slot: int, node_by_id: dict[int, ScenarioNode]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (cls_id, gs_id, class_idx), queue in sorted(self.queues.items()):
            rows.append(
                {
                    "time_slot": int(time_slot),
                    "exit_cls": node_by_id[cls_id].label,
                    "destination_gs": node_by_id[gs_id].label,
                    "traffic_class": class_idx,
                    "queued_packets": sum(item.remaining_packets for item in queue),
                    "task_count": len(queue),
                }
            )
        return rows

    def _remove_empty(self) -> None:
        self.queues = defaultdict(deque, {key: queue for key, queue in self.queues.items() if queue})


__all__ = [
    "DownlinkQueueItem",
    "DownlinkQueueManager",
    "MultiSourceTaskFactory",
    "RemoteSensingTask",
    "RoutingCandidate",
    "current_isl_edges",
    "shortest_isl_path",
    "task_edge_transmission_delay_ms",
]
