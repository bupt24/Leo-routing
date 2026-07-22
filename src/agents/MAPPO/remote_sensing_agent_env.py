"""Concurrent multi-source MAPPO environment for remote-sensing return traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from agents.MAPPO.remote_sensing_route_metrics import format_path
from env.remote_sensing_scenario import (
    LINK_ACCESS,
    LINK_ISL,
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
    RoutingCandidate,
    current_isl_edges,
    shortest_isl_path,
    task_edge_transmission_delay_ms,
)


ENV_SCHEMA_VERSION = 2
LOCAL_OBS_DIM = 24
CANDIDATE_FEATURE_DIM = 18


@dataclass(frozen=True)
class AgentRewardConfig:
    invalid_action_penalty: float = 1.0
    contention_penalty: float = 1.0


@dataclass
class ActiveRoute:
    task: RemoteSensingTask
    snapshot: ScenarioSnapshot
    current_node_id: int
    node_ids: list[int]
    visited_cls_ids: set[int]
    front_edges: list[ScenarioEdge] = field(default_factory=list)
    cls_edges: list[ScenarioEdge] = field(default_factory=list)
    selected_cls: str = ""
    exit_cls: str = ""
    route_delay_ms: float = 0.0
    energy_j_total: float = 0.0
    loss_total: float = 0.0
    reward_total: float = 0.0


@dataclass(frozen=True)
class CachedSlot:
    snapshot: ScenarioSnapshot


@dataclass(frozen=True)
class RemoteSensingEnvCache:
    slot_offset: int
    slots: list[CachedSlot]

    @classmethod
    def build(
        cls,
        config: RemoteSensingScenarioConfig,
        *,
        time_slots: int,
        slot_offset: int = 0,
    ) -> "RemoteSensingEnvCache":
        scenario = RemoteSensingScenario(config)
        total = max(1, int(time_slots)) + int(config.drain_slots) + int(config.gs_lookahead_slots)
        slots = [
            CachedSlot(scenario.build_snapshot(int(slot_offset) + index))
            for index in range(total)
        ]
        return cls(slot_offset=int(slot_offset), slots=slots)

    def slot(self, relative_slot: int) -> CachedSlot:
        return self.slots[int(relative_slot)]


def infer_default_action_dim(config: RemoteSensingScenarioConfig) -> int:
    access = max(int(config.max_access_cls_per_rls), 1)
    return max(access, 5)  # Walker CLS has at most four ISLs plus one egress action.


class RemoteSensingAgentEnv:
    """Shared-policy environment with one concurrent agent per generated task."""

    local_obs_dim = LOCAL_OBS_DIM
    candidate_feature_dim = CANDIDATE_FEATURE_DIM
    env_schema_version = ENV_SCHEMA_VERSION

    def __init__(
        self,
        config: RemoteSensingScenarioConfig,
        *,
        time_slots: int = 1,
        action_dim: int | None = None,
        max_cls_hops: int = 10,
        drain_slots: int | None = None,
        reward_config: AgentRewardConfig | None = None,
        device: str | torch.device = "cpu",
        cache: RemoteSensingEnvCache | None = None,
    ) -> None:
        self.config = config
        self.time_slots = max(1, int(time_slots))
        self.drain_slots = int(config.drain_slots if drain_slots is None else drain_slots)
        self.action_dim = int(action_dim or infer_default_action_dim(config))
        self.max_cls_hops = max(1, int(max_cls_hops))
        self.reward_config = reward_config or AgentRewardConfig()
        self.device = torch.device(device)
        self.cache = cache
        self.global_state_dim = self._infer_global_state_dim()

        self.scenario = RemoteSensingScenario(config)
        self.task_factory = MultiSourceTaskFactory(config, self.scenario.link_config.packet_size_bits)
        self.downlink = DownlinkQueueManager(config, self.scenario.link_config.packet_size_bits)
        self.episode = 0
        self.seed = 0
        self.slot_offset = 0
        self.current_slot = 0
        self.done = False
        self.snapshot: ScenarioSnapshot | None = None
        self.active_routes: dict[str, ActiveRoute] = {}
        self.pending_routes: dict[str, ActiveRoute] = {}
        self.current_candidates: dict[str, list[RoutingCandidate]] = {}
        self.routes: list[dict[str, Any]] = []
        self.topology_edges: list[ScenarioEdge] = []
        self.downlink_queue_rows: list[dict[str, Any]] = []
        self.slot_injected_packets: dict[tuple[int, int], float] = {}
        self._forecast_slots: list[ScenarioSnapshot] = []

    @property
    def active_route(self) -> ActiveRoute | None:
        return next(iter(self.active_routes.values()), None)

    def reset(
        self,
        *,
        episode: int = 0,
        seed: int = 0,
        slot_offset: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]] | None:
        self.episode = int(episode)
        self.seed = int(seed)
        self.slot_offset = int(slot_offset)
        self.current_slot = 0
        self.done = False
        self.scenario = RemoteSensingScenario(self.config)
        self.task_factory = MultiSourceTaskFactory(self.config, self.scenario.link_config.packet_size_bits)
        self.downlink = DownlinkQueueManager(self.config, self.scenario.link_config.packet_size_bits)
        self.snapshot = None
        self.active_routes = {}
        self.pending_routes = {}
        self.current_candidates = {}
        self.routes = []
        self.topology_edges = []
        self.downlink_queue_rows = []
        self.slot_injected_packets = {}
        self._forecast_slots = self._build_forecast_slots()
        return self._advance_until_decision()

    def _build_forecast_slots(self) -> list[ScenarioSnapshot]:
        total = self.time_slots + self.drain_slots + self.config.gs_lookahead_slots
        if self.cache is not None:
            if self.cache.slot_offset != self.slot_offset:
                raise ValueError(
                    f"Cache slot_offset={self.cache.slot_offset} does not match requested {self.slot_offset}."
                )
            if len(self.cache.slots) < total:
                raise ValueError("RemoteSensingEnvCache does not cover the episode and GS lookahead horizon.")
            return [self.cache.slot(index).snapshot for index in range(total)]
        forecaster = RemoteSensingScenario(self.config)
        return [forecaster.build_snapshot(self.slot_offset + index) for index in range(total)]

    def _activate_snapshot(self, relative_slot: int) -> ScenarioSnapshot:
        snapshot = self._forecast_slots[relative_slot]
        self.snapshot = snapshot
        self.scenario.node_by_id = {node.node_id: node for node in snapshot.nodes}
        self.scenario.cls_node_ids = [node.node_id for node in snapshot.nodes if node.node_type == NODE_CLS]
        self.scenario._refresh_cls_queue_metrics(snapshot.edges)
        self.topology_edges.extend(snapshot.edges)
        return snapshot

    def _advance_until_decision(
        self,
        terminal_updates: dict[str, float] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]] | None:
        terminal_updates = terminal_updates if terminal_updates is not None else {}
        total_slots = self.time_slots + self.drain_slots
        while self.current_slot < total_slots:
            snapshot = self._activate_snapshot(self.current_slot)
            self.slot_injected_packets = {}
            if self.current_slot < self.time_slots:
                future = self._forecast_slots[
                    self.current_slot : self.current_slot + self.config.gs_lookahead_slots
                ]
                tasks = self.task_factory.build_tasks(
                    snapshot,
                    future,
                    episode=self.episode,
                    seed=self.seed,
                    queued_packets_by_gs=self.downlink.queued_packets_by_gs(),
                )
                self._start_tasks(tasks)
                if self.active_routes:
                    self._refresh_all_candidates()
                    return self._build_observation_batch()
            self._finish_slot(snapshot, terminal_updates)
            self.current_slot += 1

        if self.pending_routes:
            final_time_s = (self.slot_offset + total_slots) * float(self.config.time_slot_seconds)
            events = self.downlink.fail_all(final_time_s)
            self._handle_delivery_events(events, terminal_updates)
        self.done = True
        self.snapshot = None
        return None

    def _start_tasks(self, tasks: list[RemoteSensingTask]) -> None:
        assert self.snapshot is not None
        for task in tasks:
            route = ActiveRoute(
                task=task,
                snapshot=self.snapshot,
                current_node_id=task.source_rls.node_id,
                node_ids=[task.target.node_id, task.source_rls.node_id],
                visited_cls_ids=set(),
                front_edges=[task.observation_edge],
                route_delay_ms=float(task.observation_edge.delay_ms),
                energy_j_total=float(task.observation_edge.energy_cost),
            )
            if task.destination_gs is None:
                self._finish_route(route, False, "no_reachable_gs", task.generation_time_s, {})
            elif not task.candidate_access_edges:
                self._finish_route(route, False, "no_access_edge", task.generation_time_s, {})
            else:
                self.active_routes[task.task_id] = route

    def _refresh_all_candidates(self, terminal_updates: dict[str, float] | None = None) -> None:
        terminal_updates = terminal_updates if terminal_updates is not None else {}
        self.current_candidates = {}
        for task_id, route in list(self.active_routes.items()):
            candidates = self._candidates_for(route)
            if not candidates:
                self._finish_route(
                    route,
                    False,
                    "no_available_next_hop",
                    route.snapshot.time_sec,
                    terminal_updates,
                )
                self.active_routes.pop(task_id, None)
                continue
            if len(candidates) > self.action_dim:
                raise ValueError(
                    f"action_dim={self.action_dim} cannot cover {len(candidates)} candidates for {task_id}."
                )
            self.current_candidates[task_id] = candidates

    def _candidates_for(self, route: ActiveRoute) -> list[RoutingCandidate]:
        task = route.task
        if not route.visited_cls_ids:
            return [
                RoutingCandidate("access", edge.src_id, edge.dst_id, edge.dst, edge=edge)
                for edge in task.candidate_access_edges
            ]
        assert task.destination_gs is not None
        candidates = [
            RoutingCandidate("isl", edge.src_id, edge.dst_id, edge.dst, edge=edge)
            for edge in current_isl_edges(route.snapshot, route.current_node_id)
            if edge.dst_id not in route.visited_cls_ids
        ]
        relay_hops = len(route.cls_edges)
        future = self._forecast_slots[
            self.current_slot : self.current_slot + self.config.gs_lookahead_slots
        ]
        forecast = self.task_factory.exit_forecast(
            route.current_node_id,
            task.destination_gs.node_id,
            future,
        )
        if forecast is not None and relay_hops >= int(self.config.min_cls_relay_hops):
            wait_slots, visible_slots, service_packets = forecast
            candidates.append(
                RoutingCandidate(
                    "egress",
                    route.current_node_id,
                    task.destination_gs.node_id,
                    task.destination_gs.label,
                    forecast_wait_slots=wait_slots,
                    forecast_visible_slots=visible_slots,
                    forecast_service_packets=service_packets,
                )
            )
        candidates.sort(key=lambda item: (item.kind == "egress", item.label, item.dst_id))
        return candidates

    def action_mask(self, task_id: str) -> torch.Tensor:
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        mask[: len(self.current_candidates.get(task_id, []))] = True
        return mask

    def _build_observation_batch(self) -> dict[str, dict[str, torch.Tensor]]:
        global_state = self._build_global_state()
        return {
            task_id: self._build_task_observation(route, global_state)
            for task_id, route in sorted(self.active_routes.items())
            if task_id in self.current_candidates
        }

    def _build_task_observation(
        self,
        route: ActiveRoute,
        global_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        task = route.task
        current = self.scenario.node_by_id[route.current_node_id]
        assert task.destination_gs is not None
        scale = self._position_scale()
        current_pos = current.position_km / scale
        source_pos = task.source_rls.position_km / scale
        gs_pos = task.destination_gs.position_km / scale
        direction = gs_pos - current_pos
        elapsed_s = max(float(route.snapshot.time_sec) - task.generation_time_s, 0.0)
        remaining = max(task.deadline_s - elapsed_s, 0.0) / task.deadline_s
        queue_occupancy, residual = self._node_class_capacity(route.current_node_id, task.traffic_class)
        priority_one_hot = [1.0 if task.traffic_class == index else 0.0 for index in range(3)]
        access_stage = 1.0 if not route.visited_cls_ids else 0.0
        local_obs = torch.tensor(
            [
                *[float(value.item()) for value in current_pos],
                *[float(value.item()) for value in source_pos],
                *[float(value.item()) for value in gs_pos],
                *[float(value.item()) for value in direction],
                min(task.packet_count / max(self.config.queue_capacities[-1], 1), 1.0),
                *priority_one_hot,
                min(remaining, 1.0),
                min(route.route_delay_ms / max(task.deadline_s * 1000.0, 1.0), 1.0),
                queue_occupancy,
                residual,
                access_stage,
                1.0 - access_stage,
                len(route.cls_edges) / float(self.max_cls_hops),
                len(self.active_routes) / max(float(self.config.num_targets), 1.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if local_obs.numel() != LOCAL_OBS_DIM:
            raise RuntimeError(f"Unexpected local observation size {local_obs.numel()}.")
        candidate_features = torch.zeros(
            (self.action_dim, CANDIDATE_FEATURE_DIM), dtype=torch.float32, device=self.device
        )
        for index, candidate in enumerate(self.current_candidates[task.task_id]):
            candidate_features[index] = torch.tensor(
                self._candidate_features(route, candidate), dtype=torch.float32, device=self.device
            )
        return {
            "local_obs": local_obs,
            "candidate_features": candidate_features,
            "action_mask": self.action_mask(task.task_id),
            "global_state": global_state.clone(),
        }

    def _candidate_features(self, route: ActiveRoute, candidate: RoutingCandidate) -> list[float]:
        task = route.task
        current = self.scenario.node_by_id[route.current_node_id]
        destination = self.scenario.node_by_id[candidate.dst_id]
        rel = (destination.position_km - current.position_km) / self._position_scale()
        kinds = [
            1.0 if candidate.kind == "access" else 0.0,
            1.0 if candidate.kind == "isl" else 0.0,
            1.0 if candidate.kind == "egress" else 0.0,
        ]
        distance = float(candidate.edge.distance_km) if candidate.edge is not None else 0.0
        delay = (
            task_edge_transmission_delay_ms(candidate.edge, task, self.scenario.link_config.packet_size_bits)
            if candidate.edge is not None
            else candidate.forecast_wait_slots * self.config.time_slot_seconds * 1000.0
        )
        energy = self._task_edge_energy(candidate.edge, task) if candidate.edge is not None else 0.0
        loss = float(candidate.edge.loss_risk) if candidate.edge is not None else 0.0
        if candidate.kind == "egress":
            assert task.destination_gs is not None
            queued = self.downlink.queue_length(
                route.current_node_id, task.destination_gs.node_id, task.traffic_class
            )
            capacity = float(self.config.downlink_queue_capacities[task.traffic_class])
            occupancy = min(queued / max(capacity, 1.0), 1.0)
            residual = max(capacity - queued, 0.0) / max(capacity, 1.0)
            other_injected = 0.0
            reachable = 1.0
            wait = candidate.forecast_wait_slots / max(float(self.config.gs_lookahead_slots), 1.0)
            service = min(candidate.forecast_service_packets / max(task.packet_count, 1.0), 1.0)
        else:
            occupancy, residual = self._node_class_capacity(candidate.dst_id, task.traffic_class)
            other_injected = self._other_rls_pressure(task.task_id, candidate.dst_id) / max(
                float(self.config.queue_capacities[task.traffic_class]), 1.0
            )
            assert task.destination_gs is not None
            future = self._forecast_slots[
                self.current_slot : self.current_slot + self.config.gs_lookahead_slots
            ]
            exit_ids = {
                edge.src_id
                for snap in future
                for edge in snap.edges
                if edge.link_type == "downlink"
                and edge.dst_id == task.destination_gs.node_id
                and edge.dst_type == NODE_ES
            }
            path, path_delay = shortest_isl_path(route.snapshot, candidate.dst_id, exit_ids)
            reachable = 1.0 if path else 0.0
            wait = 0.0
            service = 0.0 if not path else 1.0 / (1.0 + path_delay)
        gs_load = self.downlink.queued_packets_by_gs().get(candidate.dst_id, 0.0)
        if task.destination_gs is not None:
            gs_load = self.downlink.queued_packets_by_gs().get(task.destination_gs.node_id, 0.0)
        fit = min(residual * float(self.config.queue_capacities[task.traffic_class]) / max(task.packet_count, 1.0), 1.0)
        return [
            float(rel[0].item()),
            float(rel[1].item()),
            float(rel[2].item()),
            *kinds,
            min(distance / self._distance_norm(), 1.0),
            min(delay / max(task.deadline_s * 1000.0, 1.0), 1.0),
            min(energy / max(self.config.normalizer_energy_j, 1e-9), 1.0),
            min(loss / max(self.config.normalizer_loss, 1e-9), 1.0),
            occupancy,
            residual,
            min(other_injected, 1.0),
            reachable,
            min(wait, 1.0),
            service,
            min(gs_load / max(sum(self.config.downlink_queue_capacities), 1.0), 1.0),
            fit,
        ]

    def _other_rls_pressure(self, task_id: str, cls_id: int) -> float:
        pressure = sum(
            route.task.packet_count
            for other_id, route in self.active_routes.items()
            if other_id != task_id
            and any(edge.dst_id == cls_id for edge in route.task.candidate_access_edges)
        )
        pressure += sum(
            packets
            for (reserved_cls_id, _class_idx), packets in self.slot_injected_packets.items()
            if reserved_cls_id == cls_id
        )
        return pressure

    def _node_class_capacity(self, node_id: int, class_idx: int) -> tuple[float, float]:
        node = self.scenario.node_by_id.get(node_id)
        if node is None or node.node_type != NODE_CLS:
            return 0.0, 1.0
        base = float(self.scenario.queue_lengths[node.local_id, class_idx].item())
        injected = self.slot_injected_packets.get((node_id, class_idx), 0.0)
        capacity = float(self.config.queue_capacities[class_idx])
        occupancy = min((base + injected) / max(capacity, 1.0), 1.0)
        return occupancy, max(capacity - base - injected, 0.0) / max(capacity, 1.0)

    def _build_global_state(self) -> torch.Tensor:
        values: list[float] = [
            self.current_slot / max(float(self.time_slots + self.drain_slots - 1), 1.0)
        ]
        capacities = torch.tensor(self.config.queue_capacities, dtype=torch.float32)
        normalized = self.scenario.queue_lengths / capacities.unsqueeze(0)
        for class_idx in range(self.config.num_traffic_classes):
            values.append(float(normalized[:, class_idx].mean().item()))
            values.append(float(normalized[:, class_idx].max().item()))
        gs_nodes = sorted(
            (node for node in self.scenario.node_by_id.values() if node.node_type == NODE_ES),
            key=lambda node: node.node_id,
        )
        gs_load = self.downlink.queued_packets_by_gs()
        for gs in gs_nodes:
            values.append(min(gs_load.get(gs.node_id, 0.0) / max(sum(self.config.downlink_queue_capacities), 1.0), 1.0))
        future = self._forecast_slots[
            self.current_slot : self.current_slot + self.config.gs_lookahead_slots
        ]
        for gs in gs_nodes:
            service = sum(
                float(edge.capacity_bps) * self.config.time_slot_seconds / self.scenario.link_config.packet_size_bits
                for snap in future
                for edge in snap.edges
                if edge.dst_id == gs.node_id and edge.dst_type == NODE_ES
            )
            values.append(min(service / max(sum(self.config.downlink_queue_capacities), 1.0), 1.0))
        profile_counts = [0.0, 0.0, 0.0]
        for route in self.active_routes.values():
            profile_counts[min(route.task.traffic_class, 2)] += 1.0
        values.extend(count / max(float(self.config.num_targets), 1.0) for count in profile_counts)
        values.append(len(self.active_routes) / max(float(self.config.num_targets), 1.0))
        for task_id in sorted(self.active_routes)[: self.config.num_targets]:
            route = self.active_routes[task_id]
            task = route.task
            current = self.scenario.node_by_id[route.current_node_id]
            assert task.destination_gs is not None
            current_pos = current.position_km / self._position_scale()
            gs_pos = task.destination_gs.position_km / self._position_scale()
            elapsed = max(route.snapshot.time_sec - task.generation_time_s, 0.0)
            values.extend(float(value.item()) for value in current_pos)
            values.extend(float(value.item()) for value in gs_pos)
            values.append(min(task.packet_count / max(self.config.queue_capacities[-1], 1), 1.0))
            values.extend(1.0 if task.traffic_class == index else 0.0 for index in range(3))
            values.append(max(task.deadline_s - elapsed, 0.0) / task.deadline_s)
            values.extend([1.0, 0.0] if not route.visited_cls_ids else [0.0, 1.0])
            values.append(len(route.cls_edges) / float(self.max_cls_hops))
        per_task_dim = 14
        missing = self.config.num_targets - min(len(self.active_routes), self.config.num_targets)
        values.extend([0.0] * (missing * per_task_dim))
        state = torch.tensor(values, dtype=torch.float32, device=self.device)
        if state.numel() != self.global_state_dim:
            raise RuntimeError(f"Unexpected global state size {state.numel()} != {self.global_state_dim}.")
        return state

    def step(
        self,
        actions: dict[str, int] | int,
    ) -> tuple[
        dict[str, dict[str, torch.Tensor]] | None,
        dict[str, float],
        bool,
        dict[str, Any],
    ]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset().")
        if isinstance(actions, int):
            if len(self.active_routes) != 1:
                raise ValueError("Integer actions are only valid when exactly one task is active.")
            actions = {next(iter(self.active_routes)): int(actions)}
        rewards: dict[str, float] = {}
        decision_done: dict[str, bool] = {}
        terminal_updates: dict[str, float] = {}
        ordered_ids = sorted(
            self.active_routes,
            key=lambda task_id: (
                self.active_routes[task_id].task.priority,
                self.active_routes[task_id].task.deadline_s,
                task_id,
            ),
        )
        for task_id in ordered_ids:
            route = self.active_routes.get(task_id)
            if route is None:
                continue
            action = int(actions.get(task_id, -1))
            candidates = self.current_candidates.get(task_id, [])
            if action < 0 or action >= len(candidates):
                reward = -float(self.reward_config.invalid_action_penalty)
                rewards[task_id] = reward
                route.reward_total += reward
                self._finish_route(route, False, "invalid_action", route.snapshot.time_sec, terminal_updates)
                self.active_routes.pop(task_id, None)
                decision_done[task_id] = True
                continue
            candidate = candidates[action]
            reward, finished = self._apply_candidate(route, candidate, terminal_updates)
            rewards[task_id] = reward
            route.reward_total += reward
            if finished:
                for record in reversed(self.routes):
                    if record.get("task_id") == task_id:
                        record["reward_total"] = route.reward_total
                        break
            decision_done[task_id] = finished
            if finished:
                self.active_routes.pop(task_id, None)

        self._refresh_all_candidates(terminal_updates)
        for task_id in rewards:
            decision_done[task_id] = task_id not in self.active_routes
        if self.active_routes:
            return self._build_observation_batch(), rewards, False, {
                "decision_done": decision_done,
                "terminal_reward_updates": terminal_updates,
                "completed_routes": [],
            }
        assert self.snapshot is not None
        completed_before = len(self.routes)
        self._finish_slot(self.snapshot, terminal_updates)
        self.current_slot += 1
        next_obs = self._advance_until_decision(terminal_updates)
        return next_obs, rewards, self.done, {
            "decision_done": decision_done,
            "terminal_reward_updates": terminal_updates,
            "completed_routes": self.routes[completed_before:],
        }

    def _apply_candidate(
        self,
        route: ActiveRoute,
        candidate: RoutingCandidate,
        terminal_updates: dict[str, float],
    ) -> tuple[float, bool]:
        task = route.task
        if candidate.kind in {"access", "isl"}:
            assert candidate.edge is not None
            capacity = float(self.config.queue_capacities[task.traffic_class])
            destination = self.scenario.node_by_id[candidate.dst_id]
            base = float(self.scenario.queue_lengths[destination.local_id, task.traffic_class].item())
            injected = self.slot_injected_packets.get((candidate.dst_id, task.traffic_class), 0.0)
            if base + injected + task.packet_count > capacity + 1e-9:
                reward = -float(self.reward_config.contention_penalty)
                self._finish_route(route, False, "cls_queue_contention", route.snapshot.time_sec, terminal_updates)
                return reward, True
            self.slot_injected_packets[(candidate.dst_id, task.traffic_class)] = injected + task.packet_count
            route.node_ids.append(candidate.dst_id)
            route.route_delay_ms += task_edge_transmission_delay_ms(
                candidate.edge, task, self.scenario.link_config.packet_size_bits
            )
            route.energy_j_total += self._task_edge_energy(candidate.edge, task)
            route.loss_total += float(candidate.edge.loss_risk)
            if candidate.kind == "access":
                route.front_edges.append(candidate.edge)
                route.selected_cls = candidate.edge.dst
            else:
                route.cls_edges.append(candidate.edge)
            route.current_node_id = candidate.dst_id
            route.visited_cls_ids.add(candidate.dst_id)
            reward = self._edge_reward(route, candidate)
            if route.route_delay_ms > task.deadline_s * 1000.0:
                self._finish_route(route, False, "deadline_exceeded", route.snapshot.time_sec, terminal_updates)
                return reward, True
            if len(route.cls_edges) >= self.max_cls_hops:
                self._finish_route(route, False, "ttl_exceeded", route.snapshot.time_sec, terminal_updates)
                return reward, True
            return reward, False

        if candidate.kind == "egress":
            route.exit_cls = self.scenario.node_by_id[route.current_node_id].label
            if not self.downlink.admit(
                task,
                route.current_node_id,
                self.current_slot,
                upstream_delay_ms=route.route_delay_ms,
            ):
                reward = -float(self.reward_config.contention_penalty)
                self._finish_route(route, False, "downlink_queue_overflow", route.snapshot.time_sec, terminal_updates)
                return reward, True
            assert task.destination_gs is not None
            route.node_ids.append(task.destination_gs.node_id)
            self.pending_routes[task.task_id] = route
            return self._egress_cost(route, candidate), True
        raise ValueError(f"Unsupported candidate kind {candidate.kind!r}.")

    def _finish_slot(self, snapshot: ScenarioSnapshot, terminal_updates: dict[str, float]) -> None:
        for (cls_id, class_idx), packets in self.slot_injected_packets.items():
            node = self.scenario.node_by_id[cls_id]
            capacity = float(self.config.queue_capacities[class_idx])
            self.scenario.queue_lengths[node.local_id, class_idx] = min(
                float(self.scenario.queue_lengths[node.local_id, class_idx].item()) + packets,
                capacity,
            )
        self.downlink_queue_rows.extend(
            self.downlink.queue_rows(snapshot.time_slot, self.scenario.node_by_id)
        )
        events = self.downlink.service(snapshot)
        self._handle_delivery_events(events, terminal_updates)
        self.scenario._service_cls_queues(snapshot)

    def _handle_delivery_events(
        self,
        events: list[dict[str, Any]],
        terminal_updates: dict[str, float],
    ) -> None:
        for event in events:
            route = self.pending_routes.pop(str(event["task_id"]), None)
            if route is None:
                continue
            self._finish_route(
                route,
                bool(event["success"]),
                event.get("fail_reason"),
                float(event["completion_time_s"]),
                terminal_updates,
                delivered_packets=float(event.get("delivered_packets", 0.0)),
            )

    def _finish_route(
        self,
        route: ActiveRoute,
        success: bool,
        fail_reason: str | None,
        completion_time_s: float,
        terminal_updates: dict[str, float],
        *,
        delivered_packets: float = 0.0,
    ) -> None:
        task = route.task
        elapsed_ms = max(float(completion_time_s) - task.generation_time_s, 0.0) * 1000.0
        delay_ms = elapsed_ms + route.route_delay_ms
        terminal = (
            float(self.config.reward_success_bonus) - min(delay_ms / max(task.deadline_s * 1000.0, 1.0), 1.0)
            if success
            else -float(self.config.reward_failure_penalty)
        )
        terminal_updates[task.task_id] = terminal_updates.get(task.task_id, 0.0) + terminal
        route.reward_total += terminal
        destination = task.destination_gs.label if task.destination_gs is not None else ""
        record = {
            "episode": task.episode,
            "time_slot": task.generation_slot,
            "generation_slot": task.generation_slot,
            "completion_slot": int(completion_time_s // max(self.config.time_slot_seconds, 1e-9)),
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
            "selected_cls": route.selected_cls,
            "selected_es": destination,
            "destination_gs": destination,
            "exit_cls": route.exit_cls,
            "success": success,
            "deadline_met": success,
            "access_success": bool(route.selected_cls),
            "route_success": success,
            "fail_reason": fail_reason,
            "hop_count": max(len(route.node_ids) - 1, 0),
            "cls_relay_hops": len(route.cls_edges),
            "delay_ms_total": delay_ms,
            "total_delay_ms": delay_ms,
            "delay_ms_cls_route": max(route.route_delay_ms - float(task.observation_edge.delay_ms), 0.0),
            "completion_delay_ms": elapsed_ms,
            "energy_j_total": route.energy_j_total,
            "total_energy": route.energy_j_total,
            "loss_total": route.loss_total,
            "total_loss_risk": route.loss_total,
            "delivered_packets": delivered_packets,
            "reward_total": route.reward_total,
            "path": format_path(self.scenario, route.node_ids),
            "gs_score": task.gs_score,
            "gs_path_delay_score": task.gs_score_components.get("path_delay"),
            "gs_exit_reachability_score": task.gs_score_components.get("exit_reachability"),
            "gs_visibility_score": task.gs_score_components.get("visibility_window"),
            "gs_assigned_load_score": task.gs_score_components.get("assigned_load"),
        }
        self.routes.append(record)

    def _edge_reward(self, route: ActiveRoute, candidate: RoutingCandidate) -> float:
        assert candidate.edge is not None
        task = route.task
        occupancy, _residual = self._node_class_capacity(candidate.dst_id, task.traffic_class)
        delay = min(
            task_edge_transmission_delay_ms(candidate.edge, task, self.scenario.link_config.packet_size_bits)
            / max(task.deadline_s * 1000.0, 1.0),
            1.0,
        )
        energy = min(self._task_edge_energy(candidate.edge, task) / max(self.config.normalizer_energy_j, 1e-9), 1.0)
        loss = min(float(candidate.edge.loss_risk) / max(self.config.normalizer_loss, 1e-9), 1.0)
        weights = self.config.routing_weights
        return -float(weights.delay * delay + weights.energy * energy + weights.queue * occupancy + weights.loss * loss)

    def _egress_cost(self, route: ActiveRoute, candidate: RoutingCandidate) -> float:
        task = route.task
        assert task.destination_gs is not None
        queued = self.downlink.queue_length(route.current_node_id, task.destination_gs.node_id, task.traffic_class)
        capacity = float(self.config.downlink_queue_capacities[task.traffic_class])
        queue_cost = min(queued / max(capacity, 1.0), 1.0)
        wait_cost = min(candidate.forecast_wait_slots / max(float(self.config.gs_lookahead_slots), 1.0), 1.0)
        return -float(self.config.routing_weights.delay * wait_cost + self.config.routing_weights.queue * queue_cost)

    def _task_edge_energy(self, edge: ScenarioEdge | None, task: RemoteSensingTask) -> float:
        if edge is None or edge.capacity_bps <= 0.0:
            return 0.0
        duration_s = task.packet_count * self.scenario.link_config.packet_size_bits / float(edge.capacity_bps)
        return (self.scenario.link_config.tx_power_w + self.scenario.link_config.rx_power_w) * duration_s

    def episode_summary(self) -> dict[str, Any]:
        success = [route for route in self.routes if route["success"]]
        total = len(self.routes)

        def average(key: str, rows: list[dict[str, Any]]) -> float:
            return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

        delivered_packets = sum(float(route.get("delivered_packets", 0.0)) for route in success)
        duration_s = max(self.time_slots * self.config.time_slot_seconds, 1.0)
        return {
            "episode": self.episode,
            "seed": self.seed,
            "slot_offset": self.slot_offset,
            "time_slots": self.time_slots,
            "drain_slots": self.drain_slots,
            "action_dim": self.action_dim,
            "max_cls_hops": self.max_cls_hops,
            "total_tasks": total,
            "success_count": len(success),
            "failed_count": total - len(success),
            "success_rate": len(success) / total if total else 0.0,
            "deadline_meeting_rate": len(success) / total if total else 0.0,
            "avg_delay_success_ms": average("delay_ms_total", success),
            "avg_delay_actual_all_ms": average("delay_ms_total", self.routes),
            "avg_cls_delay_success_ms": average("delay_ms_cls_route", success),
            "avg_energy_success_j": average("energy_j_total", success),
            "avg_loss_success": average("loss_total", success),
            "avg_reward_all": average("reward_total", self.routes),
            "throughput_mbps": delivered_packets * self.scenario.link_config.packet_size_bits / duration_s / 1e6,
            "rls_reuse_count": sum(1 for route in self.routes if route.get("rls_reused")),
        }

    def _infer_global_state_dim(self) -> int:
        base = 1 + 2 * self.config.num_traffic_classes
        base += 2 * self.config.num_es
        base += 3 + 1
        return base + self.config.num_targets * 14

    def _position_scale(self) -> float:
        return max(float(self.config.cls_altitude_km) + 6371.0, 1.0)

    def _distance_norm(self) -> float:
        return max(self.config.max_cross_layer_distance_km, self.config.max_cls_isl_distance_km, 1.0)


__all__ = [
    "AgentRewardConfig",
    "ActiveRoute",
    "CANDIDATE_FEATURE_DIM",
    "CachedSlot",
    "ENV_SCHEMA_VERSION",
    "LOCAL_OBS_DIM",
    "RemoteSensingAgentEnv",
    "RemoteSensingEnvCache",
    "infer_default_action_dim",
]
