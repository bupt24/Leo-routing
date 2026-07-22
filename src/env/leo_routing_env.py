"""
Paper-aligned LEO routing environment with dynamic topology, queueing, WPQ, AQM,
delay, energy, and reward modeling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

from env.aqm_wpq import (
    allocate_wpq_service,
    compute_aqm_drop_probabilities,
    normalize_traffic_distribution,
    sanitize_aqm_params,
    sanitize_wpq_weights,
)
from env.link_model import (
    LinkModelConfig,
    SPEED_OF_LIGHT_KM_PER_MS,
    compute_link_rates_bps,
    compute_propagation_delay_ms,
    compute_reception_energy_j,
    compute_transmission_delay_ms,
    compute_transmission_energy_j,
)
from env.queue_model import (
    compute_aggregate_queueing_delay_ms,
    compute_node_service_rate_packets,
    compute_per_class_queueing_delay_ms,
    compute_served_packets,
    update_queue_lengths,
)
from env.topology import (
    check_line_of_sight,
    compute_grid_plus_isl,
    generate_walker_constellation,
)


EPSILON = 1e-9


@dataclass(frozen=True)
class TopologyConfig:
    num_planes: int = 4
    sats_per_plane: int = 8
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    phase_offset: int = 1
    max_neighbors: int = 4
    max_isl_distance_km: float = 10_000.0
    use_visibility: bool = True


@dataclass(frozen=True)
class TrafficConfig:
    num_classes: int = 3
    queue_capacities: tuple[int, ...] = (100, 100, 100)
    priority_levels: tuple[int, ...] = (0, 1, 2)
    source_nodes: tuple[int, ...] | None = None
    destination_nodes: tuple[int, ...] | None = None
    arrival_rates_pps: tuple[float, ...] = (8.0, 5.0, 3.0)
    total_packets: int = 10_000
    class_packet_budget: tuple[int, ...] | None = None
    stochastic_arrivals: bool = False


@dataclass(frozen=True)
class RewardConfig:
    weights: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    cost_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
    normalizers: tuple[float, float, float] | None = None
    delay_metric: str = "average"


@dataclass(frozen=True)
class EnvironmentConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    link: LinkModelConfig = field(default_factory=LinkModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    slot_duration_s: float = 1.0
    initial_energy_j: float = 100.0
    max_steps: int = 100
    max_end_to_end_delay_ms: float = 200.0
    default_aqm: tuple[float, float, float] = (0.4, 0.8, 0.1)
    device: str = "cpu"


@dataclass
class GraphSnapshot:
    time_index: int
    time_sec: float
    positions_km: torch.Tensor
    edge_index: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_distances_km: torch.Tensor
    valid_neighbor_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.positions_km.shape[0])


class LEORoutingEnv:
    """
    A discrete-time environment that keeps the paper's core state-transition chain:
    topology -> link rates -> service rates -> WPQ/AQM -> queue update ->
    delay/energy/loss -> observation/reward.
    """

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.device = torch.device(self.config.device)
        self._validate_config()

        self.num_classes = self.config.traffic.num_classes
        self.queue_capacities = torch.tensor(
            self.config.traffic.queue_capacities,
            dtype=torch.float32,
            device=self.device,
        )
        self.priority_levels = torch.tensor(
            self.config.traffic.priority_levels,
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_weights = torch.tensor(
            self.config.reward.weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.default_aqm = torch.tensor(
            self.config.default_aqm,
            dtype=torch.float32,
            device=self.device,
        )

        self.current_snapshot: GraphSnapshot | None = None
        self.queue_lengths: torch.Tensor | None = None
        self.remaining_energy: torch.Tensor | None = None
        self.current_wpq_weights: torch.Tensor | None = None
        self.current_aqm_params: torch.Tensor | None = None
        self.current_traffic_distribution: torch.Tensor | None = None

        self.source_nodes: torch.Tensor | None = None
        self.destination_nodes: torch.Tensor | None = None
        self.arrival_rates_pps: torch.Tensor | None = None
        self.remaining_packet_budget: torch.Tensor | None = None

        self.current_drop_probabilities: torch.Tensor | None = None
        self.current_queue_delay_ms: torch.Tensor | None = None
        self.current_total_link_delay_ms: torch.Tensor | None = None
        self.current_link_rates_bps: torch.Tensor | None = None

        self.reward_cost_mins: torch.Tensor | None = None
        self.reward_cost_maxs: torch.Tensor | None = None
        self.reward_cost_ranges: torch.Tensor | None = None
        self.time_index = 0
        self.done = False

    def _validate_config(self) -> None:
        if self.config.traffic.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if len(self.config.traffic.queue_capacities) != self.config.traffic.num_classes:
            raise ValueError("queue_capacities must match num_classes.")
        if len(self.config.traffic.priority_levels) != self.config.traffic.num_classes:
            raise ValueError("priority_levels must match num_classes.")
        if len(self.config.reward.weights) != 3:
            raise ValueError("reward weights must provide delay/energy/loss coefficients.")
        if self.config.reward.delay_metric not in {"average", "total"}:
            raise ValueError("delay_metric must be 'average' or 'total'.")
        if self.config.reward.cost_bounds is not None:
            if len(self.config.reward.cost_bounds) != 3:
                raise ValueError("cost_bounds must contain delay/energy/loss min-max pairs.")
            for lower, upper in self.config.reward.cost_bounds:
                if upper <= lower:
                    raise ValueError("Each reward cost bound must satisfy upper > lower.")
        if self.config.reward.normalizers is not None:
            if len(self.config.reward.normalizers) != 3:
                raise ValueError("normalizers must match delay/energy/loss dimensions.")
            if any(value <= 0 for value in self.config.reward.normalizers):
                raise ValueError("normalizers must be positive.")
        if self.config.slot_duration_s <= 0:
            raise ValueError("slot_duration_s must be positive.")
        if self.config.topology.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive.")

    def _build_snapshot(self, time_index: int) -> GraphSnapshot:
        topo = self.config.topology
        time_sec = time_index * self.config.slot_duration_s
        positions_km = generate_walker_constellation(
            num_planes=topo.num_planes,
            sats_per_plane=topo.sats_per_plane,
            altitude_km=topo.altitude_km,
            inclination_deg=topo.inclination_deg,
            phase_offset=topo.phase_offset,
            time_sec=time_sec,
        ).to(self.device)

        num_nodes = positions_km.shape[0]
        distances_km = torch.cdist(positions_km, positions_km, p=2)
        if topo.use_visibility:
            visibility = check_line_of_sight(positions_km, positions_km).to(self.device)
        else:
            visibility = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=self.device)

        edge_sources, edge_targets = compute_grid_plus_isl(
            topo.num_planes,
            topo.sats_per_plane,
            positions_km,
        )

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(num_nodes)]
        for src, tgt in zip(edge_sources, edge_targets):
            if src == tgt:
                continue
            distance = float(distances_km[src, tgt].item())
            if distance > topo.max_isl_distance_km:
                continue
            if topo.use_visibility and not bool(visibility[src, tgt].item()):
                continue
            neighbors[src].append((tgt, distance))

        max_neighbors = topo.max_neighbors
        neighbor_indices = torch.full(
            (num_nodes, max_neighbors),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        neighbor_distances = torch.zeros(
            (num_nodes, max_neighbors),
            dtype=torch.float32,
            device=self.device,
        )
        valid_neighbor_mask = torch.zeros(
            (num_nodes, max_neighbors),
            dtype=torch.bool,
            device=self.device,
        )

        valid_sources: list[int] = []
        valid_targets: list[int] = []
        for node_id, node_neighbors in enumerate(neighbors):
            node_neighbors.sort(key=lambda item: item[1])
            for slot_idx, (nbr, dist_km) in enumerate(node_neighbors[:max_neighbors]):
                neighbor_indices[node_id, slot_idx] = nbr
                neighbor_distances[node_id, slot_idx] = dist_km
                valid_neighbor_mask[node_id, slot_idx] = True
                valid_sources.append(node_id)
                valid_targets.append(nbr)

        if valid_sources:
            edge_index = torch.tensor(
                [valid_sources, valid_targets],
                dtype=torch.long,
                device=self.device,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return GraphSnapshot(
            time_index=time_index,
            time_sec=time_sec,
            positions_km=positions_km,
            edge_index=edge_index,
            neighbor_indices=neighbor_indices,
            neighbor_distances_km=neighbor_distances,
            valid_neighbor_mask=valid_neighbor_mask,
        )

    def _select_farthest_pair(self, snapshot: GraphSnapshot) -> tuple[int, int, int]:
        adjacency = []
        for node_id in range(snapshot.num_nodes):
            valid = snapshot.valid_neighbor_mask[node_id]
            adjacency.append(snapshot.neighbor_indices[node_id, valid].tolist())

        best_pair = (0, 0)
        best_hops = -1
        for src in range(snapshot.num_nodes):
            dist = {src: 0}
            q: deque[int] = deque([src])
            while q:
                node = q.popleft()
                for nbr in adjacency[node]:
                    if nbr not in dist:
                        dist[nbr] = dist[node] + 1
                        q.append(nbr)
            for dst, hops in dist.items():
                if hops > best_hops:
                    best_hops = hops
                    best_pair = (src, dst)
        return best_pair[0], best_pair[1], best_hops

    def _resolve_flow_nodes(self, snapshot: GraphSnapshot) -> None:
        src_cfg = self.config.traffic.source_nodes
        dst_cfg = self.config.traffic.destination_nodes

        if src_cfg is None or dst_cfg is None:
            source, destination, _ = self._select_farthest_pair(snapshot)
            if src_cfg is None:
                src_cfg = tuple([source] * self.num_classes)
            if dst_cfg is None:
                dst_cfg = tuple([destination] * self.num_classes)

        if len(src_cfg) != self.num_classes or len(dst_cfg) != self.num_classes:
            raise ValueError("source_nodes and destination_nodes must match num_classes.")

        self.source_nodes = torch.tensor(src_cfg, dtype=torch.long, device=self.device)
        self.destination_nodes = torch.tensor(dst_cfg, dtype=torch.long, device=self.device)
        if ((self.source_nodes < 0) | (self.source_nodes >= snapshot.num_nodes)).any():
            raise ValueError("source_nodes contain indices outside the topology.")
        if ((self.destination_nodes < 0) | (self.destination_nodes >= snapshot.num_nodes)).any():
            raise ValueError("destination_nodes contain indices outside the topology.")

        self.arrival_rates_pps = torch.tensor(
            self.config.traffic.arrival_rates_pps,
            dtype=torch.float32,
            device=self.device,
        )
        if self.arrival_rates_pps.numel() != self.num_classes:
            raise ValueError("arrival_rates_pps must match num_classes.")

    def _build_initial_packet_budget(self) -> torch.Tensor:
        if self.config.traffic.class_packet_budget is not None:
            if len(self.config.traffic.class_packet_budget) != self.num_classes:
                raise ValueError("class_packet_budget must match num_classes.")
            return torch.tensor(
                self.config.traffic.class_packet_budget,
                dtype=torch.float32,
                device=self.device,
            )

        total_packets = int(self.config.traffic.total_packets)
        base = total_packets // self.num_classes
        remainder = total_packets % self.num_classes
        budgets = [base] * self.num_classes
        for idx in range(remainder):
            budgets[idx] += 1
        return torch.tensor(budgets, dtype=torch.float32, device=self.device)

    def _current_link_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.current_snapshot is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = self.current_snapshot
        availability = snapshot.valid_neighbor_mask
        _, _, link_rates_bps = compute_link_rates_bps(
            snapshot.neighbor_distances_km,
            availability,
            self.config.link,
        )
        propagation_delay_ms = compute_propagation_delay_ms(
            snapshot.neighbor_distances_km,
            availability,
        )
        return availability, link_rates_bps, propagation_delay_ms

    def _refresh_observation_cache(self) -> None:
        if self.queue_lengths is None or self.current_snapshot is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = self.current_snapshot
        availability, link_rates_bps, propagation_delay_ms = self._current_link_state()
        total_service_rate_pps = compute_node_service_rate_packets(
            link_rates_bps,
            self.config.link.packet_size_bits,
        )
        service_rate_by_class_pps, active_mask = allocate_wpq_service(
            total_service_rate_pps,
            self.queue_lengths,
            self.priority_levels,
            self.current_wpq_weights,
        )
        queue_delay_by_class_ms = compute_per_class_queueing_delay_ms(
            self.queue_lengths,
            service_rate_by_class_pps,
            active_mask,
        )
        aggregate_queue_delay_ms = compute_aggregate_queueing_delay_ms(
            queue_delay_by_class_ms,
            active_mask,
        )

        alpha = self.current_aqm_params[:, 0]
        beta = self.current_aqm_params[:, 1]
        pmax = self.current_aqm_params[:, 2]
        drop_probabilities = compute_aqm_drop_probabilities(
            self.queue_lengths,
            self.queue_capacities,
            alpha,
            beta,
            pmax,
        )

        self.current_link_rates_bps = link_rates_bps
        self.current_drop_probabilities = drop_probabilities
        self.current_queue_delay_ms = aggregate_queue_delay_ms
        self.current_total_link_delay_ms = propagation_delay_ms + aggregate_queue_delay_ms.unsqueeze(1)

        self._ensure_reward_cost_bounds(
            snapshot=snapshot,
            total_service_rate_pps=total_service_rate_pps,
            propagation_delay_ms=propagation_delay_ms,
        )

    def _ensure_reward_cost_bounds(
        self,
        snapshot: GraphSnapshot,
        total_service_rate_pps: torch.Tensor,
        propagation_delay_ms: torch.Tensor,
    ) -> None:
        if (
            self.reward_cost_mins is not None
            and self.reward_cost_maxs is not None
            and self.reward_cost_ranges is not None
        ):
            return

        if self.config.reward.cost_bounds is not None:
            bounds = torch.tensor(
                self.config.reward.cost_bounds,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            positive_service = total_service_rate_pps[total_service_rate_pps > 0.0]
            service_floor = float(positive_service.median().item()) if positive_service.numel() else 1.0
            max_prop_delay_ms = float(propagation_delay_ms.max().item()) if propagation_delay_ms.numel() else 1.0
            queue_scale_ms = float(self.queue_capacities.max().item()) / max(service_floor, EPSILON) * 1000.0
            delay_scale_ms = max(
                self.config.max_end_to_end_delay_ms,
                max_prop_delay_ms + self.config.slot_duration_s * 1000.0 + queue_scale_ms,
            )
            if self.config.reward.normalizers is None:
                upper_bounds = torch.tensor(
                    [
                        delay_scale_ms,
                        snapshot.num_nodes * max(self.config.initial_energy_j, EPSILON),
                        snapshot.num_nodes * self.num_classes,
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                upper_bounds = torch.tensor(
                    self.config.reward.normalizers,
                    dtype=torch.float32,
                    device=self.device,
                )
            lower_bounds = torch.zeros_like(upper_bounds)
            bounds = torch.stack([lower_bounds, upper_bounds], dim=1)

        self.reward_cost_mins = bounds[:, 0]
        self.reward_cost_maxs = bounds[:, 1]
        self.reward_cost_ranges = (self.reward_cost_maxs - self.reward_cost_mins).clamp_min(EPSILON)

    def _build_observation(self) -> torch.Tensor:
        if (
            self.queue_lengths is None
            or self.current_drop_probabilities is None
            or self.current_total_link_delay_ms is None
            or self.remaining_energy is None
            or self.current_snapshot is None
        ):
            raise RuntimeError("Observation cache is not initialized.")

        queue_status = torch.stack(
            [self.queue_lengths, self.current_drop_probabilities],
            dim=2,
        ).reshape(self.queue_lengths.shape[0], -1)
        neighbor_count = self.current_snapshot.valid_neighbor_mask.sum(dim=1, keepdim=True).float()
        neighbor_info = torch.cat(
            [neighbor_count, self.current_snapshot.valid_neighbor_mask.float()],
            dim=1,
        )
        return torch.cat(
            [
                queue_status,
                self.current_total_link_delay_ms,
                self.remaining_energy.unsqueeze(1),
                neighbor_info,
            ],
            dim=1,
        )

    def export_graph_state(self, node_feature_dim: int = 10) -> dict[str, torch.Tensor]:
        """
        Compatibility adapter for the existing GNN pipeline.
        """
        if self.current_snapshot is None or self.current_total_link_delay_ms is None:
            raise RuntimeError("Environment has not been reset.")

        positions = self.current_snapshot.positions_km
        pos_scale = positions.norm(dim=1).max().clamp_min(1.0)
        norm_positions = positions / pos_scale

        degree = (
            self.current_snapshot.valid_neighbor_mask.float().sum(dim=1, keepdim=True)
            / self.config.topology.max_neighbors
        )
        mean_delay = self.current_total_link_delay_ms.sum(dim=1, keepdim=True) / (
            self.current_snapshot.valid_neighbor_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        delay_min = self.reward_cost_mins[0].clamp_min(0.0)
        delay_range = self.reward_cost_ranges[0].clamp_min(1.0)
        mean_delay = torch.clamp((mean_delay - delay_min) / delay_range, min=0.0, max=1.0)
        queue_load = self.queue_lengths.sum(dim=1, keepdim=True) / self.queue_capacities.sum().clamp_min(1.0)
        energy_ratio = self.remaining_energy.unsqueeze(1) / max(self.config.initial_energy_j, 1.0)
        mean_drop = self.current_drop_probabilities.mean(dim=1, keepdim=True)
        queue_delay = torch.clamp(
            (self.current_queue_delay_ms.unsqueeze(1) - delay_min) / delay_range,
            min=0.0,
            max=1.0,
        )
        time_ratio = torch.full(
            (self.current_snapshot.num_nodes, 1),
            float(self.time_index) / max(self.config.max_steps - 1, 1),
            dtype=torch.float32,
            device=self.device,
        )

        features = torch.cat(
            [
                norm_positions,
                degree,
                mean_delay,
                queue_load,
                energy_ratio,
                mean_drop,
                queue_delay,
                time_ratio,
            ],
            dim=1,
        )
        if features.shape[1] < node_feature_dim:
            pad = torch.zeros(
                (features.shape[0], node_feature_dim - features.shape[1]),
                dtype=features.dtype,
                device=features.device,
            )
            features = torch.cat([features, pad], dim=1)
        else:
            features = features[:, :node_feature_dim]

        action_mask = self.current_snapshot.valid_neighbor_mask.float()
        return {
            "state_x": features,
            "edge_index": self.current_snapshot.edge_index.clone(),
            "neighbor_indices": self.current_snapshot.neighbor_indices.clone(),
            "neighbor_delays": self.current_total_link_delay_ms.clone(),
            "action_mask": action_mask,
        }

    def default_joint_actions(self) -> dict[str, torch.Tensor]:
        if self.current_snapshot is None:
            raise RuntimeError("Environment has not been reset.")

        traffic = normalize_traffic_distribution(
            self.current_snapshot.valid_neighbor_mask.float(),
            self.current_snapshot.valid_neighbor_mask,
        )
        weights = torch.ones(
            (self.current_snapshot.num_nodes, self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        aqm = self.default_aqm.unsqueeze(0).repeat(self.current_snapshot.num_nodes, 1)
        return {"traffic": traffic, "weights": weights, "aqm": aqm}

    def greedy_shortest_path_actions(self) -> dict[str, torch.Tensor]:
        """
        Demo-friendly heuristic: route every node toward the most common destination
        using current propagation delays.
        """
        if self.current_snapshot is None or self.destination_nodes is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = self.current_snapshot
        target = int(torch.mode(self.destination_nodes).values.item())
        dist_to_target = torch.full(
            (snapshot.num_nodes,),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        dist_to_target[target] = 0.0

        reverse_adjacency: list[list[tuple[int, float]]] = [[] for _ in range(snapshot.num_nodes)]
        for node_id in range(snapshot.num_nodes):
            valid = snapshot.valid_neighbor_mask[node_id]
            neighbors = snapshot.neighbor_indices[node_id, valid]
            delays = snapshot.neighbor_distances_km[node_id, valid] / SPEED_OF_LIGHT_KM_PER_MS
            for nbr, delay in zip(neighbors.tolist(), delays.tolist()):
                reverse_adjacency[nbr].append((node_id, delay))

        visited = torch.zeros(snapshot.num_nodes, dtype=torch.bool, device=self.device)
        while True:
            masked = dist_to_target.masked_fill(visited, float("inf"))
            current_dist, current = masked.min(dim=0)
            if not torch.isfinite(current_dist):
                break
            visited[current] = True
            for predecessor, edge_cost in reverse_adjacency[int(current.item())]:
                candidate = current_dist + edge_cost
                if candidate < dist_to_target[predecessor]:
                    dist_to_target[predecessor] = candidate

        traffic = torch.zeros(
            (snapshot.num_nodes, self.config.topology.max_neighbors),
            dtype=torch.float32,
            device=self.device,
        )
        for node_id in range(snapshot.num_nodes):
            if node_id == target:
                continue
            valid_slots = snapshot.valid_neighbor_mask[node_id]
            if not valid_slots.any():
                continue
            neighbors = snapshot.neighbor_indices[node_id]
            prop_delay = snapshot.neighbor_distances_km[node_id] / SPEED_OF_LIGHT_KM_PER_MS
            candidate_score = prop_delay + dist_to_target[neighbors.clamp_min(0)]
            candidate_score = candidate_score.masked_fill(~valid_slots, float("inf"))
            best_slot = int(candidate_score.argmin().item())
            if torch.isfinite(candidate_score[best_slot]):
                traffic[node_id, best_slot] = 1.0

        traffic = normalize_traffic_distribution(traffic, snapshot.valid_neighbor_mask)
        weights = torch.ones(
            (snapshot.num_nodes, self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        aqm = self.default_aqm.unsqueeze(0).repeat(snapshot.num_nodes, 1)
        return {"traffic": traffic, "weights": weights, "aqm": aqm}

    def reset(
        self,
        seed: int | None = None,
        initial_queue_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        self.time_index = 0
        self.done = False
        self.current_snapshot = self._build_snapshot(self.time_index)
        self._resolve_flow_nodes(self.current_snapshot)

        if initial_queue_lengths is None:
            self.queue_lengths = torch.zeros(
                (self.current_snapshot.num_nodes, self.num_classes),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.queue_lengths = torch.as_tensor(
                initial_queue_lengths,
                dtype=torch.float32,
                device=self.device,
            )
            if self.queue_lengths.shape != (self.current_snapshot.num_nodes, self.num_classes):
                raise ValueError("initial_queue_lengths has an unexpected shape.")

        self.remaining_energy = torch.full(
            (self.current_snapshot.num_nodes,),
            float(self.config.initial_energy_j),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_wpq_weights = torch.ones(
            (self.current_snapshot.num_nodes, self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_aqm_params = self.default_aqm.unsqueeze(0).repeat(self.current_snapshot.num_nodes, 1)
        self.current_traffic_distribution = normalize_traffic_distribution(
            self.current_snapshot.valid_neighbor_mask.float(),
            self.current_snapshot.valid_neighbor_mask,
        )
        self.remaining_packet_budget = self._build_initial_packet_budget()
        self.reward_cost_mins = None
        self.reward_cost_maxs = None
        self.reward_cost_ranges = None
        self._refresh_observation_cache()
        return self._build_observation()

    def _build_external_arrivals(self) -> torch.Tensor:
        if (
            self.current_snapshot is None
            or self.source_nodes is None
            or self.arrival_rates_pps is None
            or self.remaining_packet_budget is None
        ):
            raise RuntimeError("Environment has not been reset.")

        arrivals = torch.zeros(
            (self.current_snapshot.num_nodes, self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        base_arrivals = self.arrival_rates_pps * self.config.slot_duration_s
        if self.config.traffic.stochastic_arrivals:
            sampled = torch.poisson(base_arrivals)
        else:
            sampled = base_arrivals
        sampled = torch.minimum(sampled, self.remaining_packet_budget)
        self.remaining_packet_budget = torch.clamp(self.remaining_packet_budget - sampled, min=0.0)
        arrivals[self.source_nodes, torch.arange(self.num_classes, device=self.device)] = sampled
        return arrivals

    def _compute_raw_costs(
        self,
        delay_cost: torch.Tensor,
        energy_cost: torch.Tensor,
        loss_cost: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([delay_cost, energy_cost, loss_cost], dim=-1)

    def _compute_cost_vector(self, raw_costs: torch.Tensor) -> torch.Tensor:
        if (
            self.reward_cost_mins is None
            or self.reward_cost_maxs is None
            or self.reward_cost_ranges is None
        ):
            raise RuntimeError("Reward cost bounds are not initialized.")

        mins = self.reward_cost_mins.to(device=raw_costs.device, dtype=raw_costs.dtype)
        ranges = self.reward_cost_ranges.to(device=raw_costs.device, dtype=raw_costs.dtype)
        while mins.dim() < raw_costs.dim():
            mins = mins.unsqueeze(0)
            ranges = ranges.unsqueeze(0)
        return torch.clamp((raw_costs - mins) / ranges, min=0.0, max=1.0)

    def _compute_scalar_reward(self, cost_vector: torch.Tensor) -> torch.Tensor:
        lambda_vector = self.reward_weights.to(device=cost_vector.device, dtype=cost_vector.dtype)
        while lambda_vector.dim() < cost_vector.dim():
            lambda_vector = lambda_vector.unsqueeze(0)
        return -(cost_vector * lambda_vector).sum(dim=-1)

    def step(self, joint_actions: dict[str, Any] | None = None) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("The environment is done. Call reset() before stepping again.")
        if self.current_snapshot is None or self.queue_lengths is None or self.remaining_energy is None:
            raise RuntimeError("Environment has not been reset.")

        snapshot = self.current_snapshot
        default_actions = self.default_joint_actions()
        joint_actions = joint_actions or default_actions

        traffic = joint_actions.get("traffic", default_actions["traffic"])
        weights = joint_actions.get("weights", default_actions["weights"])
        aqm = joint_actions.get("aqm", default_actions["aqm"])

        traffic = normalize_traffic_distribution(
            torch.as_tensor(traffic, dtype=torch.float32, device=self.device),
            snapshot.valid_neighbor_mask,
        )
        weights = sanitize_wpq_weights(
            torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        )
        aqm = sanitize_aqm_params(
            torch.as_tensor(aqm, dtype=torch.float32, device=self.device)
        ).to(self.device)

        availability, link_rates_bps, propagation_delay_ms = self._current_link_state()
        total_service_rate_pps = compute_node_service_rate_packets(
            link_rates_bps,
            self.config.link.packet_size_bits,
        )
        service_rate_by_class_pps, active_mask = allocate_wpq_service(
            total_service_rate_pps,
            self.queue_lengths,
            self.priority_levels,
            weights,
        )
        served_packets = compute_served_packets(
            self.queue_lengths,
            service_rate_by_class_pps,
            self.config.slot_duration_s,
        )

        queue_delay_by_class_ms = compute_per_class_queueing_delay_ms(
            self.queue_lengths,
            service_rate_by_class_pps,
            active_mask,
        )
        aggregate_queue_delay_ms = compute_aggregate_queueing_delay_ms(
            queue_delay_by_class_ms,
            active_mask,
        )

        alpha = aqm[:, 0]
        beta = aqm[:, 1]
        pmax = aqm[:, 2]
        drop_probabilities = compute_aqm_drop_probabilities(
            self.queue_lengths,
            self.queue_capacities,
            alpha,
            beta,
            pmax,
        )

        forwarded_packets = served_packets.unsqueeze(2) * traffic.unsqueeze(1)
        transmitted_packets_per_link = forwarded_packets.sum(dim=1)
        transmission_delay_ms = compute_transmission_delay_ms(
            transmitted_packets_per_link,
            link_rates_bps,
            self.config.link.packet_size_bits,
        )
        total_link_delay_ms = propagation_delay_ms + transmission_delay_ms + aggregate_queue_delay_ms.unsqueeze(1)

        incoming_forwarded = torch.zeros_like(self.queue_lengths)
        delivered_packets = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)

        for slot_idx in range(self.config.topology.max_neighbors):
            valid = snapshot.valid_neighbor_mask[:, slot_idx]
            if not valid.any():
                continue

            targets = snapshot.neighbor_indices[valid, slot_idx]
            packets_by_class = forwarded_packets[valid, :, slot_idx]
            for class_idx in range(self.num_classes):
                class_targets = targets
                class_packets = packets_by_class[:, class_idx]
                destination = int(self.destination_nodes[class_idx].item())
                delivered_mask = class_targets == destination
                if delivered_mask.any():
                    delivered_packets[class_idx] += class_packets[delivered_mask].sum()
                forwarded_mask = ~delivered_mask
                if forwarded_mask.any():
                    incoming_forwarded[:, class_idx].scatter_add_(
                        0,
                        class_targets[forwarded_mask],
                        class_packets[forwarded_mask],
                    )

        external_arrivals = self._build_external_arrivals()
        delivered_from_sources = external_arrivals[
            self.destination_nodes,
            torch.arange(self.num_classes, device=self.device),
        ]
        delivered_packets += delivered_from_sources
        external_arrivals[
            self.destination_nodes,
            torch.arange(self.num_classes, device=self.device),
        ] = 0.0

        total_arrivals = external_arrivals + incoming_forwarded
        aqm_drops = total_arrivals * drop_probabilities
        accepted_arrivals = total_arrivals - aqm_drops
        queue_update = update_queue_lengths(
            self.queue_lengths,
            accepted_arrivals,
            served_packets,
            self.queue_capacities,
        )

        tx_energy_per_link, link_duration_s = compute_transmission_energy_j(
            transmitted_packets_per_link,
            link_rates_bps,
            availability,
            self.config.link,
        )
        rx_energy_per_link = compute_reception_energy_j(
            link_duration_s,
            availability,
            self.config.link,
        )
        tx_energy_by_node = tx_energy_per_link.sum(dim=1)
        rx_energy_by_node = torch.zeros_like(tx_energy_by_node)
        for slot_idx in range(self.config.topology.max_neighbors):
            valid = snapshot.valid_neighbor_mask[:, slot_idx]
            if not valid.any():
                continue
            rx_energy_by_node.scatter_add_(
                0,
                snapshot.neighbor_indices[valid, slot_idx],
                rx_energy_per_link[valid, slot_idx],
            )

        node_energy = tx_energy_by_node + rx_energy_by_node
        next_energy = torch.clamp(self.remaining_energy - node_energy, min=0.0)

        packet_delay_sum_ms = (transmitted_packets_per_link * total_link_delay_ms).sum()
        total_transmitted_packets = transmitted_packets_per_link.sum().clamp_min(1.0)
        average_packet_delay_ms = packet_delay_sum_ms / total_transmitted_packets
        delay_cost = (
            average_packet_delay_ms
            if self.config.reward.delay_metric == "average"
            else packet_delay_sum_ms
        )
        energy_cost = node_energy.sum()
        loss_surrogate = drop_probabilities.sum()
        raw_costs = self._compute_raw_costs(delay_cost, energy_cost, loss_surrogate)
        cost_vector = self._compute_cost_vector(raw_costs)
        reward = float(self._compute_scalar_reward(cost_vector).item())

        local_delay_cost = total_link_delay_ms.sum(dim=1) / availability.float().sum(dim=1).clamp_min(1.0)
        local_energy_cost = node_energy
        local_loss_cost = drop_probabilities.sum(dim=1)
        local_raw_costs = self._compute_raw_costs(
            local_delay_cost,
            local_energy_cost,
            local_loss_cost,
        )
        local_costs = self._compute_cost_vector(local_raw_costs)
        local_rewards = self._compute_scalar_reward(local_costs)

        self.queue_lengths = queue_update.next_queue_lengths
        self.remaining_energy = next_energy
        self.current_wpq_weights = weights
        self.current_aqm_params = aqm
        self.current_traffic_distribution = traffic

        self.time_index += 1
        max_step_reached = self.time_index >= self.config.max_steps
        energy_exhausted = bool((self.remaining_energy <= 0.0).all().item())
        packets_drained = bool(
            (self.remaining_packet_budget <= 0.0).all().item()
            and (self.queue_lengths <= 0.0).all().item()
        )
        self.done = max_step_reached or energy_exhausted or packets_drained

        if not self.done:
            self.current_snapshot = self._build_snapshot(self.time_index)
        self._refresh_observation_cache()
        obs = self._build_observation()

        delivered_total = float(delivered_packets.sum().item())
        step_metrics = {
            "return": reward,
            "delay": float(delay_cost.item()),
            "energy": float(energy_cost.item()),
            "loss": float(loss_surrogate.item()),
            "delay_average_ms": float(average_packet_delay_ms.item()),
            "delay_total_ms": float(packet_delay_sum_ms.item()),
            "delivered_total": delivered_total,
            "success": delivered_total > 0.0,
        }

        info: dict[str, Any] = {
            "costs": {
                "delay": float(delay_cost.item()),
                "delay_average_ms": float(average_packet_delay_ms.item()),
                "delay_total_ms": float(packet_delay_sum_ms.item()),
                "energy": float(energy_cost.item()),
                "loss_surrogate": float(loss_surrogate.item()),
                "delay_constraint_ms": float(self.config.max_end_to_end_delay_ms),
                "delay_constraint_violation": bool(
                    average_packet_delay_ms.item() > self.config.max_end_to_end_delay_ms
                ),
            },
            "step_metrics": step_metrics,
            "raw_costs": raw_costs.detach().cpu(),
            "cost_vector": cost_vector.detach().cpu(),
            "lambda_vector": self.reward_weights.detach().cpu(),
            "normalized_costs": cost_vector.detach().cpu(),
            "served_packets": served_packets.detach().cpu(),
            "queue_lengths": self.queue_lengths.detach().cpu(),
            "drop_probabilities": drop_probabilities.detach().cpu(),
            "aqm_drops": aqm_drops.detach().cpu(),
            "overflow_drops": queue_update.overflow_drops.detach().cpu(),
            "delivered_packets": delivered_packets.detach().cpu(),
            "node_energy": node_energy.detach().cpu(),
            "remaining_energy": self.remaining_energy.detach().cpu(),
            "remaining_packet_budget": self.remaining_packet_budget.detach().cpu(),
            "local_raw_costs": local_raw_costs.detach().cpu(),
            "local_costs": local_costs.detach().cpu(),
            "local_rewards": local_rewards.detach().cpu(),
            "action_mask": self.current_snapshot.valid_neighbor_mask.detach().cpu(),
            "neighbor_indices": self.current_snapshot.neighbor_indices.detach().cpu(),
            "neighbor_delays_ms": self.current_total_link_delay_ms.detach().cpu(),
            "termination": {
                "done": self.done,
                "max_step_reached": max_step_reached,
                "energy_exhausted": energy_exhausted,
                "packets_drained": packets_drained,
            },
        }
        return obs, reward, self.done, info
