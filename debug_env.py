"""Small closed-loop debug runner for the paper-style LEO routing environment."""

from __future__ import annotations

import pathlib
import sys

import torch


CURRENT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.leo_routing_env import (  # noqa: E402
    EnvironmentConfig,
    LEORoutingEnv,
    RewardConfig,
    TopologyConfig,
    TrafficConfig,
)
from agents.mappo_action_adapter import build_joint_actions_from_logits  # noqa: E402


def print_topology_dynamics(config: EnvironmentConfig, horizon: int = 180) -> None:
    probe_env = LEORoutingEnv(config)
    probe_env.reset(seed=7)

    prev_edge_index = probe_env.current_snapshot.edge_index.clone()
    prev_delays = probe_env.current_snapshot.neighbor_distances_km.clone()
    edge_change_steps: list[int] = []
    delay_change_steps: list[int] = []
    edge_counts = {int(prev_edge_index.shape[1])}

    for step in range(1, horizon + 1):
        snapshot = probe_env._build_snapshot(step)
        edge_counts.add(int(snapshot.edge_index.shape[1]))
        if not torch.equal(snapshot.edge_index, prev_edge_index):
            edge_change_steps.append(step)
        if not torch.allclose(snapshot.neighbor_distances_km, prev_delays, atol=1e-4, rtol=1e-4):
            delay_change_steps.append(step)
        prev_edge_index = snapshot.edge_index.clone()
        prev_delays = snapshot.neighbor_distances_km.clone()

    first_edge_change = edge_change_steps[0] if edge_change_steps else None
    first_delay_change = delay_change_steps[0] if delay_change_steps else None
    print(f"topology first edge change step: {first_edge_change}")
    print(f"topology first delay change step: {first_delay_change}")
    print(f"topology edge-count set over first {horizon} steps: {sorted(edge_counts)}")


def build_random_mappo_actions(env: LEORoutingEnv) -> dict[str, torch.Tensor]:
    num_nodes = env.current_snapshot.num_nodes
    max_neighbors = env.config.topology.max_neighbors
    traffic_logits = torch.randn(num_nodes, max_neighbors)
    wpq_logits = torch.randn(num_nodes, env.num_classes)
    aqm_logits = torch.randn(num_nodes, 3)
    return build_joint_actions_from_logits(
        traffic_logits,
        wpq_logits,
        aqm_logits,
        env.current_snapshot.valid_neighbor_mask,
        pmax_upper_bound=0.1,
    )


def run_mappo_smoke_test(config: EnvironmentConfig) -> None:
    mappo_env = LEORoutingEnv(config)
    mappo_env.reset(seed=7)
    first_actions = build_random_mappo_actions(mappo_env)
    _, first_reward, _, first_info = mappo_env.step(first_actions)
    second_actions = build_random_mappo_actions(mappo_env)
    _, second_reward, _, second_info = mappo_env.step(second_actions)
    print(f"MAPPO traffic row sums sample: {first_actions['traffic'][:3].sum(dim=1).tolist()}")
    print(f"MAPPO WPQ weights sample: {first_actions['weights'][:3].tolist()}")
    print(f"MAPPO AQM sample: {first_actions['aqm'][:3].tolist()}")
    print(
        "MAPPO step smoke: "
        f"step1_reward={first_reward:.4f} step1_served={first_info['served_packets'].sum().item():.4f} "
        f"step2_reward={second_reward:.4f} step2_served={second_info['served_packets'].sum().item():.4f} "
        f"step2_aqm_max={second_info['drop_probabilities'].max().item():.4f}"
    )


def main() -> None:
    config = EnvironmentConfig(
        topology=TopologyConfig(
            num_planes=6,
            sats_per_plane=8,
            max_neighbors=4,
            use_visibility=True,
        ),
        traffic=TrafficConfig(
            num_classes=3,
            queue_capacities=(100, 100, 100),
            priority_levels=(0, 1, 2),
            arrival_rates_pps=(8.0, 5.0, 3.0),
            stochastic_arrivals=False,
        ),
        reward=RewardConfig(
            weights=(0.5, 0.3, 0.2),
            delay_metric="average",
        ),
        slot_duration_s=1.0,
        max_steps=20,
    )

    env = LEORoutingEnv(config)
    obs = env.reset(seed=7)
    print(f"reset obs shape: {tuple(obs.shape)}")
    print(f"source nodes: {env.source_nodes.tolist()}")
    print(f"destination nodes: {env.destination_nodes.tolist()}")
    print(f"edge_index shape: {tuple(env.current_snapshot.edge_index.shape)}")
    print(f"mean degree: {env.current_snapshot.valid_neighbor_mask.sum(dim=1).float().mean().item():.3f}")
    print_topology_dynamics(config, horizon=180)

    run_mappo_smoke_test(config)

    common_source = int(env.source_nodes[0].item())
    first_delivery_step = None
    first_drop_step = None
    for step_idx in range(config.max_steps):
        edge_before = env.current_snapshot.edge_index.clone()
        delays_before = env.current_snapshot.neighbor_distances_km.clone()
        actions = env.greedy_shortest_path_actions()
        obs, reward, done, info = env.step(actions)
        edge_changed = not torch.equal(env.current_snapshot.edge_index, edge_before)
        delay_changed = not torch.allclose(
            env.current_snapshot.neighbor_distances_km,
            delays_before,
            atol=1e-4,
            rtol=1e-4,
        )
        queue_total = info["queue_lengths"].sum().item()
        delivered = info["delivered_packets"].sum().item()
        aqm_max = info["drop_probabilities"].max().item()
        overflow = info["overflow_drops"].sum().item()
        remaining_energy = info["remaining_energy"].mean().item()
        delay_ms = info["costs"]["delay_average_ms"]
        energy_j = info["costs"]["energy"]
        loss = info["costs"]["loss_surrogate"]
        source_queue = info["queue_lengths"][common_source].tolist()
        source_drop = info["drop_probabilities"][common_source].tolist()
        if first_delivery_step is None and delivered > 0.0:
            first_delivery_step = step_idx + 1
        if first_drop_step is None and (aqm_max > 0.0 or overflow > 0.0):
            first_drop_step = step_idx + 1
        print(
            f"step={step_idx + 1} reward={reward:.4f} "
            f"edge_changed={edge_changed} "
            f"delay_changed={delay_changed} "
            f"delay_ms={delay_ms:.4f} energy_j={energy_j:.4f} "
            f"loss={loss:.4f} queue_total={queue_total:.4f} "
            f"delivered={delivered:.4f} served={info['served_packets'].sum().item():.4f} "
            f"aqm_max={aqm_max:.4f} overflow={overflow:.4f} "
            f"source_q={source_queue} source_drop={source_drop} "
            f"mean_energy_left={remaining_energy:.2f}"
        )
        if done:
            break

    print(f"first delivery step: {first_delivery_step}")
    print(f"first drop step: {first_drop_step}")

    graph_state = env.export_graph_state(node_feature_dim=10)
    print(f"state_x shape: {tuple(graph_state['state_x'].shape)}")
    print(f"neighbor_indices shape: {tuple(graph_state['neighbor_indices'].shape)}")


if __name__ == "__main__":
    main()
