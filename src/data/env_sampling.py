"""
Shared helpers for sampling offline transitions directly from LEORoutingEnv.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch

from env.aqm_wpq import normalize_traffic_distribution
from env.leo_routing_env import (
    EnvironmentConfig,
    LEORoutingEnv,
    RewardConfig,
    TopologyConfig,
    TrafficConfig,
)


DEFAULT_NODE_FEATURE_DIM = 10
DEFAULT_NUM_CLASSES = 3
DEFAULT_QUEUE_CAPACITIES = (100, 100, 100)
DEFAULT_PRIORITY_LEVELS = (0, 1, 2)


def infer_walker_dimensions(max_sats: int) -> tuple[int, int]:
    max_sats = max(int(max_sats), 1)
    if max_sats <= 3:
        return 1, max_sats

    best_planes = 1
    best_sats_per_plane = max_sats
    best_score: tuple[int, int] | None = None

    for num_planes in range(2, max_sats + 1):
        sats_per_plane = max_sats // num_planes
        if sats_per_plane < 2:
            break
        used_sats = num_planes * sats_per_plane
        score = (max_sats - used_sats, abs(num_planes - sats_per_plane))
        if best_score is None or score < best_score:
            best_score = score
            best_planes = num_planes
            best_sats_per_plane = sats_per_plane

    return best_planes, best_sats_per_plane


def build_env_config(
    max_sats: int,
    action_dim: int,
    device: str | torch.device = "cpu",
    time_step_sec: float = 1.0,
    max_steps: int = 32,
) -> EnvironmentConfig:
    num_planes, sats_per_plane = infer_walker_dimensions(max_sats)
    max_neighbors = max(1, min(int(action_dim), 4))

    return EnvironmentConfig(
        topology=TopologyConfig(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            max_neighbors=max_neighbors,
            use_visibility=True,
        ),
        traffic=TrafficConfig(
            num_classes=DEFAULT_NUM_CLASSES,
            queue_capacities=DEFAULT_QUEUE_CAPACITIES,
            priority_levels=DEFAULT_PRIORITY_LEVELS,
            arrival_rates_pps=(8.0, 5.0, 3.0),
            total_packets=10_000,
            stochastic_arrivals=False,
        ),
        reward=RewardConfig(
            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            delay_metric="average",
        ),
        slot_duration_s=float(time_step_sec),
        initial_energy_j=100.0,
        max_end_to_end_delay_ms=200.0,
        max_steps=max_steps,
        default_aqm=(0.4, 0.8, 0.1),
        device=str(device),
    )


def build_env(
    max_sats: int,
    action_dim: int,
    device: str | torch.device = "cpu",
    time_step_sec: float = 1.0,
    max_steps: int = 32,
) -> LEORoutingEnv:
    return LEORoutingEnv(
        build_env_config(
            max_sats=max_sats,
            action_dim=action_dim,
            device=device,
            time_step_sec=time_step_sec,
            max_steps=max_steps,
        )
    )


def pad_graph_state_to_action_dim(
    graph_state: dict[str, torch.Tensor],
    action_dim: int,
) -> dict[str, torch.Tensor]:
    neighbor_indices = graph_state["neighbor_indices"]
    neighbor_delays = graph_state["neighbor_delays"]
    action_mask = graph_state["action_mask"]
    current_dim = neighbor_indices.shape[1]

    if action_dim == current_dim:
        return {
            "state_x": graph_state["state_x"],
            "edge_index": graph_state["edge_index"],
            "neighbor_indices": neighbor_indices,
            "neighbor_delays": neighbor_delays,
            "action_mask": action_mask,
        }

    if action_dim < current_dim:
        return {
            "state_x": graph_state["state_x"],
            "edge_index": graph_state["edge_index"],
            "neighbor_indices": neighbor_indices[:, :action_dim],
            "neighbor_delays": neighbor_delays[:, :action_dim],
            "action_mask": action_mask[:, :action_dim],
        }

    pad_width = action_dim - current_dim
    pad_indices = torch.full(
        (neighbor_indices.shape[0], pad_width),
        -1,
        dtype=neighbor_indices.dtype,
        device=neighbor_indices.device,
    )
    pad_delays = torch.zeros(
        (neighbor_delays.shape[0], pad_width),
        dtype=neighbor_delays.dtype,
        device=neighbor_delays.device,
    )
    pad_mask = torch.zeros(
        (action_mask.shape[0], pad_width),
        dtype=action_mask.dtype,
        device=action_mask.device,
    )
    return {
        "state_x": graph_state["state_x"],
        "edge_index": graph_state["edge_index"],
        "neighbor_indices": torch.cat([neighbor_indices, pad_indices], dim=1),
        "neighbor_delays": torch.cat([neighbor_delays, pad_delays], dim=1),
        "action_mask": torch.cat([action_mask, pad_mask], dim=1),
    }


def build_random_joint_actions(env: LEORoutingEnv) -> dict[str, torch.Tensor]:
    if env.current_snapshot is None:
        raise RuntimeError("Environment has not been reset.")

    num_nodes = env.current_snapshot.num_nodes
    max_neighbors = env.config.topology.max_neighbors

    traffic_logits = torch.rand((num_nodes, max_neighbors), dtype=torch.float32, device=env.device)
    traffic = normalize_traffic_distribution(
        traffic_logits,
        env.current_snapshot.valid_neighbor_mask,
    )
    weights = 0.1 + torch.rand((num_nodes, env.num_classes), dtype=torch.float32, device=env.device)
    alpha = torch.rand(num_nodes, dtype=torch.float32, device=env.device) * 0.3
    beta = 0.6 + torch.rand(num_nodes, dtype=torch.float32, device=env.device) * 0.3
    pmax = torch.rand(num_nodes, dtype=torch.float32, device=env.device) * 0.1
    aqm = torch.stack([alpha, beta, pmax], dim=1)
    return {"traffic": traffic, "weights": weights, "aqm": aqm}


def build_mixed_joint_actions(
    env: LEORoutingEnv,
    use_expert_policy: bool = True,
    expert_ratio: float = 1.0,
) -> dict[str, torch.Tensor]:
    if not use_expert_policy:
        return build_random_joint_actions(env)

    expert_actions = env.greedy_shortest_path_actions()
    if expert_ratio >= 1.0:
        return expert_actions

    random_actions = build_random_joint_actions(env)
    num_nodes = expert_actions["traffic"].shape[0]
    selector = (torch.rand((num_nodes, 1), device=env.device) < float(expert_ratio))
    selector_weights = selector.expand(-1, env.num_classes)
    selector_aqm = selector.expand(-1, 3)

    return {
        "traffic": torch.where(selector, expert_actions["traffic"], random_actions["traffic"]),
        "weights": torch.where(selector_weights, expert_actions["weights"], random_actions["weights"]),
        "aqm": torch.where(selector_aqm, expert_actions["aqm"], random_actions["aqm"]),
    }


def extract_discrete_actions(
    traffic: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    masked_traffic = traffic.clone()
    masked_traffic = torch.where(valid_mask, masked_traffic, torch.full_like(masked_traffic, -1.0))
    actions = masked_traffic.argmax(dim=1)
    chosen_values = masked_traffic.gather(1, actions.unsqueeze(1)).squeeze(1)
    has_action = valid_mask.any(dim=1) & (chosen_values > 0.0)
    invalid = torch.full_like(actions, -1)
    return torch.where(has_action, actions, invalid)


def collect_step_samples(
    env: LEORoutingEnv,
    node_feature_dim: int,
    action_dim: int,
    joint_actions: dict[str, torch.Tensor],
    snapshot_id: int,
    episode_id: int | None = None,
    timestep: int | None = None,
) -> list[dict[str, Any]]:
    if env.current_snapshot is None:
        raise RuntimeError("Environment has not been reset.")

    episode_id = int(snapshot_id if episode_id is None else episode_id)
    timestep = int(env.time_index if timestep is None else timestep)
    graph_state = pad_graph_state_to_action_dim(
        env.export_graph_state(node_feature_dim=node_feature_dim),
        action_dim=action_dim,
    )
    discrete_actions = extract_discrete_actions(
        joint_actions["traffic"],
        env.current_snapshot.valid_neighbor_mask,
    )
    active_nodes = torch.where(discrete_actions >= 0)[0]
    if active_nodes.numel() == 0:
        env.step(joint_actions)
        return []

    destination = int(env.destination_nodes[0].item())
    _, _, done, info = env.step(joint_actions)
    next_graph_state = pad_graph_state_to_action_dim(
        env.export_graph_state(node_feature_dim=node_feature_dim),
        action_dim=action_dim,
    )

    local_rewards = torch.as_tensor(info["local_rewards"], dtype=torch.float32)
    samples: list[dict[str, Any]] = []
    for node_id in active_nodes.tolist():
        action = int(discrete_actions[node_id].item())
        next_node = int(graph_state["neighbor_indices"][node_id, action].item())
        if next_node < 0:
            continue

        next_action_mask = next_graph_state["action_mask"][next_node].clone()
        done_flag = float(done or next_node == destination)
        if done_flag > 0.0:
            next_action_mask.zero_()

        samples.append(
            {
                "snapshot_id": snapshot_id,
                "episode_id": episode_id,
                "timestep": timestep,
                "state_x": graph_state["state_x"].detach().cpu(),
                "edge_index": graph_state["edge_index"].detach().cpu(),
                "neighbor_indices": graph_state["neighbor_indices"].detach().cpu(),
                "neighbor_delays": graph_state["neighbor_delays"].detach().cpu(),
                "next_state_x": next_graph_state["state_x"].detach().cpu(),
                "next_edge_index": next_graph_state["edge_index"].detach().cpu(),
                "next_neighbor_indices": next_graph_state["neighbor_indices"].detach().cpu(),
                "next_neighbor_delays": next_graph_state["neighbor_delays"].detach().cpu(),
                "curr_idx": node_id,
                "dest_idx": destination,
                "next_curr_idx": next_node,
                "next_dest_idx": destination,
                "action": action,
                "reward": float(local_rewards[node_id].item()),
                "done": done_flag,
                "action_mask": graph_state["action_mask"][node_id].detach().cpu(),
                "next_action_mask": next_action_mask.detach().cpu(),
            }
        )
    return samples


def build_batch_from_samples(
    samples: list[dict[str, Any]],
    batch_size: int,
    device: str | torch.device = "cpu",
    action_dim: int | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot build a batch from an empty sample list.")

    actual_batch_size = max(int(batch_size), 1)
    if len(samples) >= actual_batch_size:
        selected = random.sample(samples, actual_batch_size)
    else:
        selected = [random.choice(samples) for _ in range(actual_batch_size)]

    sample0 = selected[0]
    curr_idx = torch.tensor([s["curr_idx"] for s in selected], dtype=torch.long, device=device)
    dest_idx = torch.tensor([s["dest_idx"] for s in selected], dtype=torch.long, device=device)
    next_curr_idx = torch.tensor([s["next_curr_idx"] for s in selected], dtype=torch.long, device=device)
    next_dest_idx = torch.tensor([s.get("next_dest_idx", s["dest_idx"]) for s in selected], dtype=torch.long, device=device)
    actions = torch.tensor([s["action"] for s in selected], dtype=torch.long, device=device)
    rewards = torch.tensor([s["reward"] for s in selected], dtype=torch.float32, device=device)
    dones = torch.tensor([s["done"] for s in selected], dtype=torch.float32, device=device)
    action_mask = torch.stack([s["action_mask"] for s in selected])
    next_action_mask = torch.stack([s["next_action_mask"] for s in selected])

    if action_dim is not None:
        current_dim = int(action_mask.shape[1])
        target_dim = int(action_dim)
        if current_dim > target_dim:
            if (actions >= target_dim).any():
                raise ValueError(
                    "Sampled actions exceed the requested action_dim. "
                    "Please regenerate the offline dataset or use a compatible action_dim."
                )
            action_mask = action_mask[:, :target_dim]
            next_action_mask = next_action_mask[:, :target_dim]
        elif current_dim < target_dim:
            pad_width = target_dim - current_dim
            pad_mask = torch.zeros((action_mask.shape[0], pad_width), dtype=action_mask.dtype)
            action_mask = torch.cat([action_mask, pad_mask], dim=1)
            next_action_mask = torch.cat([next_action_mask, pad_mask.clone()], dim=1)

    action_mask = action_mask.to(device)
    next_action_mask = next_action_mask.to(device)

    return {
        "state": (
            sample0["state_x"].to(device),
            sample0["edge_index"].to(device),
            curr_idx,
            dest_idx,
        ),
        "next_state": (
            sample0.get("next_state_x", sample0["state_x"]).to(device),
            sample0.get("next_edge_index", sample0["edge_index"]).to(device),
            next_curr_idx,
            next_dest_idx,
        ),
        "action": actions,
        "reward": rewards,
        "done": dones,
        "action_mask": action_mask,
        "next_action_mask": next_action_mask,
    }


def build_eval_data(
    max_sats: int,
    action_dim: int,
    node_feature_dim: int = DEFAULT_NODE_FEATURE_DIM,
    eval_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    env = build_env(
        max_sats=max_sats,
        action_dim=action_dim,
        device="cpu",
        time_step_sec=1.0,
        max_steps=16,
    )
    env.reset(seed=seed)
    graph_state = pad_graph_state_to_action_dim(
        env.export_graph_state(node_feature_dim=node_feature_dim),
        action_dim=action_dim,
    )

    valid_nodes = torch.where(graph_state["action_mask"].sum(dim=1) > 0)[0]
    if valid_nodes.numel() == 0:
        valid_nodes = torch.arange(graph_state["state_x"].shape[0])

    if valid_nodes.numel() >= eval_size:
        indices = valid_nodes[torch.randperm(valid_nodes.numel())[:eval_size]]
    else:
        choices = torch.randint(0, valid_nodes.numel(), (eval_size,))
        indices = valid_nodes[choices]

    dest = int(env.destination_nodes[0].item())
    dest_idx = torch.full((eval_size,), dest, dtype=torch.long)

    return {
        "state_x": graph_state["state_x"].to(device),
        "edge_index": graph_state["edge_index"].to(device),
        "curr_idx": indices.to(device),
        "dest_idx": dest_idx.to(device),
        "action_mask": graph_state["action_mask"][indices].to(device),
        "neighbor_indices": graph_state["neighbor_indices"].to(device),
        "neighbor_delays": graph_state["neighbor_delays"].to(device),
    }
