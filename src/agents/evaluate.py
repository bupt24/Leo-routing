"""
Environment-rollout evaluation for the offline GNN routing policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
CHECKPOINT_DIR = SRC_DIR.parent / "checkpoints"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.model import LEO_Routing_GNN_CQL
from data.env_sampling import (
    DEFAULT_PRIORITY_LEVELS,
    DEFAULT_QUEUE_CAPACITIES,
    infer_walker_dimensions,
    pad_graph_state_to_action_dim,
)
from env.aqm_wpq import normalize_traffic_distribution
from env.leo_routing_env import (
    EnvironmentConfig,
    LEORoutingEnv,
    RewardConfig,
    TopologyConfig,
    TrafficConfig,
)


def build_eval_env(
    args: Any,
    device: torch.device,
    max_steps: int | None = None,
) -> LEORoutingEnv:
    eval_max_steps = int(max_steps or getattr(args, "eval_max_steps", 32))
    action_dim = int(getattr(args, "action_dim", 4))
    configured_neighbors = int(getattr(args, "max_neighbors", 4))
    max_neighbors = max(1, min(action_dim, configured_neighbors, 4))
    dataset_name = getattr(args, "dataset", "real")

    if dataset_name == "walker":
        num_planes = int(getattr(args, "num_planes", 6))
        sats_per_plane = int(getattr(args, "sats_per_plane", 8))
        use_visibility = bool(getattr(args, "use_visibility", False))
    else:
        max_sats = int(
            getattr(args, "num_sats", 0)
            or getattr(args, "num_nodes", 0)
            or getattr(args, "num_planes", 6) * getattr(args, "sats_per_plane", 8)
        )
        num_planes, sats_per_plane = infer_walker_dimensions(max_sats)
        use_visibility = True

    config = EnvironmentConfig(
        topology=TopologyConfig(
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            altitude_km=float(getattr(args, "altitude_km", 550.0)),
            inclination_deg=float(getattr(args, "inclination_deg", 53.0)),
            max_neighbors=max_neighbors,
            use_visibility=use_visibility,
        ),
        traffic=TrafficConfig(
            num_classes=3,
            queue_capacities=DEFAULT_QUEUE_CAPACITIES,
            priority_levels=DEFAULT_PRIORITY_LEVELS,
            arrival_rates_pps=(8.0, 5.0, 3.0),
            total_packets=10_000,
            stochastic_arrivals=False,
        ),
        reward=RewardConfig(),
        slot_duration_s=1.0,
        initial_energy_j=100.0,
        max_steps=eval_max_steps,
        max_end_to_end_delay_ms=200.0,
        default_aqm=(0.4, 0.8, 0.1),
        device=str(device),
    )
    return LEORoutingEnv(config)


def build_model_joint_actions(
    model: LEO_Routing_GNN_CQL,
    env: LEORoutingEnv,
    node_feature_dim: int,
    action_dim: int,
) -> dict[str, torch.Tensor]:
    if env.current_snapshot is None or env.destination_nodes is None:
        raise RuntimeError("Environment has not been reset.")

    graph_state = pad_graph_state_to_action_dim(
        env.export_graph_state(node_feature_dim=node_feature_dim),
        action_dim=action_dim,
    )
    num_nodes = graph_state["state_x"].shape[0]
    device = graph_state["state_x"].device
    current_idx = torch.arange(num_nodes, device=device, dtype=torch.long)
    target = int(torch.mode(env.destination_nodes).values.item())
    dest_idx = torch.full((num_nodes,), target, dtype=torch.long, device=device)

    q_values = model(
        graph_state["state_x"],
        graph_state["edge_index"],
        current_idx,
        dest_idx,
    )
    action_mask = graph_state["action_mask"] > 0.0
    masked_q = q_values.masked_fill(~action_mask, -1e9)
    chosen_actions = masked_q.argmax(dim=1)

    joint_actions = env.default_joint_actions()
    traffic = torch.zeros_like(joint_actions["traffic"])
    active_nodes = action_mask.any(dim=1) & (current_idx != dest_idx)
    if active_nodes.any():
        active_idx = torch.where(active_nodes)[0]
        traffic[active_idx, chosen_actions[active_idx]] = 1.0
    joint_actions["traffic"] = normalize_traffic_distribution(
        traffic,
        env.current_snapshot.valid_neighbor_mask,
    )
    return joint_actions


def rollout_episode(
    model: LEO_Routing_GNN_CQL,
    env: LEORoutingEnv,
    node_feature_dim: int,
    action_dim: int,
    seed: int,
) -> dict[str, float]:
    env.reset(seed=seed)
    episode_return = 0.0
    episode_delay = 0.0
    episode_energy = 0.0
    episode_loss = 0.0
    episode_success = False

    done = False
    while not done:
        joint_actions = build_model_joint_actions(
            model=model,
            env=env,
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
        )
        _, reward, done, info = env.step(joint_actions)
        step_metrics = info["step_metrics"]
        episode_return += float(reward)
        episode_delay += float(step_metrics["delay"])
        episode_energy += float(step_metrics["energy"])
        episode_loss += float(step_metrics["loss"])
        episode_success = episode_success or bool(step_metrics["success"])

    return {
        "return": episode_return,
        "delay": episode_delay,
        "energy": episode_energy,
        "loss": episode_loss,
        "success": 1.0 if episode_success else 0.0,
    }


def evaluate_policy(
    model: LEO_Routing_GNN_CQL,
    env: LEORoutingEnv,
    device: torch.device,
    node_feature_dim: int,
    action_dim: int,
    num_episodes: int = 16,
    seed: int = 42,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    episode_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for episode_idx in range(num_episodes):
            metrics = rollout_episode(
                model=model,
                env=env,
                node_feature_dim=node_feature_dim,
                action_dim=action_dim,
                seed=seed + episode_idx,
            )
            episode_metrics.append(metrics)

    if was_training:
        model.train()

    if not episode_metrics:
        return {
            "mean_return": 0.0,
            "mean_delay": 0.0,
            "mean_energy": 0.0,
            "mean_loss": 0.0,
            "success_rate": 0.0,
        }

    return {
        "mean_return": float(sum(item["return"] for item in episode_metrics) / len(episode_metrics)),
        "mean_delay": float(sum(item["delay"] for item in episode_metrics) / len(episode_metrics)),
        "mean_energy": float(sum(item["energy"] for item in episode_metrics) / len(episode_metrics)),
        "mean_loss": float(sum(item["loss"] for item in episode_metrics) / len(episode_metrics)),
        "success_rate": float(sum(item["success"] for item in episode_metrics) / len(episode_metrics)),
    }


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = build_eval_env(args, device=device, max_steps=args.eval_max_steps)
    model = LEO_Routing_GNN_CQL(
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        dropout=args.dropout,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    metrics = evaluate_policy(
        model=model,
        env=env,
        device=device,
        node_feature_dim=args.node_feature_dim,
        action_dim=args.action_dim,
        num_episodes=args.eval_episodes,
        seed=args.eval_seed,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / "evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        "Evaluation | "
        f"return={metrics['mean_return']:.4f} | "
        f"delay={metrics['mean_delay']:.4f} | "
        f"energy={metrics['mean_energy']:.4f} | "
        f"loss={metrics['mean_loss']:.4f} | "
        f"success_rate={metrics['success_rate']:.4f}"
    )
    print(f"评估结果已保存至: {save_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the offline GNN routing policy with environment rollout.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "leo_routing_best.pt"),
    )
    parser.add_argument("--dataset", choices=["synthetic", "leo", "walker", "real", "precomputed"], default="real")
    parser.add_argument("--num-sats", type=int, default=120)
    parser.add_argument("--num-nodes", type=int, default=128)
    parser.add_argument("--num-planes", type=int, default=6)
    parser.add_argument("--sats-per-plane", type=int, default=10)
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--inclination-deg", type=float, default=53.0)
    parser.add_argument("--use-visibility", action="store_true")
    parser.add_argument("--node-feature-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-max-steps", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=42)
    evaluate_checkpoint(parser.parse_args())
