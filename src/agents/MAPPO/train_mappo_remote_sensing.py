#!/usr/bin/env python3
"""Train shared-policy MAPPO on concurrent multi-source remote-sensing tasks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import torch


CURRENT_DIR = Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[2]
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.remote_sensing_agent_env import (  # noqa: E402
    ENV_SCHEMA_VERSION,
    RemoteSensingAgentEnv,
    RemoteSensingEnvCache,
)
from agents.MAPPO.vanilla_mappo import (  # noqa: E402
    PPOConfig,
    RolloutBuffer,
    VanillaMAPPOPolicy,
    VanillaMAPPOTrainer,
)
from env.remote_sensing_scenario import load_scenario_config  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "remote_sensing_mappo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO for joint RLS access and CLS routing.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--time-slots", type=int, default=10)
    parser.add_argument("--eval-time-slots", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--ttl-cap", type=int, default=10)
    parser.add_argument("--drain-slots", type=int, default=-1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--env-cache", choices=["off", "memory"], default="memory")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_observations(
    observations: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("Cannot stack an empty observation list.")
    return {
        "local_obs": torch.stack([obs["local_obs"] for obs in observations]).to(device),
        "candidate_features": torch.stack([obs["candidate_features"] for obs in observations]).to(device),
        "action_mask": torch.stack([obs["action_mask"] for obs in observations]).bool().to(device),
        "global_state": torch.stack([obs["global_state"] for obs in observations]).to(device),
    }


def run_episode(
    *,
    env: RemoteSensingAgentEnv,
    policy: VanillaMAPPOPolicy,
    episode: int,
    seed: int,
    slot_offset: int,
    device: torch.device,
    deterministic: bool,
    collect_buffer: bool,
) -> tuple[dict[str, Any], RolloutBuffer]:
    buffer = RolloutBuffer()
    observation_map = env.reset(episode=episode, seed=seed, slot_offset=slot_offset)
    policy_was_training = policy.training
    if deterministic:
        policy.eval()
    while observation_map:
        task_ids = sorted(observation_map)
        batch = stack_observations([observation_map[task_id] for task_id in task_ids], device)
        actions, log_probs, values = policy.act_batch(batch, deterministic=deterministic)
        action_map = dict(zip(task_ids, actions))
        next_map, rewards, done, info = env.step(action_map)
        if collect_buffer:
            decision_done = info.get("decision_done", {})
            for index, task_id in enumerate(task_ids):
                buffer.add(
                    obs=observation_map[task_id],
                    action=actions[index],
                    log_prob=log_probs[index],
                    value=values[index],
                    reward=float(rewards.get(task_id, 0.0)),
                    done=bool(decision_done.get(task_id, task_id not in (next_map or {}))),
                    task_id=task_id,
                )
            for task_id, reward_delta in info.get("terminal_reward_updates", {}).items():
                buffer.add_terminal_reward(task_id, float(reward_delta))
        observation_map = next_map
        if done:
            break
    if deterministic and policy_was_training:
        policy.train()
    return env.episode_summary(), buffer


def summarize_routes(routes: list[dict[str, Any]], episode: int, seed: int) -> dict[str, Any]:
    successes = [route for route in routes if route.get("success")]

    def average(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0

    return {
        "episode": episode,
        "seed": seed,
        "total_tasks": len(routes),
        "success_count": len(successes),
        "failed_count": len(routes) - len(successes),
        "success_rate": len(successes) / len(routes) if routes else 0.0,
        "deadline_meeting_rate": len(successes) / len(routes) if routes else 0.0,
        "avg_delay_success_ms": average("delay_ms_total", successes),
        "avg_delay_actual_all_ms": average("delay_ms_total", routes),
        "avg_cls_delay_success_ms": average("delay_ms_cls_route", successes),
        "avg_energy_success_j": average("energy_j_total", successes),
        "avg_loss_success": average("loss_total", successes),
        "avg_reward_all": average("reward_total", routes),
    }


def run_parallel_rollout(
    *,
    envs: list[RemoteSensingAgentEnv],
    policy: VanillaMAPPOPolicy,
    episode: int,
    base_seed: int,
    slot_offset: int,
    device: torch.device,
) -> tuple[dict[str, Any], RolloutBuffer]:
    buffer = RolloutBuffer()
    observation_maps = [
        env.reset(
            episode=episode,
            seed=int(base_seed) + (episode - 1) * len(envs) + env_index,
            slot_offset=slot_offset,
        )
        for env_index, env in enumerate(envs)
    ]
    while any(observation_maps):
        references: list[tuple[int, str]] = []
        observations: list[dict[str, torch.Tensor]] = []
        for env_index, observation_map in enumerate(observation_maps):
            for task_id in sorted(observation_map or {}):
                references.append((env_index, task_id))
                observations.append(observation_map[task_id])
        batch = stack_observations(observations, device)
        actions, log_probs, values = policy.act_batch(batch, deterministic=False)
        actions_by_env: list[dict[str, int]] = [{} for _ in envs]
        batch_index_by_ref: dict[tuple[int, str], int] = {}
        for index, (env_index, task_id) in enumerate(references):
            actions_by_env[env_index][task_id] = actions[index]
            batch_index_by_ref[(env_index, task_id)] = index

        for env_index, env in enumerate(envs):
            if not observation_maps[env_index]:
                continue
            current_map = observation_maps[env_index] or {}
            next_map, rewards, _done, info = env.step(actions_by_env[env_index])
            decision_done = info.get("decision_done", {})
            for task_id in sorted(current_map):
                index = batch_index_by_ref[(env_index, task_id)]
                trajectory_id = f"env{env_index}:{task_id}"
                buffer.add(
                    obs=current_map[task_id],
                    action=actions[index],
                    log_prob=log_probs[index],
                    value=values[index],
                    reward=float(rewards.get(task_id, 0.0)),
                    done=bool(decision_done.get(task_id, task_id not in (next_map or {}))),
                    task_id=trajectory_id,
                )
            for task_id, reward_delta in info.get("terminal_reward_updates", {}).items():
                buffer.add_terminal_reward(f"env{env_index}:{task_id}", float(reward_delta))
            observation_maps[env_index] = next_map
    routes = [route for env in envs for route in env.routes]
    seed = int(base_seed) + (episode - 1) * len(envs)
    return summarize_routes(routes, episode, seed), buffer


@torch.no_grad()
def evaluate_policy(
    policy: VanillaMAPPOPolicy,
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    episode: int,
) -> dict[str, float]:
    summaries: list[dict[str, Any]] = []
    eval_slots = int(args.eval_time_slots or args.time_slots)
    for index in range(max(int(args.eval_episodes), 0)):
        env = RemoteSensingAgentEnv(
            config,
            time_slots=eval_slots,
            max_cls_hops=args.ttl_cap,
            drain_slots=None if args.drain_slots < 0 else args.drain_slots,
            device="cpu",
        )
        summary, _buffer = run_episode(
            env=env,
            policy=policy,
            episode=episode,
            seed=int(args.base_seed) + 100_000 + index,
            slot_offset=0,
            device=device,
            deterministic=True,
            collect_buffer=False,
        )
        summaries.append(summary)
    if not summaries:
        return {"eval_reward": 0.0, "eval_delay": 0.0, "eval_success_rate": 0.0}
    return {
        "eval_reward": sum(float(row["avg_reward_all"]) for row in summaries) / len(summaries),
        "eval_delay": sum(float(row["avg_delay_success_ms"]) for row in summaries) / len(summaries),
        "eval_success_rate": sum(float(row["success_rate"]) for row in summaries) / len(summaries),
    }


def save_checkpoint(
    path: Path,
    policy: VanillaMAPPOPolicy,
    trainer: VanillaMAPPOTrainer,
    args: argparse.Namespace,
    episode: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": policy.checkpoint_metadata(),
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "episode": int(episode),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_training_curves(rows: list[dict[str, Any]], output_dir: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    if not rows:
        return ""
    episodes = [row["episode"] for row in rows]
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(episodes, [row["avg_reward_all"] for row in rows])
    axes[1].plot(episodes, [row["avg_delay_success_ms"] for row in rows])
    axes[2].plot(episodes, [row["success_rate"] for row in rows])
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Delay (ms)")
    axes[2].set_ylabel("Success")
    axes[2].set_xlabel("Episode")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
    figure.tight_layout()
    path = output_dir / "mappo_training_curves.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return str(path)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.time_slots <= 0 or args.num_envs <= 0:
        raise SystemExit("--episodes, --time-slots and --num-envs must be positive.")
    device = choose_device(args.device)
    config = load_scenario_config(args.config)
    cache = (
        RemoteSensingEnvCache.build(config, time_slots=args.time_slots, slot_offset=0)
        if args.env_cache == "memory"
        else None
    )
    envs = [
        RemoteSensingAgentEnv(
            config,
            time_slots=args.time_slots,
            max_cls_hops=args.ttl_cap,
            drain_slots=None if args.drain_slots < 0 else args.drain_slots,
            cache=cache,
            device="cpu",
        )
        for _ in range(args.num_envs)
    ]
    env = envs[0]
    policy = VanillaMAPPOPolicy(
        local_obs_dim=env.local_obs_dim,
        candidate_feature_dim=env.candidate_feature_dim,
        global_state_dim=env.global_state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        env_schema_version=ENV_SCHEMA_VERSION,
    ).to(device)
    trainer = VanillaMAPPOTrainer(
        policy,
        PPOConfig(
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
        ),
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "mappo_training_metrics.csv"
    rows: list[dict[str, Any]] = []
    best_eval_reward = float("-inf")
    best_written = False
    for episode in range(1, args.episodes + 1):
        policy.train()
        if len(envs) == 1:
            summary, buffer = run_episode(
                env=env,
                policy=policy,
                episode=episode,
                seed=args.base_seed + episode - 1,
                slot_offset=0,
                device=device,
                deterministic=False,
                collect_buffer=True,
            )
        else:
            summary, buffer = run_parallel_rollout(
                envs=envs,
                policy=policy,
                episode=episode,
                base_seed=args.base_seed,
                slot_offset=0,
                device=device,
            )
        update = trainer.update(buffer, device)
        eval_metrics = {"eval_reward": "", "eval_delay": "", "eval_success_rate": ""}
        did_eval = args.eval_episodes > 0 and (
            episode == args.episodes or episode % max(args.eval_interval, 1) == 0
        )
        if did_eval:
            eval_metrics = evaluate_policy(policy, config, args, device, episode)
        row = {"episode": episode, "steps": len(buffer), **summary, **update, **eval_metrics}
        rows.append(row)
        fieldnames = list(dict.fromkeys(key for item in rows for key in item))
        write_csv(metrics_path, rows, fieldnames)
        if episode == args.episodes or episode % max(args.checkpoint_interval, 1) == 0:
            save_checkpoint(latest_path, policy, trainer, args, episode, row)
        if did_eval and float(eval_metrics["eval_reward"]) > best_eval_reward:
            best_eval_reward = float(eval_metrics["eval_reward"])
            save_checkpoint(best_path, policy, trainer, args, episode, row)
            best_written = True
        if args.progress_interval > 0 and (
            episode == 1 or episode == args.episodes or episode % args.progress_interval == 0
        ):
            print(
                f"Episode {episode}/{args.episodes}: agents={summary['total_tasks']} steps={len(buffer)} "
                f"reward={summary['avg_reward_all']:.4f} delay={summary['avg_delay_success_ms']:.4f} "
                f"success={summary['success_rate']:.4f} loss={update['total_loss']:.4f}",
                flush=True,
            )
    plot_path = "" if args.no_plot else plot_training_curves(rows, output_dir)
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "env_schema_version": ENV_SCHEMA_VERSION,
                "episodes": args.episodes,
                "time_slots": args.time_slots,
                "drain_slots": config.drain_slots if args.drain_slots < 0 else args.drain_slots,
                "action_dim": env.action_dim,
                "latest_checkpoint": str(latest_path),
                "best_checkpoint": str(best_path) if best_written else "",
                "metrics_csv": str(metrics_path),
                "plot": plot_path,
                "last_episode": rows[-1] if rows else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output directory: {output_dir}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "evaluate_policy",
    "run_episode",
    "run_parallel_rollout",
    "stack_observations",
]
