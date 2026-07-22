#!/usr/bin/env python3
"""Evaluate a schema-v2 MAPPO checkpoint on concurrent remote-sensing tasks."""

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

from agents.MAPPO.remote_sensing_agent_env import RemoteSensingAgentEnv  # noqa: E402
from agents.MAPPO.train_mappo_remote_sensing import run_episode, summarize_routes  # noqa: E402
from agents.MAPPO.vanilla_mappo import build_policy_from_checkpoint_payload  # noqa: E402
from env.remote_sensing_scenario import load_scenario_config  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "remote_sensing_mappo_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-source MAPPO remote-sensing routing.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--time-slots", type=int, default=600)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--ttl-cap", type=int, default=10)
    parser.add_argument("--drain-slots", type=int, default=-1)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.time_slots <= 0:
        raise SystemExit("--episodes and --time-slots must be positive.")
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location=device)
    policy = build_policy_from_checkpoint_payload(payload).to(device)
    config = load_scenario_config(args.config)
    env = RemoteSensingAgentEnv(
        config,
        time_slots=args.time_slots,
        action_dim=policy.action_dim,
        max_cls_hops=args.ttl_cap,
        drain_slots=None if args.drain_slots < 0 else args.drain_slots,
        device="cpu",
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for index in range(args.episodes):
        episode = index + 1
        summary, _buffer = run_episode(
            env=env,
            policy=policy,
            episode=episode,
            seed=args.base_seed + index,
            slot_offset=0,
            device=device,
            deterministic=True,
            collect_buffer=False,
        )
        episode_rows.append(summary)
        route_rows.extend(env.routes)
        queue_rows.extend(env.downlink_queue_rows)
        print(
            f"Episode {episode}/{args.episodes}: delay={summary['avg_delay_success_ms']:.4f} "
            f"success={summary['success_rate']:.4f} reward={summary['avg_reward_all']:.4f}",
            flush=True,
        )
    route_csv = output_dir / "mappo_routes.csv"
    episode_csv = output_dir / "mappo_episode_metrics.csv"
    queue_csv = output_dir / "mappo_downlink_queues.csv"
    write_csv(route_csv, route_rows)
    write_csv(episode_csv, episode_rows)
    write_csv(queue_csv, queue_rows)
    overall = summarize_routes(route_rows, 0, args.base_seed)
    summary = {
        "checkpoint": str(checkpoint_path),
        "config": args.config,
        "episodes": args.episodes,
        "time_slots": args.time_slots,
        "drain_slots": config.drain_slots if args.drain_slots < 0 else args.drain_slots,
        "base_seed": args.base_seed,
        "ttl_cap": args.ttl_cap,
        "action_dim": policy.action_dim,
        "route_csv": str(route_csv),
        "episode_csv": str(episode_csv),
        "downlink_queue_csv": str(queue_csv),
        **overall,
    }
    summary_path = output_dir / "mappo_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Output directory: {output_dir}")
    print(f"Routes CSV: {route_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
