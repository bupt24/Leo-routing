"""Shared command-line runner for Random01-04."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_random_baseline import (  # noqa: E402
    RandomBaselineConfig,
    run_random_baseline,
    write_random_outputs,
)
from env.remote_sensing_scenario import load_scenario_config  # noqa: E402


def parse_args(variant: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {variant.title()} in the concurrent remote-sensing scenario."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"),
    )
    parser.add_argument("--time-slots", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ttl-margin", type=int, default=3)
    parser.add_argument("--ttl-cap", type=int, default=10)
    parser.add_argument("--drain-slots", type=int, default=None)
    if variant in {"random03", "random04"}:
        parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / f"remote_sensing_{variant}_baseline"),
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def run_cli(variant: str) -> None:
    args = parse_args(variant)
    scenario_config = load_scenario_config(args.config)
    random_config = RandomBaselineConfig(
        seed=args.random_seed,
        ttl_margin=args.ttl_margin,
        ttl_cap=args.ttl_cap,
        drain_slots=args.drain_slots,
        max_attempts=getattr(args, "max_attempts", 1),
        variant=variant,
    )
    routes, metrics, topology_edges = run_random_baseline(
        scenario_config,
        num_time_slots=args.time_slots,
        random_config=random_config,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    write_random_outputs(output_dir, routes, metrics, topology_edges)
    print(f"Output directory: {output_dir}")
    print(f"Tasks: {output_dir / 'tasks.csv'}")
    print(f"Slot queues: {output_dir / 'slot_queues.csv'}")
    print(f"Topology: {output_dir / 'topology.csv'}")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(
        f"{variant} summary: tasks={metrics['task_count']} "
        f"success={metrics['success_count']} "
        f"success_rate={metrics['success_rate']:.4f} "
        f"deadline_rate={metrics['deadline_meeting_rate']:.4f} "
        f"throughput_mbps={metrics['throughput_mbps']:.4f}"
    )


__all__ = ["parse_args", "run_cli"]
