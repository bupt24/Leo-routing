#!/usr/bin/env python3
"""Run the deterministic remote-sensing-driven CLS-layer routing scenario."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_scenario import (  # noqa: E402
    RemoteSensingScenario,
    load_scenario_config,
    write_scenario_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic remote-sensing-task-driven CLS-layer routing scenario."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"),
        help="Path to the remote sensing scenario YAML config.",
    )
    parser.add_argument(
        "--time-slots",
        type=int,
        default=1,
        help="Number of deterministic time slots to simulate.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ttl-cap", type=int, default=10)
    parser.add_argument("--drain-slots", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "outputs" / "remote_sensing_scenario"),
        help="Root output directory. A timestamped subdirectory will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_scenario_config(args.config)
    scenario = RemoteSensingScenario(config)
    result = scenario.run(
        num_time_slots=args.time_slots,
        random_seed=args.random_seed,
        ttl_cap=args.ttl_cap,
        drain_slots=args.drain_slots,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    write_scenario_outputs(result, output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Routes: {output_dir / 'routes.csv'}")
    print(f"Topology edges: {output_dir / 'topology_edges.csv'}")
    print(f"Nodes: {output_dir / 'nodes.csv'}")
    print(f"Metrics: {output_dir / 'metrics_summary.json'}")
    print(
        "Summary: "
        f"tasks={result.metrics_summary['task_count']} "
        f"success={result.metrics_summary['success_count']} "
        f"success_rate={result.metrics_summary['success_rate']:.4f} "
        f"deadline_rate={result.metrics_summary['deadline_meeting_rate']:.4f} "
        f"throughput_mbps={result.metrics_summary['throughput_mbps']:.4f}"
    )


if __name__ == "__main__":
    main()
