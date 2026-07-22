#!/usr/bin/env python3
"""Run Random04 episodes and plot single-packet end-to-end delay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.link_model import LinkModelConfig, SPEED_OF_LIGHT_KM_PER_MS  # noqa: E402
from env.remote_sensing_scenario import (  # noqa: E402
    LINK_ACCESS,
    LINK_OBSERVATION,
    NODE_CLS,
    NODE_ES,
    NODE_TARGET,
    RemoteSensingScenario,
    ScenarioEdge,
    ScenarioNode,
    ScenarioSnapshot,
    load_scenario_config,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "remote_sensing_random04_episode_curve"


def load_random04_module():
    module_path = REPO_ROOT / "src" / "env" / "04remote_sensing_random_baseline.py"
    spec = importlib.util.spec_from_file_location("remote_sensing_random_baseline_04_episode", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Random04 baseline implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RANDOM04 = load_random04_module()


@dataclass(frozen=True)
class SlotIndex:
    snapshot: ScenarioSnapshot
    targets: list[ScenarioNode]
    random_edges_by_src_id: dict[int, list[ScenarioEdge]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Random04 episodes and plot one single-packet delay point per episode."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Scenario YAML config.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes.")
    parser.add_argument("--time-slots", type=int, default=600, help="Slots per episode.")
    parser.add_argument("--base-seed", type=int, default=42, help="Seed used by episode 1.")
    parser.add_argument("--deadline-ms", type=float, default=30000.0, help="Per-task timeout deadline.")
    parser.add_argument("--ttl-margin", type=int, default=3, help="TTL margin added to shortest hops.")
    parser.add_argument("--ttl-cap", type=int, default=10, help="Maximum Random routing TTL.")
    parser.add_argument("--max-attempts", type=int, default=8, help="Maximum naive-random attempts per task.")
    parser.add_argument("--avg-window", type=int, default=100, help="Episode window size for the averaged plot.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory. A timestamped subdirectory is created unless --output-dir is set.",
    )
    parser.add_argument("--output-dir", default="", help="Exact output directory.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print progress every N episodes. Use 0 to disable progress messages.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Only write CSV/JSON, skip PNG.")
    return parser.parse_args()


def build_slot_index(snapshot: ScenarioSnapshot) -> SlotIndex:
    targets = [node for node in snapshot.nodes if node.node_type == NODE_TARGET]
    random_edges_by_src_id: dict[int, list[ScenarioEdge]] = {}

    for edge in snapshot.edges:
        if (
            edge.src_type == NODE_CLS
            and edge.dst_type in {NODE_CLS, NODE_ES}
            and edge.available
            and edge.is_data_flow_allowed
            and edge.link_type in RANDOM04.RANDOM_ALLOWED_LINK_TYPES
        ):
            random_edges_by_src_id.setdefault(edge.src_id, []).append(edge)

    for edges in random_edges_by_src_id.values():
        edges.sort(key=lambda edge: (edge.dst, edge.dst_id, edge.link_type))

    return SlotIndex(
        snapshot=snapshot,
        targets=targets,
        random_edges_by_src_id=random_edges_by_src_id,
    )


def single_packet_edge_delay_ms(edge: ScenarioEdge, link_config: LinkModelConfig) -> float:
    if edge.link_type == LINK_OBSERVATION:
        return float(edge.delay_ms)
    if edge.capacity_bps <= 0.0:
        return math.inf
    prop_delay_ms = float(edge.distance_km) / SPEED_OF_LIGHT_KM_PER_MS
    tx_delay_ms = float(link_config.packet_size_bits) / float(edge.capacity_bps) * 1000.0
    return prop_delay_ms + tx_delay_ms


def shortest_hops_to_any_es(slot_index: SlotIndex, start_cls_id: int) -> int | None:
    queue: list[tuple[int, int]] = [(start_cls_id, 0)]
    visited = {start_cls_id}
    head = 0
    while head < len(queue):
        node_id, hops = queue[head]
        head += 1
        for edge in slot_index.random_edges_by_src_id.get(node_id, []):
            if edge.dst_type == NODE_ES:
                return hops + 1
            if edge.dst_id in visited:
                continue
            visited.add(edge.dst_id)
            queue.append((edge.dst_id, hops + 1))
    return None


def naive_random_cls_delay_to_es(
    slot_index: SlotIndex,
    start_cls_id: int,
    random_config: Any,
    rng: random.Random,
    link_config: LinkModelConfig,
) -> tuple[bool, float]:
    shortest_hops = shortest_hops_to_any_es(slot_index, start_cls_id)
    if shortest_hops is None:
        return False, 0.0

    ttl = RANDOM04.compute_random_ttl(shortest_hops, random_config.ttl_margin, random_config.ttl_cap)
    visited = {start_cls_id}
    current = start_cls_id
    total_delay_ms = 0.0

    for _hop_index in range(ttl):
        next_edges = [
            edge for edge in slot_index.random_edges_by_src_id.get(current, [])
            if edge.dst_id not in visited
        ]
        if not next_edges:
            return False, total_delay_ms

        edge = rng.choice(next_edges)
        total_delay_ms += single_packet_edge_delay_ms(edge, link_config)
        if total_delay_ms > random_config.deadline_ms:
            return False, total_delay_ms
        if edge.dst_type == NODE_ES:
            return True, total_delay_ms

        visited.add(edge.dst_id)
        current = edge.dst_id

    return False, total_delay_ms


def find_edge(snapshot: ScenarioSnapshot, src_label: str, dst_label: str, link_type: str) -> ScenarioEdge | None:
    for edge in snapshot.edges:
        if edge.src == src_label and edge.dst == dst_label and edge.link_type == link_type:
            return edge
    return None


def label_to_id(snapshot: ScenarioSnapshot, label: str) -> int | None:
    for node in snapshot.nodes:
        if node.label == label:
            return node.node_id
    return None


def retry_random04_task_delay_ms(
    scenario: RemoteSensingScenario,
    slot_index: SlotIndex,
    target: ScenarioNode,
    task_index: int,
    rng: random.Random,
    random_config: Any,
    link_config: LinkModelConfig,
) -> tuple[bool, float, int]:
    deterministic_route = scenario.route_task(slot_index.snapshot, target, task_index)
    if not deterministic_route.get("access_success"):
        return False, 0.0, 1

    rls_label = str(deterministic_route.get("rls_node") or "")
    cls_label = str(deterministic_route.get("selected_cls") or "")
    if not rls_label or not cls_label:
        return False, 0.0, 1

    observation_edge = find_edge(slot_index.snapshot, target.label, rls_label, LINK_OBSERVATION)
    access_edge = find_edge(slot_index.snapshot, rls_label, cls_label, LINK_ACCESS)
    cls_id = label_to_id(slot_index.snapshot, cls_label)
    if observation_edge is None or access_edge is None or cls_id is None:
        return False, 0.0, 1

    front_delay_ms = (
        single_packet_edge_delay_ms(observation_edge, link_config)
        + single_packet_edge_delay_ms(access_edge, link_config)
    )
    max_attempts = max(1, int(random_config.max_attempts))
    attempted_delay_ms = 0.0

    for attempt_index in range(max_attempts):
        cls_success, cls_delay_ms = naive_random_cls_delay_to_es(
            slot_index,
            cls_id,
            random_config=random_config,
            rng=rng,
            link_config=link_config,
        )
        attempted_delay_ms += front_delay_ms + cls_delay_ms
        if cls_success:
            return True, attempted_delay_ms, attempt_index + 1

    return False, attempted_delay_ms, max_attempts


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return value


def run_episode(
    episode_number: int,
    seed: int,
    scenario: RemoteSensingScenario,
    slot_indices: list[SlotIndex],
    random_config: Any,
    link_config: LinkModelConfig,
) -> dict[str, Any]:
    slot_avg_delays: list[float] = []
    success_count = 0
    total_count = 0
    delay_sum = 0.0
    attempt_sum = 0
    task_index = 0

    for slot_index in slot_indices:
        slot_delays: list[float] = []
        for target in slot_index.targets:
            rng = random.Random(seed + task_index)
            success, delay_ms, attempt_count = retry_random04_task_delay_ms(
                scenario,
                slot_index,
                target,
                task_index=task_index,
                rng=rng,
                random_config=random_config,
                link_config=link_config,
            )
            total_count += 1
            task_index += 1
            attempt_sum += attempt_count
            if not success:
                continue
            success_count += 1
            delay_sum += delay_ms
            slot_delays.append(delay_ms)

        slot_avg_delay = average(slot_delays)
        if slot_avg_delay is not None:
            slot_avg_delays.append(slot_avg_delay)

    return {
        "episode": episode_number,
        "seed": seed,
        "avg_delay_ms": average(slot_avg_delays),
        "weighted_avg_delay_ms": delay_sum / success_count if success_count else None,
        "success_rate": success_count / total_count if total_count else None,
        "success_count": success_count,
        "failed_count": total_count - success_count,
        "avg_attempt_count": attempt_sum / total_count if total_count else None,
    }


def build_window_rows(rows: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    window_size = max(1, int(window_size))
    windows: list[dict[str, Any]] = []
    for start in range(0, len(rows), window_size):
        window = rows[start:start + window_size]
        delays = [
            float(row["avg_delay_ms"])
            for row in window
            if row.get("avg_delay_ms") is not None
        ]
        success_rates = [
            float(row["success_rate"])
            for row in window
            if row.get("success_rate") is not None
        ]
        attempts = [
            float(row["avg_attempt_count"])
            for row in window
            if row.get("avg_attempt_count") is not None
        ]
        windows.append(
            {
                "episode_start": int(window[0]["episode"]),
                "episode_end": int(window[-1]["episode"]),
                "episode_midpoint": (int(window[0]["episode"]) + int(window[-1]["episode"])) / 2.0,
                "avg_delay_ms": average(delays),
                "avg_success_rate": average(success_rates),
                "avg_attempt_count": average(attempts),
                "episode_count": len(window),
            }
        )
    return windows


def write_csv(rows: list[dict[str, Any]], csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(row.get(field)) for field in fieldnames})


def plot_window_delay(window_rows: list[dict[str, Any]], plot_path: Path, label: str = "Random04") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return f"Skipped plot: {exc}"

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [float(row["episode_midpoint"]) for row in window_rows]
    delays = [
        float(row["avg_delay_ms"]) if row.get("avg_delay_ms") is not None else math.nan
        for row in window_rows
    ]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(
        episodes,
        delays,
        color="#1f77b4",
        marker="^",
        markersize=6,
        linewidth=1.6,
        label=label,
    )
    ax.set_xlim(0, 1000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average End-to-End Delay (ms)")
    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return None


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")
    if args.time_slots <= 0:
        raise SystemExit("--time-slots must be positive.")

    scenario_config = load_scenario_config(args.config)
    scenario = RemoteSensingScenario(scenario_config)
    random_config = RANDOM04.RetryRandomBaselineConfig(
        seed=args.base_seed,
        deadline_ms=args.deadline_ms,
        ttl_margin=args.ttl_margin,
        ttl_cap=args.ttl_cap,
        max_attempts=args.max_attempts,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prebuilding {args.time_slots} slot snapshots...", flush=True)
    slot_indices = [
        build_slot_index(scenario.build_snapshot(time_slot))
        for time_slot in range(args.time_slots)
    ]

    rows: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        episode_number = episode_index + 1
        seed = args.base_seed + episode_index
        row = run_episode(
            episode_number,
            seed,
            scenario,
            slot_indices,
            random_config,
            scenario.link_config,
        )
        rows.append(row)
        if args.progress_interval > 0 and (
            episode_number == 1
            or episode_number == args.episodes
            or episode_number % args.progress_interval == 0
        ):
            print(
                "Episode "
                f"{episode_number}/{args.episodes}: "
                f"avg_delay_ms={format_csv_value(row['avg_delay_ms'])} "
                f"success_rate={format_csv_value(row['success_rate'])} "
                f"avg_attempt_count={format_csv_value(row['avg_attempt_count'])}",
                flush=True,
            )

    window_rows = build_window_rows(rows, args.avg_window)

    episode_csv_path = output_dir / "random04_episode_end_to_end_delay.csv"
    avg_csv_path = output_dir / f"random04_episode_end_to_end_delay_avg{args.avg_window}.csv"
    plot_path = output_dir / f"random04_episode_end_to_end_delay_avg{args.avg_window}_y0.png"
    summary_path = output_dir / "episode_summary.json"

    write_csv(
        rows,
        episode_csv_path,
        [
            "episode",
            "seed",
            "avg_delay_ms",
            "weighted_avg_delay_ms",
            "success_rate",
            "success_count",
            "failed_count",
            "avg_attempt_count",
        ],
    )
    write_csv(
        window_rows,
        avg_csv_path,
        [
            "episode_start",
            "episode_end",
            "episode_midpoint",
            "avg_delay_ms",
            "avg_success_rate",
            "avg_attempt_count",
            "episode_count",
        ],
    )

    plot_message = None
    if not args.no_plot:
        plot_message = plot_window_delay(window_rows, plot_path)

    avg_episode_delay = average([
        float(row["avg_delay_ms"])
        for row in rows
        if row.get("avg_delay_ms") is not None
    ])
    avg_success_rate = average([
        float(row["success_rate"])
        for row in rows
        if row.get("success_rate") is not None
    ])
    avg_attempt_count = average([
        float(row["avg_attempt_count"])
        for row in rows
        if row.get("avg_attempt_count") is not None
    ])
    summary = {
        "episodes": args.episodes,
        "time_slots_per_episode": args.time_slots,
        "base_seed": args.base_seed,
        "last_seed": args.base_seed + args.episodes - 1,
        "deadline_ms": args.deadline_ms,
        "ttl_margin": args.ttl_margin,
        "ttl_cap": args.ttl_cap,
        "max_attempts": args.max_attempts,
        "avg_window": args.avg_window,
        "delay_metric": "single_packet_cumulative_retry_end_to_end_delay_ms",
        "avg_episode_delay_ms": avg_episode_delay,
        "avg_success_rate": avg_success_rate,
        "avg_attempt_count": avg_attempt_count,
        "episode_csv": str(episode_csv_path),
        "avg_csv": str(avg_csv_path),
        "plot": "" if args.no_plot or plot_message else str(plot_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Episode CSV: {episode_csv_path}")
    print(f"Averaged CSV: {avg_csv_path}")
    print(f"Summary JSON: {summary_path}")
    if args.no_plot:
        print("Plot: skipped by --no-plot")
    elif plot_message:
        print(f"Plot: {plot_message}")
    else:
        print(f"Delay plot: {plot_path}")
    print(
        "Overall: "
        f"episodes={args.episodes} "
        f"time_slots={args.time_slots} "
        f"avg_episode_delay_ms={format_csv_value(avg_episode_delay)} "
        f"avg_success_rate={format_csv_value(avg_success_rate)} "
        f"avg_attempt_count={format_csv_value(avg_attempt_count)}"
    )


if __name__ == "__main__":
    main()
