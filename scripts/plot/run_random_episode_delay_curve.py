#!/usr/bin/env python3
"""Run Random baseline episodes and plot episode-level delay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
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
from env.remote_sensing_random_baseline import (  # noqa: E402
    RANDOM_ALLOWED_LINK_TYPES,
    RandomBaselineConfig,
    compute_random_ttl,
)
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "remote_sensing_random_episode_curve"


@dataclass(frozen=True)
class SlotIndex:
    snapshot: ScenarioSnapshot
    targets: list[ScenarioNode]
    observation_edges_by_target_id: dict[int, list[ScenarioEdge]]
    access_edges_by_rls_id: dict[int, list[ScenarioEdge]]
    random_edges_by_src_id: dict[int, list[ScenarioEdge]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Random baseline episodes and plot one delay point per episode."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Scenario YAML config.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes.")
    parser.add_argument("--time-slots", type=int, default=600, help="Slots per episode.")
    parser.add_argument("--base-seed", type=int, default=42, help="Seed used by episode 1.")
    parser.add_argument("--deadline-ms", type=float, default=30000.0, help="Per-task timeout deadline.")
    parser.add_argument("--ttl-margin", type=int, default=3, help="TTL margin added to shortest hops.")
    parser.add_argument("--ttl-cap", type=int, default=10, help="Maximum Random routing TTL.")
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
    observation_edges_by_target_id: dict[int, list[ScenarioEdge]] = {}
    access_edges_by_rls_id: dict[int, list[ScenarioEdge]] = {}
    random_edges_by_src_id: dict[int, list[ScenarioEdge]] = {}

    for edge in snapshot.edges:
        if edge.link_type == LINK_OBSERVATION and edge.available and edge.is_data_flow_allowed:
            observation_edges_by_target_id.setdefault(edge.src_id, []).append(edge)
        elif edge.link_type == LINK_ACCESS and edge.available and edge.is_data_flow_allowed:
            access_edges_by_rls_id.setdefault(edge.src_id, []).append(edge)
        elif (
            edge.src_type == NODE_CLS
            and edge.dst_type in {NODE_CLS, NODE_ES}
            and edge.available
            and edge.is_data_flow_allowed
            and edge.link_type in RANDOM_ALLOWED_LINK_TYPES
        ):
            random_edges_by_src_id.setdefault(edge.src_id, []).append(edge)

    for edges in observation_edges_by_target_id.values():
        edges.sort(key=lambda edge: (edge.dst, edge.dst_id))
    for edges in access_edges_by_rls_id.values():
        edges.sort(key=lambda edge: (edge.dst, edge.dst_id))
    for edges in random_edges_by_src_id.values():
        edges.sort(key=lambda edge: (edge.dst, edge.dst_id, edge.link_type))

    return SlotIndex(
        snapshot=snapshot,
        targets=targets,
        observation_edges_by_target_id=observation_edges_by_target_id,
        access_edges_by_rls_id=access_edges_by_rls_id,
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


def shortest_hops_to_any_es_avoiding(
    slot_index: SlotIndex,
    start_cls_id: int,
    blocked_node_ids: set[int],
    max_hops: int | None,
) -> int | None:
    queue: list[tuple[int, int]] = [(start_cls_id, 0)]
    visited = {start_cls_id}
    head = 0
    while head < len(queue):
        node_id, hops = queue[head]
        head += 1
        if max_hops is not None and hops >= max_hops:
            continue
        for edge in slot_index.random_edges_by_src_id.get(node_id, []):
            if edge.dst_id in blocked_node_ids or edge.dst_id in visited:
                continue
            next_hops = hops + 1
            if edge.dst_type == NODE_ES:
                return next_hops
            if max_hops is not None and next_hops >= max_hops:
                continue
            visited.add(edge.dst_id)
            queue.append((edge.dst_id, next_hops))
    return None


def random_route_delay_to_es(
    slot_index: SlotIndex,
    start_cls_id: int,
    random_config: RandomBaselineConfig,
    rng: random.Random,
    link_config: LinkModelConfig,
) -> float | None:
    shortest_hops = shortest_hops_to_any_es(slot_index, start_cls_id)
    if shortest_hops is None:
        return None

    ttl = compute_random_ttl(shortest_hops, random_config.ttl_margin, random_config.ttl_cap)
    visited = {start_cls_id}
    current = start_cls_id
    total_delay_ms = 0.0

    for hop_index in range(ttl):
        remaining_hops_after_next = ttl - hop_index - 1
        next_edges = [
            edge for edge in slot_index.random_edges_by_src_id.get(current, [])
            if edge.dst_id not in visited
        ]
        feasible_edges: list[ScenarioEdge] = []
        for edge in next_edges:
            if edge.dst_type == NODE_ES:
                feasible_edges.append(edge)
                continue
            if remaining_hops_after_next <= 0:
                continue
            reachable_hops = shortest_hops_to_any_es_avoiding(
                slot_index,
                edge.dst_id,
                blocked_node_ids=visited,
                max_hops=remaining_hops_after_next,
            )
            if reachable_hops is not None:
                feasible_edges.append(edge)

        if not feasible_edges:
            return None

        edge = rng.choice(feasible_edges)
        total_delay_ms += single_packet_edge_delay_ms(edge, link_config)
        if total_delay_ms > random_config.deadline_ms:
            return None
        if edge.dst_type == NODE_ES:
            return total_delay_ms
        visited.add(edge.dst_id)
        current = edge.dst_id
    return None


def route_task_delay_ms(
    slot_index: SlotIndex,
    target: ScenarioNode,
    rng: random.Random,
    random_config: RandomBaselineConfig,
    link_config: LinkModelConfig,
) -> float | None:
    observation_edges = slot_index.observation_edges_by_target_id.get(target.node_id, [])
    if not observation_edges:
        return None
    observation_edge = rng.choice(observation_edges)

    access_edges = slot_index.access_edges_by_rls_id.get(observation_edge.dst_id, [])
    if not access_edges:
        return None
    access_edge = rng.choice(access_edges)

    cls_delay_ms = random_route_delay_to_es(
        slot_index,
        access_edge.dst_id,
        random_config=random_config,
        rng=rng,
        link_config=link_config,
    )
    if cls_delay_ms is None:
        return None

    return (
        single_packet_edge_delay_ms(observation_edge, link_config)
        + single_packet_edge_delay_ms(access_edge, link_config)
        + cls_delay_ms
    )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return ordered[lower_idx]
    lower = ordered[lower_idx]
    upper = ordered[upper_idx]
    return lower + (upper - lower) * (rank - lower_idx)


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
    slot_indices: list[SlotIndex],
    random_config: RandomBaselineConfig,
    link_config: LinkModelConfig,
) -> dict[str, Any]:
    slot_avg_delays: list[float] = []
    slot_p90_delays: list[float] = []
    success_count = 0
    total_count = 0
    delay_sum = 0.0
    task_index = 0

    for slot_index in slot_indices:
        slot_delays: list[float] = []
        for target in slot_index.targets:
            rng = random.Random(seed + task_index)
            delay_ms = route_task_delay_ms(
                slot_index,
                target,
                rng=rng,
                random_config=random_config,
                link_config=link_config,
            )
            total_count += 1
            task_index += 1
            if delay_ms is None:
                continue
            success_count += 1
            delay_sum += delay_ms
            slot_delays.append(delay_ms)

        slot_avg_delay = average(slot_delays)
        if slot_avg_delay is not None:
            slot_avg_delays.append(slot_avg_delay)
        slot_p90_delay = percentile(slot_delays, 90.0)
        if slot_p90_delay is not None:
            slot_p90_delays.append(slot_p90_delay)

    return {
        "episode": episode_number,
        "seed": seed,
        "avg_delay_ms": average(slot_avg_delays),
        "weighted_avg_delay_ms": delay_sum / success_count if success_count else None,
        "avg_p90_delay_ms": average(slot_p90_delays),
        "success_rate": success_count / total_count if total_count else None,
        "success_count": success_count,
        "failed_count": total_count - success_count,
    }


def write_episode_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "seed",
        "avg_delay_ms",
        "weighted_avg_delay_ms",
        "avg_p90_delay_ms",
        "success_rate",
        "success_count",
        "failed_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_csv_value(row.get(field)) for field in fieldnames})


def plot_episode_delay(rows: list[dict[str, Any]], plot_path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return f"Skipped plot: {exc}"

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = [int(row["episode"]) for row in rows]
    delays = [
        float(row["avg_delay_ms"]) if row.get("avg_delay_ms") is not None else math.nan
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        episodes,
        delays,
        color="#1f77b4",
        marker="^",
        markersize=3,
        linewidth=1.1,
        label="Random",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average End-to-End Delay (ms)")
    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
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
    random_config = RandomBaselineConfig(
        seed=args.base_seed,
        deadline_ms=args.deadline_ms,
        ttl_margin=args.ttl_margin,
        ttl_cap=args.ttl_cap,
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
                f"success_rate={format_csv_value(row['success_rate'])}",
                flush=True,
            )

    csv_path = output_dir / "episode_delay_curve.csv"
    plot_path = output_dir / "episode_delay_curve.png"
    summary_path = output_dir / "episode_summary.json"
    write_episode_csv(rows, csv_path)

    plot_message = None
    if not args.no_plot:
        plot_message = plot_episode_delay(rows, plot_path)

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
    summary = {
        "episodes": args.episodes,
        "time_slots_per_episode": args.time_slots,
        "base_seed": args.base_seed,
        "last_seed": args.base_seed + args.episodes - 1,
        "deadline_ms": args.deadline_ms,
        "ttl_margin": args.ttl_margin,
        "ttl_cap": args.ttl_cap,
        "avg_episode_delay_ms": avg_episode_delay,
        "avg_success_rate": avg_success_rate,
        "csv": str(csv_path),
        "plot": "" if args.no_plot or plot_message else str(plot_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Episode CSV: {csv_path}")
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
        f"avg_success_rate={format_csv_value(avg_success_rate)}"
    )


if __name__ == "__main__":
    main()
