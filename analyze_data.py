"""Analyze offline transitions with episode-aware trajectory reconstruction."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "offline_trajectories.pkl"


def load_samples(data_path: Path) -> list[dict]:
    with open(data_path, "rb") as f:
        return pickle.load(f)


def summarize(values: Iterable[float], precision: int = 1) -> str:
    values = list(values)
    if not values:
        return "n/a"
    avg = sum(values) / len(values)
    return f"min={min(values):.{precision}f}, max={max(values):.{precision}f}, avg={avg:.{precision}f}"


def group_samples_by_episode(samples: list[dict]) -> tuple[dict[int, list[dict]], int]:
    episode_groups: dict[int, list[dict]] = defaultdict(list)
    missing_episode_metadata = 0

    for idx, sample in enumerate(samples):
        if "episode_id" in sample and "timestep" in sample:
            enriched = sample
        else:
            missing_episode_metadata += 1
            enriched = dict(sample)
            enriched["episode_id"] = int(sample.get("snapshot_id", idx))
            enriched["timestep"] = int(sample.get("timestep", 0))

        episode_id = int(enriched["episode_id"])
        episode_groups[episode_id].append(enriched)

    for episode_id, episode_samples in episode_groups.items():
        episode_samples.sort(
            key=lambda item: (
                int(item["timestep"]),
                int(item["curr_idx"]),
                int(item["action"]),
            )
        )
        episode_groups[episode_id] = episode_samples

    return dict(sorted(episode_groups.items())), missing_episode_metadata


def group_episode_by_timestep(episode_samples: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for sample in episode_samples:
        grouped[int(sample["timestep"])].append(sample)
    return dict(sorted(grouped.items()))


def format_timestep_preview(timestep_samples: list[dict], max_items: int = 4) -> str:
    preview_items = []
    for sample in timestep_samples[:max_items]:
        preview_items.append(
            f"{sample['curr_idx']}->{sample['next_curr_idx']} "
            f"(a={sample['action']}, r={sample['reward']:.2f}, done={int(sample['done'] > 0.5)})"
        )
    if len(timestep_samples) > max_items:
        preview_items.append("...")
    return ", ".join(preview_items)


def print_episode_preview(episode_id: int, episode_samples: list[dict], max_timesteps: int = 5) -> None:
    timestep_groups = group_episode_by_timestep(episode_samples)
    success_steps = sorted(
        {
            int(sample["timestep"])
            for sample in episode_samples
            if float(sample["done"]) > 0.5
        }
    )
    destinations = sorted({int(sample["dest_idx"]) for sample in episode_samples})
    print(
        f"Episode {episode_id}: "
        f"transitions={len(episode_samples)}, "
        f"timesteps={len(timestep_groups)}, "
        f"destinations={destinations}, "
        f"success={bool(success_steps)}"
    )
    if success_steps:
        print(f"  success timesteps: {success_steps}")

    for timestep, timestep_samples in list(timestep_groups.items())[:max_timesteps]:
        print(f"  t={timestep}: {format_timestep_preview(timestep_samples)}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    samples = load_samples(DATA_PATH)
    if not samples:
        print("数据文件为空，没有可分析的 transition。")
        return

    episode_groups, fallback_count = group_samples_by_episode(samples)

    print("训练数据分析:")
    print(f"数据文件: {DATA_PATH}")
    print(f"总转换数: {len(samples)}")
    print(f"到达终点的转换数: {sum(1 for sample in samples if float(sample['done']) > 0.5)}")
    print(f"episode 数: {len(episode_groups)}")

    if fallback_count > 0:
        print(
            "警告: 当前数据中有 "
            f"{fallback_count} 条样本缺少 episode_id/timestep，"
            "已回退到基于 snapshot_id 的伪分组。建议重新生成 offline_trajectories.pkl。"
        )

    successful_episodes = [
        episode_id
        for episode_id, episode_samples in episode_groups.items()
        if any(float(sample["done"]) > 0.5 for sample in episode_samples)
    ]
    transition_counts = [len(episode_samples) for episode_samples in episode_groups.values()]
    timestep_counts = [
        len(group_episode_by_timestep(episode_samples))
        for episode_samples in episode_groups.values()
    ]
    active_nodes_per_timestep = [
        len(episode_samples) / max(len(group_episode_by_timestep(episode_samples)), 1)
        for episode_samples in episode_groups.values()
    ]
    reward_sums = [
        sum(float(sample["reward"]) for sample in episode_samples)
        for episode_samples in episode_groups.values()
    ]

    print(f"成功 episode 数: {len(successful_episodes)}")
    print(f"transition/episode: {summarize(transition_counts)}")
    print(f"timestep/episode: {summarize(timestep_counts)}")
    print(f"active transitions/timestep: {summarize(active_nodes_per_timestep)}")
    print(f"episode reward sum: {summarize(reward_sums, precision=2)}")

    print("\n前10个样本:")
    for idx, sample in enumerate(samples[:10]):
        print(
            f"  [{idx}] ep={sample.get('episode_id', 'NA')}, "
            f"t={sample.get('timestep', 'NA')}, "
            f"curr={sample['curr_idx']}, dst={sample['dest_idx']}, "
            f"next={sample['next_curr_idx']}, action={sample['action']}, "
            f"reward={sample['reward']:.2f}, done={sample['done']}"
        )

    print("\nEpisode 预览:")
    preview_episode_ids = list(episode_groups.keys())[:3]
    for episode_id in preview_episode_ids:
        print_episode_preview(episode_id, episode_groups[episode_id])


if __name__ == "__main__":
    main()
