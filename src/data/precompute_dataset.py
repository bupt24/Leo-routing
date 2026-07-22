"""
Precompute offline trajectories by stepping the paper-style LEO routing environment.
"""

from __future__ import annotations

import os
import pickle
import random
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.env_sampling import (
    build_batch_from_samples,
    build_env,
    build_eval_data,
    build_mixed_joint_actions,
    collect_step_samples,
)


def generate_expert_trajectories(
    
    num_trajectories: int = 10_000,
    max_sats: int = 100,
    num_gateways: int = 6,
    node_feature_dim: int = 10,
    action_dim: int = 4,
    expert_ratio: float = 0.8,
    save_path: str | None = None,
    use_walker: bool = True,
):
    """
    Generate offline transitions directly from LEORoutingEnv.

    `num_gateways` and `use_walker` are kept for backward compatibility with the
    previous API. The new implementation always samples from the paper-style
    Walker/Grid+ environment.
    """
    del num_gateways
    if not use_walker:
        print("新实现统一从论文环境采样，已忽略 use_walker=False 并回退到 Walker/Grid+ 环境。")

    env = build_env(
        max_sats=max_sats,
        action_dim=action_dim,
        device="cpu",
        time_step_sec=1.0,
        max_steps=32,
    )

    all_data: list[dict] = []
    episode_id = 0
    snapshot_id = 0
    while len(all_data) < num_trajectories:
        env.reset(seed=episode_id)
        timestep = 0
        while not env.done and len(all_data) < num_trajectories:
            joint_actions = build_mixed_joint_actions(
                env,
                use_expert_policy=True,
                expert_ratio=expert_ratio,
            )
            step_samples = collect_step_samples(
                env=env,
                node_feature_dim=node_feature_dim,
                action_dim=action_dim,
                joint_actions=joint_actions,
                snapshot_id=snapshot_id,
                episode_id=episode_id,
                timestep=timestep,
            )
            if step_samples:
                remaining = num_trajectories - len(all_data)
                all_data.extend(step_samples[:remaining])
            snapshot_id += 1
            timestep += 1
        episode_id += 1

    print(f"共生成 {len(all_data)} 个环境采样样本")

    if save_path is None:
        save_path = str(SRC_DIR.parent / "data" / "offline_trajectories.pkl")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"数据已保存至: {save_path}")
    return all_data


class PrecomputedOfflineDataset:
    """
    Load precomputed transitions and batch them by snapshot so state/next-state
    graph structure stays consistent inside each training batch.
    """

    def __init__(
        self,
        data_path: str | None = None,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
        total_batches: int = 10,
        action_dim: int | None = None,
    ):
        if data_path is None:
            data_path = str(SRC_DIR.parent / "data" / "offline_trajectories.pkl")

        regenerate = False
        if not os.path.exists(data_path):
            regenerate = True
        else:
            with open(data_path, "rb") as f:
                loaded = pickle.load(f)
            if not loaded:
                regenerate = True
            else:
                first = loaded[0]
                regenerate = not (
                    isinstance(first, dict)
                    and "snapshot_id" in first
                    and "episode_id" in first
                    and "timestep" in first
                    and "next_state_x" in first
                    and "next_edge_index" in first
                )

        if regenerate:
            print("离线数据不存在或为旧格式，正在用新环境重新生成...")
            generate_expert_trajectories(save_path=data_path)

        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.grouped_samples: dict[int, list[dict]] = {}
        for sample in self.data:
            snapshot_id = int(sample["snapshot_id"])
            self.grouped_samples.setdefault(snapshot_id, []).append(sample)
        self.snapshot_ids = list(self.grouped_samples.keys())

        print(
            f"加载 {len(self.data)} 个预计算样本，"
            f"共 {len(self.snapshot_ids)} 个拓扑快照组"
        )

        self.batch_size = batch_size
        self.device = device
        self.total_batches = total_batches
        self.action_dim = action_dim

    def __iter__(self):
        random.shuffle(self.snapshot_ids)
        if not self.snapshot_ids:
            return

        for batch_idx in range(self.total_batches):
            snapshot_id = self.snapshot_ids[batch_idx % len(self.snapshot_ids)]
            snapshot_samples = self.grouped_samples[snapshot_id]
            yield build_batch_from_samples(
                snapshot_samples,
                batch_size=self.batch_size,
                device=self.device,
                action_dim=self.action_dim,
            )


def create_eval_from_precomputed(device: str = "cpu", action_dim: int = 4):
    """
    Create a fixed evaluation graph directly from the new environment.
    """
    return build_eval_data(
        max_sats=96,
        action_dim=action_dim,
        node_feature_dim=10,
        eval_size=64,
        device=device,
        seed=42,
    )


if __name__ == "__main__":
    generate_expert_trajectories(
        num_trajectories=20_000,
        max_sats=100,
        expert_ratio=0.8,
    )
