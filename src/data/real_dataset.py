"""
Online offline-dataset sampler that now steps LEORoutingEnv directly.

The class name is preserved for compatibility with the existing training script.
"""

from __future__ import annotations

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
    build_mixed_joint_actions,
    collect_step_samples,
)


class RealTLEOfflineDataset:
    """
    Compatibility wrapper that now samples batches from the paper-style routing
    environment instead of the previous hand-written surrogate topology.

    The TLE-related constructor arguments are preserved so existing callers keep
    working, but the trajectory source is the new environment model.
    """

    def __init__(
        self,
        tle_file: str | None = None,
        shell: str = "shell1",
        max_sats: int = 100,
        num_gateways: int = 6,
        node_feature_dim: int = 10,
        action_dim: int = 4,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
        total_batches: int = 10,
        time_step_sec: float = 1.0,
        use_expert_policy: bool = True,
        expert_ratio: float = 0.7,
    ):
        del tle_file
        del shell
        del num_gateways

        self.max_sats = max_sats
        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device
        self.total_batches = total_batches
        self.time_step_sec = time_step_sec
        self.use_expert_policy = use_expert_policy
        self.expert_ratio = expert_ratio

        self.env = build_env(
            max_sats=max_sats,
            action_dim=action_dim,
            device="cpu",
            time_step_sec=time_step_sec,
            max_steps=32,
        )
        self.episode_id = 0
        self.snapshot_id = 0
        self.timestep = 0

    def __iter__(self):
        for batch_idx in range(self.total_batches):
            if self.env.current_snapshot is None or self.env.done:
                self.env.reset(seed=self.episode_id)
                self.timestep = 0

            joint_actions = build_mixed_joint_actions(
                self.env,
                use_expert_policy=self.use_expert_policy,
                expert_ratio=self.expert_ratio,
            )
            step_samples = collect_step_samples(
                env=self.env,
                node_feature_dim=self.node_feature_dim,
                action_dim=self.action_dim,
                joint_actions=joint_actions,
                snapshot_id=self.snapshot_id,
                episode_id=self.episode_id,
                timestep=self.timestep,
            )
            self.snapshot_id += 1
            self.timestep += 1
            if self.env.done:
                self.episode_id += 1

            if not step_samples:
                continue

            yield build_batch_from_samples(
                step_samples,
                batch_size=self.batch_size,
                device=self.device,
                action_dim=self.action_dim,
            )


if __name__ == "__main__":
    dataset = RealTLEOfflineDataset(
        max_sats=60,
        batch_size=32,
        total_batches=3,
        use_expert_policy=True,
        expert_ratio=0.7,
    )

    print("\n测试数据集迭代:")
    for i, batch in enumerate(dataset):
        print(f"Batch {i + 1}:")
        print(f"  State X shape: {batch['state'][0].shape}")
        print(f"  Next State X shape: {batch['next_state'][0].shape}")
        print(f"  Actions shape: {batch['action'].shape}")
        print(f"  Rewards: mean={batch['reward'].mean():.4f}, min={batch['reward'].min():.4f}")
        print(f"  Done ratio: {batch['done'].mean():.2%}")
