"""Vanilla MAPPO components for remote-sensing CLS next-hop routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_VALUE = -1e9
ENV_SCHEMA_VERSION = 2
ENVIRONMENT_MODE = "multi_source_concurrent_v2"


def _as_batch(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    if tensor.dim() == dims - 1:
        return tensor.unsqueeze(0)
    return tensor


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.bool()
    if not mask.any(dim=-1).all():
        raise ValueError("Every policy row must have at least one valid action.")
    return logits.masked_fill(~mask, MASK_VALUE)


class CandidateEdgeActor(nn.Module):
    def __init__(
        self,
        local_obs_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        candidate_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_obs = _as_batch(local_obs, 2)
        candidate_features = _as_batch(candidate_features, 3)
        local_hidden = self.local_encoder(local_obs)
        edge_hidden = self.edge_encoder(candidate_features)
        expanded_local = local_hidden.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        scores = self.scorer(torch.cat([expanded_local, edge_hidden], dim=-1)).squeeze(-1)
        if action_mask is not None:
            scores = masked_logits(scores, _as_batch(action_mask, 2))
        return scores


class GlobalStateCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        global_state = _as_batch(global_state, 2)
        return self.value_net(global_state).squeeze(-1)


class VanillaMAPPOPolicy(nn.Module):
    def __init__(
        self,
        local_obs_dim: int,
        candidate_feature_dim: int,
        global_state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        env_schema_version: int = ENV_SCHEMA_VERSION,
    ) -> None:
        super().__init__()
        self.local_obs_dim = int(local_obs_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.env_schema_version = int(env_schema_version)
        self.actor = CandidateEdgeActor(local_obs_dim, candidate_feature_dim, hidden_dim)
        self.critic = GlobalStateCritic(global_state_dim, hidden_dim)

    def forward_actor(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(
            obs["local_obs"],
            obs["candidate_features"],
            obs["action_mask"],
        )

    def forward_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(obs["global_state"])

    @torch.no_grad()
    def act_batch(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        logits = self.forward_actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.forward_value(obs)
        return [int(action.item()) for action in actions], log_probs, values

    @torch.no_grad()
    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        actions, log_probs, values = self.act_batch(obs, deterministic=deterministic)
        return actions[0], log_probs[0], values[0]

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        candidate_features: torch.Tensor,
        action_mask: torch.Tensor,
        global_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(local_obs, candidate_features, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(global_state)
        return log_probs, entropy, values

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "env_schema_version": self.env_schema_version,
            "environment_mode": ENVIRONMENT_MODE,
            "local_obs_dim": self.local_obs_dim,
            "candidate_feature_dim": self.candidate_feature_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }


@dataclass
class RolloutBuffer:
    local_obs: list[torch.Tensor] = field(default_factory=list)
    candidate_features: list[torch.Tensor] = field(default_factory=list)
    action_mask: list[torch.Tensor] = field(default_factory=list)
    global_state: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.local_obs.clear()
        self.candidate_features.clear()
        self.action_mask.clear()
        self.global_state.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.task_ids.clear()

    def add(
        self,
        obs: dict[str, torch.Tensor],
        action: int,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        task_id: str = "",
    ) -> None:
        self.local_obs.append(obs["local_obs"].detach())
        self.candidate_features.append(obs["candidate_features"].detach())
        self.action_mask.append(obs["action_mask"].detach())
        self.global_state.append(obs["global_state"].detach())
        self.actions.append(int(action))
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.task_ids.append(str(task_id))

    def add_terminal_reward(self, task_id: str, reward_delta: float) -> bool:
        for index in range(len(self.task_ids) - 1, -1, -1):
            if self.task_ids[index] == str(task_id):
                self.rewards[index] += float(reward_delta)
                self.dones[index] = True
                return True
        return False

    @staticmethod
    def _stack_to_device(tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        stacked = torch.stack(tensors)
        if stacked.device != device:
            stacked = stacked.to(device)
        return stacked

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        trajectory_order: list[str] = []
        grouped: dict[str, list[int]] = {}
        for index, task_id in enumerate(self.task_ids):
            if task_id not in grouped:
                grouped[task_id] = []
                trajectory_order.append(task_id)
            grouped[task_id].append(index)
        order = [index for task_id in trajectory_order for index in grouped[task_id]]

        def ordered(values: list[Any]) -> list[Any]:
            return [values[index] for index in order]

        return {
            "local_obs": self._stack_to_device(ordered(self.local_obs), device),
            "candidate_features": self._stack_to_device(ordered(self.candidate_features), device),
            "action_mask": self._stack_to_device(ordered(self.action_mask), device).bool(),
            "global_state": self._stack_to_device(ordered(self.global_state), device),
            "actions": torch.tensor(ordered(self.actions), dtype=torch.long, device=device),
            "old_log_probs": self._stack_to_device(ordered(self.log_probs), device).float(),
            "values": self._stack_to_device(ordered(self.values), device).float(),
            "rewards": torch.tensor(ordered(self.rewards), dtype=torch.float32, device=device),
            "dones": torch.tensor(ordered(self.dones), dtype=torch.float32, device=device),
        }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.tensor(0.0, dtype=rewards.dtype, device=rewards.device)
    next_value = torch.tensor(0.0, dtype=values.dtype, device=values.device)
    for step in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[step]
        delta = rewards[step] + float(gamma) * next_value * not_done - values[step]
        last_advantage = delta + float(gamma) * float(gae_lambda) * not_done * last_advantage
        advantages[step] = last_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 0.5
    update_epochs: int = 2
    minibatch_size: int = 2048


class VanillaMAPPOTrainer:
    def __init__(self, policy: VanillaMAPPOPolicy, config: PPOConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(config.lr))

    def update(self, buffer: RolloutBuffer, device: torch.device) -> dict[str, float]:
        if len(buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0,
            }

        data = buffer.tensors(device)
        with torch.no_grad():
            advantages, returns = compute_gae(
                data["rewards"],
                data["values"],
                data["dones"],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        num_steps = data["actions"].shape[0]
        minibatch_size = max(1, min(int(self.config.minibatch_size), num_steps))
        metrics: dict[str, float] = {}
        for _epoch in range(max(1, int(self.config.update_epochs))):
            indices = torch.randperm(num_steps, device=device)
            for start in range(0, num_steps, minibatch_size):
                idx = indices[start : start + minibatch_size]
                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    data["local_obs"][idx],
                    data["candidate_features"][idx],
                    data["action_mask"][idx],
                    data["global_state"][idx],
                    data["actions"][idx],
                )
                ratio = torch.exp(new_log_probs - data["old_log_probs"][idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_ratio),
                    1.0 + float(self.config.clip_ratio),
                ) * advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[idx])
                entropy_loss = entropy.mean()
                total_loss = (
                    policy_loss
                    + float(self.config.value_coef) * value_loss
                    - float(self.config.entropy_coef) * entropy_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config.max_grad_norm))
                self.optimizer.step()

                metrics = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                }
        return metrics


def build_policy_from_checkpoint_payload(payload: dict[str, Any]) -> VanillaMAPPOPolicy:
    metadata = payload["model_config"]
    schema_version = int(metadata.get("env_schema_version", 1))
    environment_mode = str(metadata.get("environment_mode", "legacy_single_task"))
    if schema_version != ENV_SCHEMA_VERSION or environment_mode != ENVIRONMENT_MODE:
        raise ValueError(
            f"Checkpoint environment mode {environment_mode!r} / schema v{schema_version} is "
            f"incompatible with {ENVIRONMENT_MODE!r} / v{ENV_SCHEMA_VERSION}; "
            "retrain the policy with the multi-source environment."
        )
    policy = VanillaMAPPOPolicy(
        local_obs_dim=int(metadata["local_obs_dim"]),
        candidate_feature_dim=int(metadata["candidate_feature_dim"]),
        global_state_dim=int(metadata["global_state_dim"]),
        action_dim=int(metadata["action_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        env_schema_version=schema_version,
    )
    policy.load_state_dict(payload["model_state_dict"])
    return policy


__all__ = [
    "CandidateEdgeActor",
    "ENVIRONMENT_MODE",
    "ENV_SCHEMA_VERSION",
    "GlobalStateCritic",
    "PPOConfig",
    "RolloutBuffer",
    "VanillaMAPPOPolicy",
    "VanillaMAPPOTrainer",
    "build_policy_from_checkpoint_payload",
    "compute_gae",
    "masked_logits",
]
