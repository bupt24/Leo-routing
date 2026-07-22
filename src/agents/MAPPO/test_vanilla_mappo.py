from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.vanilla_mappo import (  # noqa: E402
    PPOConfig,
    RolloutBuffer,
    VanillaMAPPOPolicy,
    VanillaMAPPOTrainer,
    build_policy_from_checkpoint_payload,
    compute_gae,
)


class VanillaMAPPOTest(unittest.TestCase):
    def test_actor_masks_invalid_actions(self) -> None:
        policy = VanillaMAPPOPolicy(
            local_obs_dim=4,
            candidate_feature_dim=3,
            global_state_dim=5,
            action_dim=4,
            hidden_dim=16,
        )
        obs = {
            "local_obs": torch.randn(4),
            "candidate_features": torch.randn(4, 3),
            "action_mask": torch.tensor([True, False, True, False]),
            "global_state": torch.randn(5),
        }

        logits = policy.forward_actor(obs)

        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertLess(float(logits[0, 1].item()), -1e8)
        self.assertLess(float(logits[0, 3].item()), -1e8)

    def test_act_batch_respects_masks_for_deterministic_actions(self) -> None:
        policy = VanillaMAPPOPolicy(
            local_obs_dim=4,
            candidate_feature_dim=3,
            global_state_dim=5,
            action_dim=3,
            hidden_dim=16,
        )
        with torch.no_grad():
            for param in policy.parameters():
                param.zero_()
        obs = {
            "local_obs": torch.randn(2, 4),
            "candidate_features": torch.randn(2, 3, 3),
            "action_mask": torch.tensor(
                [
                    [False, True, True],
                    [False, False, True],
                ]
            ),
            "global_state": torch.randn(2, 5),
        }

        actions, log_probs, values = policy.act_batch(obs, deterministic=True)

        self.assertEqual(actions, [1, 2])
        self.assertEqual(tuple(log_probs.shape), (2,))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_gae_is_finite(self) -> None:
        rewards = torch.tensor([1.0, 0.5, -0.2])
        values = torch.tensor([0.1, 0.2, 0.3])
        dones = torch.tensor([0.0, 1.0, 1.0])

        advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)

        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())

    def test_ppo_update_changes_parameters(self) -> None:
        torch.manual_seed(1)
        policy = VanillaMAPPOPolicy(
            local_obs_dim=4,
            candidate_feature_dim=3,
            global_state_dim=5,
            action_dim=3,
            hidden_dim=16,
        )
        trainer = VanillaMAPPOTrainer(
            policy,
            PPOConfig(update_epochs=2, minibatch_size=4, entropy_coef=0.0),
        )
        buffer = RolloutBuffer()
        for idx in range(8):
            obs = {
                "local_obs": torch.randn(4),
                "candidate_features": torch.randn(3, 3),
                "action_mask": torch.tensor([True, True, False]),
                "global_state": torch.randn(5),
            }
            action, log_prob, value = policy.act(obs, deterministic=False)
            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=1.0 if idx % 2 == 0 else -0.5,
                done=idx % 3 == 2,
            )

        before = [param.detach().clone() for param in policy.parameters()]
        metrics = trainer.update(buffer, torch.device("cpu"))
        after = list(policy.parameters())

        self.assertTrue(torch.isfinite(torch.tensor(list(metrics.values()))).all())
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(before, after)))

    def test_rollout_groups_transitions_by_task_and_backfills_terminal_reward(self) -> None:
        buffer = RolloutBuffer()
        observation = {
            "local_obs": torch.zeros(1),
            "candidate_features": torch.zeros(1, 1),
            "action_mask": torch.tensor([True]),
            "global_state": torch.zeros(1),
        }
        for task_id, reward in (("a", 1.0), ("b", 10.0), ("a", 2.0), ("b", 20.0)):
            buffer.add(
                observation,
                0,
                torch.tensor(0.0),
                torch.tensor(0.0),
                reward,
                False,
                task_id=task_id,
            )
        self.assertTrue(buffer.add_terminal_reward("a", 5.0))
        data = buffer.tensors(torch.device("cpu"))

        self.assertEqual(data["rewards"].tolist(), [1.0, 7.0, 10.0, 20.0])
        self.assertEqual(data["dones"].tolist(), [0.0, 1.0, 0.0, 0.0])

    def test_legacy_checkpoint_reports_explicit_incompatibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "incompatible"):
            build_policy_from_checkpoint_payload(
                {
                    "model_config": {
                        "local_obs_dim": 4,
                        "candidate_feature_dim": 3,
                        "global_state_dim": 5,
                        "action_dim": 3,
                        "hidden_dim": 16,
                    },
                    "model_state_dict": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
