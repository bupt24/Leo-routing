from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.remote_sensing_agent_env import (  # noqa: E402
    CANDIDATE_FEATURE_DIM,
    LOCAL_OBS_DIM,
    RemoteSensingAgentEnv,
    RemoteSensingEnvCache,
)
from env.remote_sensing_scenario import load_scenario_config  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"


class RemoteSensingAgentEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_scenario_config(CONFIG_PATH)
        cls.config = replace(base, drain_slots=1)
        cls.cache = RemoteSensingEnvCache.build(cls.config, time_slots=1)

    def make_env(self) -> RemoteSensingAgentEnv:
        return RemoteSensingAgentEnv(
            self.config,
            time_slots=1,
            drain_slots=1,
            cache=self.cache,
        )

    def test_reset_returns_at_most_four_concurrent_task_observations(self) -> None:
        env = self.make_env()
        observations = env.reset(episode=2, seed=42)

        self.assertIsInstance(observations, dict)
        assert observations is not None
        self.assertGreater(len(observations), 0)
        self.assertLessEqual(len(observations), 4)
        self.assertEqual(set(observations), set(env.active_routes))
        for task_id, observation in observations.items():
            self.assertEqual(tuple(observation["local_obs"].shape), (LOCAL_OBS_DIM,))
            self.assertEqual(
                tuple(observation["candidate_features"].shape),
                (env.action_dim, CANDIDATE_FEATURE_DIM),
            )
            self.assertLessEqual(int(observation["action_mask"].sum().item()), 4)
            task = env.active_routes[task_id].task
            self.assertIsNotNone(task.destination_gs)
            self.assertLessEqual(task.candidate_rls_count, 3)
            self.assertLessEqual(len(task.candidate_access_edges), 4)

    def test_cached_resets_are_reproducible_for_every_task(self) -> None:
        first = self.make_env()
        second = self.make_env()
        first_obs = first.reset(episode=3, seed=17)
        second_obs = second.reset(episode=3, seed=17)

        assert first_obs is not None and second_obs is not None
        self.assertEqual(set(first_obs), set(second_obs))
        for task_id in first_obs:
            for key in ("local_obs", "candidate_features", "action_mask", "global_state"):
                self.assertTrue(torch.equal(first_obs[task_id][key], second_obs[task_id][key]))
            first_task = first.active_routes[task_id].task
            second_task = second.active_routes[task_id].task
            self.assertEqual(first_task.source_rls.label, second_task.source_rls.label)
            self.assertEqual(first_task.destination_gs.label, second_task.destination_gs.label)
            self.assertEqual(first_task.data_size_mb, second_task.data_size_mb)

    def test_candidate_features_react_to_queue_capacity_and_other_injection(self) -> None:
        env = self.make_env()
        observations = env.reset(episode=4, seed=9)
        assert observations is not None
        task_id = next(iter(observations))
        route = env.active_routes[task_id]
        candidate = env.current_candidates[task_id][0]
        env.active_routes = {task_id: route}
        before = env._candidate_features(route, candidate)

        destination = env.scenario.node_by_id[candidate.dst_id]
        class_idx = route.task.traffic_class
        capacity = float(env.config.queue_capacities[class_idx])
        env.scenario.queue_lengths[destination.local_id, class_idx] = capacity * 0.4
        env.slot_injected_packets[(candidate.dst_id, class_idx)] = capacity * 0.2
        after = env._candidate_features(route, candidate)

        self.assertGreater(after[10], before[10], "queue occupancy must increase")
        self.assertLess(after[11], before[11], "remaining capacity must decrease")
        self.assertGreater(after[12], before[12], "other-RLS injection pressure must increase")

    def test_step_accepts_a_task_action_mapping(self) -> None:
        env = self.make_env()
        observations = env.reset(episode=5, seed=21)
        assert observations is not None
        actions = {
            task_id: int(torch.where(obs["action_mask"])[0][0].item())
            for task_id, obs in observations.items()
        }

        next_observations, rewards, done, info = env.step(actions)

        self.assertEqual(set(rewards), set(actions))
        self.assertIsInstance(info["decision_done"], dict)
        self.assertIsInstance(info["terminal_reward_updates"], dict)
        self.assertTrue(done or next_observations is None or isinstance(next_observations, dict))


if __name__ == "__main__":
    unittest.main()
