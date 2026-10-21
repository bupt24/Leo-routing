from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import random
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.MAPPO.remote_sensing_agent_env import (  # noqa: E402
    RemoteSensingAgentEnv,
    RemoteSensingEnvCache,
)
from env.channel_fading import (  # noqa: E402
    ChannelFadingConfig,
    CorrelatedLognormalFading,
    make_channel_link_key,
)
from env.link_model import compute_fspl, compute_link_rates_bps, compute_snr  # noqa: E402
from env.remote_sensing_scenario import (  # noqa: E402
    LINK_ACCESS,
    LINK_DOWNLINK,
    LINK_ISL,
    LINK_RLS_BACKUP,
    RemoteSensingScenario,
    RemoteSensingScenarioConfig,
    ScenarioSnapshot,
    _load_simple_yaml,
    load_scenario_config,
    scenario_config_from_dict,
)
from env.remote_sensing_task_core import MultiSourceTaskFactory  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
COMMUNICATION_TYPES = {LINK_ACCESS, LINK_ISL, LINK_DOWNLINK, LINK_RLS_BACKUP}


def capacity_signature(snapshots: list[ScenarioSnapshot]) -> list[tuple[object, ...]]:
    return [
        (
            snapshot.time_slot,
            edge.link_type,
            edge.src_type,
            edge.src,
            edge.dst_type,
            edge.dst,
            edge.capacity_bps,
        )
        for snapshot in snapshots
        for edge in snapshot.edges
        if edge.link_type in COMMUNICATION_TYPES
    ]


def task_signature(tasks: list[object]) -> list[tuple[object, ...]]:
    return [
        (
            task.task_id,
            task.profile_name,
            task.data_size_mb,
            task.packet_count,
            task.source_rls.label,
            task.destination_gs.label if task.destination_gs is not None else None,
        )
        for task in tasks
    ]


class CorrelatedLognormalFadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChannelFadingConfig(
            enabled=True,
            seed=2026,
            sigma_db=2.0,
            rho=0.95,
        )
        self.key = make_channel_link_key(
            LINK_ISL,
            ("cls", 2),
            ("cls", 7),
            undirected=True,
        )

    def test_seed_reproducibility_temporal_updates_and_gap_filling(self) -> None:
        first = CorrelatedLognormalFading(self.config)
        second = CorrelatedLognormalFading(self.config)
        first_values = [first.loss_db(self.key, slot) for slot in range(6)]
        second_values = [second.loss_db(self.key, slot) for slot in range(6)]

        self.assertEqual(first_values, second_values)
        self.assertTrue(all(math.isfinite(value) for value in first_values))
        self.assertTrue(any(a != b for a, b in zip(first_values, first_values[1:])))

        gap_process = CorrelatedLognormalFading(self.config)
        self.assertEqual(gap_process.loss_db(self.key, 5), first_values[5])

    def test_same_slot_is_idempotent_and_does_not_advance_rng(self) -> None:
        repeated = CorrelatedLognormalFading(self.config)
        reference = CorrelatedLognormalFading(self.config)
        slot_zero = repeated.loss_db(self.key, 0)
        for _ in range(5):
            self.assertEqual(repeated.loss_db(self.key, 0), slot_zero)
        reference.loss_db(self.key, 0)

        self.assertEqual(repeated.loss_db(self.key, 1), reference.loss_db(self.key, 1))

    def test_different_seed_changes_sequence(self) -> None:
        first = CorrelatedLognormalFading(self.config, seed=1)
        second = CorrelatedLognormalFading(self.config, seed=2)

        self.assertNotEqual(first.loss_db(self.key, 0), second.loss_db(self.key, 0))

    def test_link_key_direction_rules(self) -> None:
        forward = make_channel_link_key(
            LINK_ISL, ("cls", 1), ("cls", 3), undirected=True
        )
        reverse = make_channel_link_key(
            LINK_ISL, ("cls", 3), ("cls", 1), undirected=True
        )
        directed_forward = make_channel_link_key(
            LINK_ACCESS, ("rls", 1), ("cls", 3), undirected=False
        )
        directed_reverse = make_channel_link_key(
            LINK_ACCESS, ("cls", 3), ("rls", 1), undirected=False
        )

        self.assertEqual(forward, reverse)
        self.assertNotEqual(directed_forward, directed_reverse)

    def test_sampling_does_not_modify_global_rng_states(self) -> None:
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        try:
            process = CorrelatedLognormalFading(self.config)
            before_python = random.getstate()
            before_torch = torch.random.get_rng_state().clone()
            for slot in range(4):
                process.loss_db(self.key, slot)

            self.assertEqual(random.getstate(), before_python)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_torch))
        finally:
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)

    def test_fading_produces_finite_snr_and_rate(self) -> None:
        scenario_config = load_scenario_config(CONFIG_PATH)
        link_config = scenario_config.link_budgets.access
        distance = torch.tensor(4_000.0)
        available = torch.tensor(True)
        fspl = compute_fspl(distance, link_config.carrier_frequency_hz)
        process = CorrelatedLognormalFading(self.config)
        for slot in range(20):
            fading_db = process.loss_db(self.key, slot)
            snr = compute_snr(fspl, link_config, fading_loss_db=fading_db)
            _, _, rate = compute_link_rates_bps(
                distance,
                available,
                link_config,
                fading_loss_db=fading_db,
            )
            self.assertTrue(torch.isfinite(snr).all().item())
            self.assertTrue(torch.isfinite(rate).all().item())
            self.assertGreater(float(rate.item()), 0.0)


class RemoteSensingChannelFadingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_scenario_config(CONFIG_PATH)
        cls.base = replace(base, drain_slots=0, gs_lookahead_slots=3)
        cls.enabled = replace(
            cls.base,
            channel_fading=replace(
                cls.base.channel_fading,
                enabled=True,
                sigma_db=2.0,
                rho=0.95,
            ),
        )

    @staticmethod
    def build_sequence(
        config: RemoteSensingScenarioConfig,
        seed: int,
        slots: int = 3,
    ) -> list[ScenarioSnapshot]:
        scenario = RemoteSensingScenario(config, channel_seed=seed)
        return [scenario.build_snapshot(slot) for slot in range(slots)]

    def test_disabled_fading_preserves_complete_snapshot(self) -> None:
        first_config = replace(
            self.base,
            channel_fading=ChannelFadingConfig(
                enabled=False,
                seed=1,
                sigma_db=0.1,
                rho=0.0,
            ),
        )
        second_config = replace(
            self.base,
            channel_fading=ChannelFadingConfig(
                enabled=False,
                seed=999,
                sigma_db=20.0,
                rho=0.99,
            ),
        )
        first_scenario = RemoteSensingScenario(first_config)
        second_scenario = RemoteSensingScenario(second_config)
        first = first_scenario.build_snapshot(0)
        second = second_scenario.build_snapshot(0)

        self.assertEqual(
            [edge.csv_row() for edge in first.edges],
            [edge.csv_row() for edge in second.edges],
        )
        self.assertEqual(len(first.nodes), len(second.nodes))
        for first_node, second_node in zip(first.nodes, second.nodes):
            self.assertEqual(first_node.label, second_node.label)
            self.assertTrue(torch.equal(first_node.position_km, second_node.position_km))
        self.assertEqual(first_scenario.channel_fading._generators, {})
        self.assertEqual(second_scenario.channel_fading._generators, {})

    def test_same_seed_repeats_and_different_seed_changes_capacities_only(self) -> None:
        first = self.build_sequence(self.enabled, 100)
        repeated = self.build_sequence(self.enabled, 100)
        different = self.build_sequence(self.enabled, 101)

        self.assertEqual(capacity_signature(first), capacity_signature(repeated))
        first_capacities = capacity_signature(first)
        different_capacities = capacity_signature(different)
        self.assertEqual(
            [row[:-1] for row in first_capacities],
            [row[:-1] for row in different_capacities],
        )
        self.assertTrue(
            any(a[-1] != b[-1] for a, b in zip(first_capacities, different_capacities))
        )

        for first_snapshot, different_snapshot in zip(first, different):
            self.assertEqual(len(first_snapshot.nodes), len(different_snapshot.nodes))
            for first_node, different_node in zip(
                first_snapshot.nodes, different_snapshot.nodes
            ):
                self.assertTrue(
                    torch.equal(first_node.position_km, different_node.position_km)
                )
            self.assertEqual(
                [
                    (edge.link_type, edge.src, edge.dst, edge.distance_km, edge.available)
                    for edge in first_snapshot.edges
                ],
                [
                    (edge.link_type, edge.src, edge.dst, edge.distance_km, edge.available)
                    for edge in different_snapshot.edges
                ],
            )

    def test_bidirectional_space_links_share_fading_state(self) -> None:
        snapshot = self.build_sequence(self.enabled, 321, slots=1)[0]
        by_key = {
            (edge.link_type, edge.src_id, edge.dst_id): edge for edge in snapshot.edges
        }
        for link_type in (LINK_ISL, LINK_RLS_BACKUP):
            paired = [
                edge
                for edge in snapshot.edges
                if edge.link_type == link_type
                and (link_type, edge.dst_id, edge.src_id) in by_key
            ]
            self.assertTrue(paired, f"expected reverse pairs for {link_type}")
            for edge in paired:
                reverse = by_key[(link_type, edge.dst_id, edge.src_id)]
                self.assertEqual(edge.capacity_bps, reverse.capacity_bps)
                self.assertEqual(edge.energy_cost, reverse.energy_cost)

    def test_task_seed_and_channel_seed_are_independent(self) -> None:
        first = self.build_sequence(self.enabled, 501)
        different_channel = self.build_sequence(self.enabled, 502)
        repeated_channel = self.build_sequence(self.enabled, 501)
        factory = MultiSourceTaskFactory(
            self.enabled,
            self.enabled.link_budgets.access.packet_size_bits,
        )
        first_tasks = factory.build_tasks(first[0], first, episode=0, seed=77)
        different_channel_tasks = factory.build_tasks(
            different_channel[0], different_channel, episode=0, seed=77
        )
        factory.build_tasks(repeated_channel[0], repeated_channel, episode=0, seed=78)

        self.assertEqual(task_signature(first_tasks), task_signature(different_channel_tasks))
        self.assertEqual(capacity_signature(first), capacity_signature(repeated_channel))
        self.assertNotEqual(capacity_signature(first), capacity_signature(different_channel))

    def test_reset_reproducibility_and_cache_seed_guard(self) -> None:
        cache = RemoteSensingEnvCache.build(
            self.enabled,
            time_slots=1,
            channel_seed=601,
        )
        env = RemoteSensingAgentEnv(
            self.enabled,
            time_slots=1,
            drain_slots=0,
            cache=cache,
        )
        env.reset(seed=11, channel_seed=602)
        first = capacity_signature(env._forecast_slots)
        env.reset(seed=99, channel_seed=602)
        repeated = capacity_signature(env._forecast_slots)
        fresh = capacity_signature(self.build_sequence(self.enabled, 602, slots=4))

        self.assertEqual(first, repeated)
        self.assertEqual(first, fresh)
        self.assertNotEqual(first, capacity_signature([slot.snapshot for slot in cache.slots]))
        self.assertEqual(env.channel_seed, 602)

    def test_faded_capacities_and_energies_are_finite(self) -> None:
        snapshots = self.build_sequence(self.enabled, 701, slots=4)
        values = [
            value
            for snapshot in snapshots
            for edge in snapshot.edges
            if edge.link_type in COMMUNICATION_TYPES
            for value in (edge.capacity_bps, edge.energy_cost)
        ]
        self.assertTrue(values)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(value >= 0.0 for value in values))

    def test_configuration_validation(self) -> None:
        self.assertEqual(self.base.channel_fading.seed, 2026)
        fallback = scenario_config_from_dict(_load_simple_yaml(CONFIG_PATH))
        self.assertEqual(fallback.channel_fading, self.base.channel_fading)
        with self.assertRaisesRegex(ValueError, "model"):
            scenario_config_from_dict(
                {"channel_fading": {"model": "unsupported"}}
            )
        with self.assertRaisesRegex(ValueError, "sigma_db"):
            scenario_config_from_dict(
                {"channel_fading": {"sigma_db": -1.0}}
            )
        with self.assertRaisesRegex(ValueError, "rho"):
            scenario_config_from_dict(
                {"channel_fading": {"rho": 1.0}}
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            scenario_config_from_dict(
                {"channel_fading": {"seed": 1.5}}
            )


if __name__ == "__main__":
    unittest.main()
