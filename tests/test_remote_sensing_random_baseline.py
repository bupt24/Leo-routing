from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_random_baseline import (  # noqa: E402
    RandomBaselineConfig,
    RandomBaselineSimulator,
    write_random_outputs,
)
from env.remote_sensing_scenario import (  # noqa: E402
    LINK_ACCESS,
    LINK_DOWNLINK,
    LINK_ISL,
    LINK_OBSERVATION,
    NODE_CLS,
    NODE_ES,
    NODE_RLS,
    NODE_TARGET,
    RemoteSensingScenario,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioNode,
    ScenarioSnapshot,
)
from env.remote_sensing_task_core import (  # noqa: E402
    DownlinkQueueManager,
    MultiSourceTaskFactory,
    RemoteSensingTask,
    task_edge_transmission_delay_ms,
)


def node(node_id: int, label: str, node_type: str, local_id: int = -1) -> ScenarioNode:
    return ScenarioNode(
        node_id=node_id,
        label=label,
        node_type=node_type,
        position_km=torch.tensor([float(node_id), 0.0, 0.0]),
        local_id=local_id,
    )


def edge(
    src: ScenarioNode,
    dst: ScenarioNode,
    link_type: str,
    *,
    delay_ms: float = 1.0,
    distance_km: float = 1.0,
    capacity_bps: float = 1_000_000.0,
    quality: float = 0.0,
) -> ScenarioEdge:
    return ScenarioEdge(
        time_slot=0,
        src_id=src.node_id,
        dst_id=dst.node_id,
        src=src.label,
        dst=dst.label,
        src_type=src.node_type,
        dst_type=dst.node_type,
        link_type=link_type,
        distance_km=distance_km,
        delay_ms=delay_ms,
        capacity_bps=capacity_bps,
        energy_cost=0.1,
        queue_length=0.0,
        loss_risk=0.0,
        available=True,
        is_data_flow_allowed=True,
        observation_quality=quality,
        off_nadir_angle_deg=1.0,
        slant_range_km=distance_km,
    )


def task(
    task_id: str,
    source: ScenarioNode,
    target: ScenarioNode,
    gs: ScenarioNode,
    observation: ScenarioEdge,
    *,
    traffic_class: int,
    priority: int,
    packets: float,
    deadline_s: float = 100.0,
) -> RemoteSensingTask:
    return RemoteSensingTask(
        task_id=task_id,
        episode=0,
        generation_slot=0,
        generation_time_s=0.0,
        target=target,
        source_rls=source,
        observation_edge=observation,
        candidate_access_edges=[],
        profile_name="test",
        data_size_mb=1.0,
        packet_count=packets,
        traffic_class=traffic_class,
        priority=priority,
        deadline_s=deadline_s,
        destination_gs=gs,
    )


class MultiSourceTaskFactoryTest(unittest.TestCase):
    def test_unique_rls_profiles_fixed_gs_and_reproducibility(self) -> None:
        targets = [node(i, f"T{i}", NODE_TARGET, i) for i in range(4)]
        rls = [node(10 + i, f"R{i}", NODE_RLS, i) for i in range(4)]
        cls_a = node(20, "C0", NODE_CLS, 0)
        cls_b = node(21, "C1", NODE_CLS, 1)
        gs = node(30, "GS", NODE_ES, 0)
        edges: list[ScenarioEdge] = []
        observation_pairs = [(0, 1), (0, 1), (2, 0), (3, 0)]
        for target_index, (best, alternate) in enumerate(observation_pairs):
            edges.append(edge(targets[target_index], rls[best], LINK_OBSERVATION, quality=0.9))
            edges.append(edge(targets[target_index], rls[alternate], LINK_OBSERVATION, quality=0.8))
        for satellite in rls:
            edges.append(edge(satellite, cls_a, LINK_ACCESS))
        edges.append(edge(cls_a, cls_b, LINK_ISL))
        snapshot = ScenarioSnapshot(0, 0.0, targets + rls + [cls_a, cls_b, gs], edges)
        future = ScenarioSnapshot(
            1,
            30.0,
            snapshot.nodes,
            [edge(cls_b, gs, LINK_DOWNLINK, capacity_bps=100_000_000.0)],
        )
        base = RemoteSensingScenarioConfig(
            num_targets=4,
            cls_total_sats=2,
            max_access_cls_per_rls=4,
            max_concurrent_tasks=4,
        )
        config = replace(base, beam=replace(base.beam, max_rls_per_target=3))
        factory = MultiSourceTaskFactory(config, packet_size_bits=12_000.0)

        first = factory.build_tasks(snapshot, [snapshot, future], episode=1, seed=99)
        second = factory.build_tasks(snapshot, [snapshot, future], episode=9, seed=99)

        self.assertEqual(len(first), 4)
        self.assertEqual(len({item.source_rls.node_id for item in first}), 4)
        self.assertTrue(all(item.candidate_rls_count <= 3 for item in first))
        self.assertTrue(all(len(item.candidate_access_edges) <= 4 for item in first))
        self.assertTrue(all(item.destination_gs is not None for item in first))
        self.assertEqual(
            [(item.task_id, item.profile_name, item.data_size_mb, item.source_rls.label) for item in first],
            [(item.task_id, item.profile_name, item.data_size_mb, item.source_rls.label) for item in second],
        )
        ranges = {"urgent": (5.0, 10.0), "normal": (20.0, 50.0), "bulk": (80.0, 150.0)}
        for item in first:
            lower, upper = ranges[item.profile_name]
            self.assertGreaterEqual(item.data_size_mb, lower)
            self.assertLessEqual(item.data_size_mb, upper)


class DownlinkQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(RemoteSensingScenarioConfig(), time_slot_seconds=1.0)
        self.target = node(0, "T", NODE_TARGET)
        self.rls = node(1, "R", NODE_RLS)
        self.cls = node(2, "C", NODE_CLS, 0)
        self.gs1 = node(3, "G1", NODE_ES)
        self.gs2 = node(4, "G2", NODE_ES)
        self.obs = edge(self.target, self.rls, LINK_OBSERVATION)

    def snapshot(self, time_slot: int, gs_nodes: list[ScenarioNode]) -> ScenarioSnapshot:
        links = [edge(self.cls, gs, LINK_DOWNLINK, capacity_bps=1_000.0) for gs in gs_nodes]
        return ScenarioSnapshot(
            time_slot,
            float(time_slot),
            [self.target, self.rls, self.cls, self.gs1, self.gs2],
            links,
        )

    def test_strict_priority_and_partial_cross_slot_service(self) -> None:
        manager = DownlinkQueueManager(self.config, packet_size_bits=100.0)
        urgent = task("u", self.rls, self.target, self.gs1, self.obs, traffic_class=0, priority=0, packets=6)
        normal = task("n", self.rls, self.target, self.gs1, self.obs, traffic_class=1, priority=1, packets=15)
        self.assertTrue(manager.admit(normal, self.cls.node_id, 0))
        self.assertTrue(manager.admit(urgent, self.cls.node_id, 0))

        first_events = manager.service(self.snapshot(0, [self.gs1]))
        self.assertEqual([event["task_id"] for event in first_events], ["u"])
        self.assertEqual(manager.queue_length(self.cls.node_id, self.gs1.node_id, 1), 15)
        second_events = manager.service(self.snapshot(1, [self.gs1]))
        self.assertEqual(second_events, [])
        self.assertEqual(manager.queue_length(self.cls.node_id, self.gs1.node_id, 1), 5)
        third_events = manager.service(self.snapshot(2, [self.gs1]))
        self.assertEqual([event["task_id"] for event in third_events], ["n"])

    def test_same_priority_gs_queues_share_service_time_equally(self) -> None:
        manager = DownlinkQueueManager(self.config, packet_size_bits=100.0)
        first = task("a", self.rls, self.target, self.gs1, self.obs, traffic_class=1, priority=1, packets=10)
        second = task("b", self.rls, self.target, self.gs2, self.obs, traffic_class=1, priority=1, packets=10)
        manager.admit(first, self.cls.node_id, 0)
        manager.admit(second, self.cls.node_id, 0)

        manager.service(self.snapshot(0, [self.gs1, self.gs2]))

        self.assertEqual(manager.queue_length(self.cls.node_id, self.gs1.node_id, 1), 5)
        self.assertEqual(manager.queue_length(self.cls.node_id, self.gs2.node_id, 1), 5)

    def test_timeout_reports_packets_already_delivered_in_previous_slots(self) -> None:
        manager = DownlinkQueueManager(self.config, packet_size_bits=100.0)
        expiring = task(
            "partial",
            self.rls,
            self.target,
            self.gs1,
            self.obs,
            traffic_class=1,
            priority=1,
            packets=15,
            deadline_s=1.5,
        )
        manager.admit(expiring, self.cls.node_id, 0)
        manager.service(self.snapshot(0, [self.gs1]))

        events = manager.expire(2.0)

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["success"])
        self.assertEqual(events[0]["delivered_packets"], 10)


class RandomSemanticsTest(unittest.TestCase):
    def test_dijkstra_requires_relay_even_when_access_cls_can_downlink(self) -> None:
        target = node(0, "T", NODE_TARGET)
        rls = node(1, "R", NODE_RLS)
        access_cls = node(2, "C0", NODE_CLS, 0)
        relay_cls = node(3, "C1", NODE_CLS, 1)
        gs = node(4, "G", NODE_ES)
        observation = edge(target, rls, LINK_OBSERVATION)
        access = edge(rls, access_cls, LINK_ACCESS)
        direct = edge(access_cls, gs, LINK_DOWNLINK)
        relay = edge(access_cls, relay_cls, LINK_ISL)
        relay_downlink = edge(relay_cls, gs, LINK_DOWNLINK)
        nodes = [target, rls, access_cls, relay_cls, gs]
        snapshot = ScenarioSnapshot(0, 0.0, nodes, [observation, access, direct, relay, relay_downlink])
        config = replace(RemoteSensingScenarioConfig(), cls_total_sats=2, min_cls_relay_hops=1)
        simulator = object.__new__(RandomBaselineSimulator)
        simulator.config = config
        simulator.scenario = RemoteSensingScenario(config)
        simulator.scenario.node_by_id = {item.node_id: item for item in nodes}
        simulator.random_config = RandomBaselineConfig(variant="dijkstra", ttl_cap=3)
        item = task("t", rls, target, gs, observation, traffic_class=0, priority=0, packets=1)
        item.candidate_access_edges = [access]

        planned, reason, _, _ = simulator._plan_once(item, snapshot, [snapshot], {}, [item], random.Random(1))
        self.assertEqual(reason, "")
        self.assertIsNotNone(planned)
        assert planned is not None
        self.assertEqual(planned.node_ids[-2:], [access_cls.node_id, relay_cls.node_id])

        simulator.random_config = RandomBaselineConfig(variant="dijkstra", ttl_cap=1)
        planned, reason, _, _ = simulator._plan_once(item, snapshot, [snapshot], {}, [item], random.Random(1))
        self.assertIsNone(planned)
        self.assertEqual(reason, "no_path_to_gs")
        simulator.random_config = RandomBaselineConfig(variant="dijkstra", ttl_cap=3)

        no_relay_snapshot = ScenarioSnapshot(0, 0.0, nodes, [observation, access, direct])
        planned, reason, _, _ = simulator._plan_once(
            item, no_relay_snapshot, [no_relay_snapshot], {}, [item], random.Random(1)
        )
        self.assertIsNone(planned)
        self.assertEqual(reason, "no_path_to_gs")

        simulator.random_config = RandomBaselineConfig(variant="random01", ttl_cap=3)
        random_path, reason = simulator._random_cls_path(
            snapshot,
            access_cls.node_id,
            {access_cls.node_id, relay_cls.node_id},
            random.Random(1),
            informed=True,
            min_relay_hops=1,
        )
        self.assertEqual(reason, "")
        self.assertEqual(random_path, [access_cls.node_id, relay_cls.node_id])

    def test_cls_service_rate_and_task_delay_use_consistent_units(self) -> None:
        config = replace(RemoteSensingScenarioConfig(), cls_total_sats=1, time_slot_seconds=1.0)
        scenario = RemoteSensingScenario(config)
        cls = node(0, "C", NODE_CLS, 0)
        gs = node(1, "G", NODE_ES)
        scenario.node_by_id = {0: cls, 1: gs}
        scenario.cls_node_ids = [0]
        scenario.queue_lengths[0, 0] = 10.0
        link = edge(cls, gs, LINK_DOWNLINK, delay_ms=3.0, capacity_bps=12_000.0)
        snapshot = ScenarioSnapshot(0, 0.0, [cls, gs], [link])

        scenario._service_cls_queues(snapshot)

        self.assertAlmostEqual(float(scenario.cls_service_rates_pps[0].item()), 1.0)
        self.assertAlmostEqual(float(scenario.queue_lengths[0, 0].item()), 9.0)

        propagation_ms, rate_bps, _ = scenario._communication_metrics(1_000.0)
        self.assertAlmostEqual(propagation_ms, 1_000_000.0 / 299_792.458, places=4)
        link.delay_ms = propagation_ms
        link.capacity_bps = rate_bps
        item = task("t", cls, cls, gs, link, traffic_class=0, priority=0, packets=64)
        expected = propagation_ms + 64 * scenario.link_config.packet_size_bits / rate_bps * 1_000.0
        self.assertAlmostEqual(
            task_edge_transmission_delay_ms(link, item, scenario.link_config.packet_size_bits),
            expected,
            places=4,
        )

    def test_random01_looks_ahead_but_random02_can_enter_dead_end(self) -> None:
        start = node(0, "A", NODE_CLS)
        dead = node(1, "B_DEAD", NODE_CLS)
        exit_cls = node(2, "C_EXIT", NODE_CLS)
        snapshot = ScenarioSnapshot(
            0,
            0.0,
            [start, dead, exit_cls],
            [edge(start, dead, LINK_ISL), edge(start, exit_cls, LINK_ISL)],
        )
        simulator = object.__new__(RandomBaselineSimulator)
        simulator.random_config = RandomBaselineConfig(ttl_cap=2, ttl_margin=0, variant="random01")
        informed_path, informed_reason = simulator._random_cls_path(
            snapshot, start.node_id, {exit_cls.node_id}, random.Random(1), informed=True
        )
        simulator.random_config = RandomBaselineConfig(ttl_cap=3, variant="random02")

        class FirstChoice:
            @staticmethod
            def choice(items):
                return items[0]

        naive_path, naive_reason = simulator._random_cls_path(
            snapshot, start.node_id, {exit_cls.node_id}, FirstChoice(), informed=False
        )

        self.assertEqual(informed_reason, "")
        self.assertEqual(informed_path[-1], exit_cls.node_id)
        self.assertEqual(naive_path[-1], dead.node_id)
        self.assertEqual(naive_reason, "no_available_next_hop")

    def test_writer_emits_unified_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_random_outputs(
                directory,
                [{"task_id": "t", "success": False}],
                {"task_count": 1, "_downlink_queue_rows": []},
                [],
            )
            names = {path.name for path in Path(directory).iterdir()}
        self.assertTrue({"tasks.csv", "slot_queues.csv", "topology.csv", "summary.json"} <= names)


if __name__ == "__main__":
    unittest.main()
