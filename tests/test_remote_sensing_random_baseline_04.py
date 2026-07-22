from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_random_baseline import (  # noqa: E402
    RandomBaselineConfig,
    RandomBaselineSimulator,
)
from env.remote_sensing_scenario import (  # noqa: E402
    LINK_ACCESS,
    LINK_DOWNLINK,
    NODE_CLS,
    NODE_ES,
    NODE_RLS,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioNode,
    ScenarioSnapshot,
)


def node(node_id: int, label: str, node_type: str, local_id: int = -1) -> ScenarioNode:
    return ScenarioNode(
        node_id=node_id,
        label=label,
        node_type=node_type,
        local_id=local_id,
        position_km=torch.tensor([float(node_id), 0.0, 0.0]),
    )


def edge(src: ScenarioNode, dst: ScenarioNode, link_type: str, distance: float) -> ScenarioEdge:
    return ScenarioEdge(
        time_slot=0,
        src_id=src.node_id,
        dst_id=dst.node_id,
        src=src.label,
        dst=dst.label,
        src_type=src.node_type,
        dst_type=dst.node_type,
        link_type=link_type,
        distance_km=distance,
        delay_ms=1.0,
        capacity_bps=1_000_000.0,
        energy_cost=0.0,
        queue_length=0.0,
        loss_risk=0.0,
        available=True,
        is_data_flow_allowed=True,
    )


class CooperativeAccessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rls = node(0, "R", NODE_RLS)
        self.clear_cls = node(1, "CLEAR", NODE_CLS, 0)
        self.busy_cls = node(2, "BUSY", NODE_CLS, 1)
        self.gs = node(3, "GS", NODE_ES)
        self.clear_access = edge(self.rls, self.clear_cls, LINK_ACCESS, 10.0)
        self.busy_access = edge(self.rls, self.busy_cls, LINK_ACCESS, 1.0)
        self.current = ScenarioSnapshot(
            0,
            0.0,
            [self.rls, self.clear_cls, self.busy_cls, self.gs],
            [],
        )
        self.future = ScenarioSnapshot(
            1,
            30.0,
            self.current.nodes,
            [
                edge(self.clear_cls, self.gs, LINK_DOWNLINK, 1.0),
                edge(self.busy_cls, self.gs, LINK_DOWNLINK, 1.0),
            ],
        )
        config = RemoteSensingScenarioConfig(
            cls_total_sats=2,
            queue_capacities=(100, 100, 100),
            max_cross_layer_distance_km=100.0,
        )
        self.simulator = object.__new__(RandomBaselineSimulator)
        self.simulator.config = config
        self.simulator.random_config = RandomBaselineConfig(variant="random04")
        self.simulator.scenario = SimpleNamespace(
            node_by_id={item.node_id: item for item in self.current.nodes},
            queue_lengths=torch.tensor(
                [[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]], dtype=torch.float32
            ),
        )
        self.task = SimpleNamespace(
            task_id="task",
            traffic_class=0,
            packet_count=10.0,
            destination_gs=self.gs,
            source_rls=self.rls,
            candidate_access_edges=[self.clear_access, self.busy_access],
        )

    def score(self, candidate: ScenarioEdge, injected: dict[tuple[int, int], float]) -> float:
        return self.simulator._cooperative_access_score(
            self.task,
            candidate,
            self.current,
            [self.future],
            injected,
            [self.task],
        )

    def test_queue_and_remaining_capacity_can_outweigh_shorter_distance(self) -> None:
        clear_score = self.score(self.clear_access, {})
        busy_score = self.score(self.busy_access, {})

        self.assertLess(clear_score, busy_score)

    def test_prior_rls_injection_increases_candidate_cost(self) -> None:
        before = self.score(self.clear_access, {})
        after = self.score(self.clear_access, {(self.clear_cls.node_id, 0): 50.0})

        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
