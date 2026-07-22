from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    import torch
except ModuleNotFoundError:
    import types

    class _FakeTensor:
        def __init__(self, value: object):
            self.value = value

        def unsqueeze(self, _dim: int) -> "_FakeTensor":
            return self

        def repeat(self, *_shape: int) -> "_FakeTensor":
            return self

    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.float32 = object()
    torch.device = lambda value: value

    def _tensor(value: object, dtype: object | None = None) -> _FakeTensor:
        return _FakeTensor(value)

    def _zeros(*_shape: int, dtype: object | None = None) -> _FakeTensor:
        return _FakeTensor(0.0)

    def _ones(*_shape: int, dtype: object | None = None) -> _FakeTensor:
        return _FakeTensor(1.0)

    torch.tensor = _tensor
    torch.zeros = _zeros
    torch.ones = _ones
    sys.modules["torch"] = torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


def make_node(node_id: int, label: str, node_type: str, local_id: int = -1) -> ScenarioNode:
    return ScenarioNode(
        node_id=node_id,
        label=label,
        node_type=node_type,
        position_km=torch.tensor([float(node_id), 0.0, 0.0], dtype=torch.float32),
        local_id=local_id,
    )


def make_edge(
    src: ScenarioNode,
    dst: ScenarioNode,
    link_type: str,
    delay_ms: float,
    energy_cost: float,
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
        distance_km=1.0,
        delay_ms=delay_ms,
        capacity_bps=1.0,
        energy_cost=energy_cost,
        queue_length=0.0,
        loss_risk=0.0,
        available=True,
        is_data_flow_allowed=True,
    )


def make_scenario(
    nodes: list[ScenarioNode],
    edges: list[ScenarioEdge],
    config: RemoteSensingScenarioConfig | None = None,
) -> tuple[RemoteSensingScenario, ScenarioSnapshot]:
    config = config or RemoteSensingScenarioConfig(cls_total_sats=2)
    scenario = RemoteSensingScenario(config)
    scenario.node_by_id = {node.node_id: node for node in nodes}
    snapshot = ScenarioSnapshot(time_slot=0, time_sec=0.0, nodes=nodes, edges=edges)
    return scenario, snapshot


class RemoteSensingRouteTaskTest(unittest.TestCase):
    def test_route_task_chooses_reachable_access_cls(self) -> None:
        target = make_node(0, "target1", NODE_TARGET)
        rls = make_node(1, "RLS1", NODE_RLS)
        unreachable_cls = make_node(2, "CLS1", NODE_CLS, local_id=0)
        reachable_cls = make_node(3, "CLS2", NODE_CLS, local_id=1)
        es = make_node(4, "ES1", NODE_ES)
        nodes = [target, rls, unreachable_cls, reachable_cls, es]
        edges = [
            make_edge(target, rls, LINK_OBSERVATION, delay_ms=1.0, energy_cost=0.0),
            make_edge(rls, unreachable_cls, LINK_ACCESS, delay_ms=1.0, energy_cost=0.0),
            make_edge(rls, reachable_cls, LINK_ACCESS, delay_ms=50.0, energy_cost=0.0),
            make_edge(reachable_cls, es, LINK_DOWNLINK, delay_ms=1.0, energy_cost=0.0),
        ]
        scenario, snapshot = make_scenario(nodes, edges)

        route = scenario.route_task(snapshot, target, task_index=0)

        self.assertTrue(route["access_success"])
        self.assertTrue(route["route_success"])
        self.assertEqual(route["selected_cls"], "CLS2")
        self.assertEqual(route["selected_es"], "ES1")
        self.assertEqual(route["path"], "target1 -> RLS1 -> CLS2 -> ES1")

    def test_route_task_keeps_access_success_when_no_cls_path_reaches_es(self) -> None:
        target = make_node(0, "target1", NODE_TARGET)
        rls = make_node(1, "RLS1", NODE_RLS)
        cls = make_node(2, "CLS1", NODE_CLS, local_id=0)
        es = make_node(3, "ES1", NODE_ES)
        nodes = [target, rls, cls, es]
        edges = [
            make_edge(target, rls, LINK_OBSERVATION, delay_ms=1.0, energy_cost=0.0),
            make_edge(rls, cls, LINK_ACCESS, delay_ms=1.0, energy_cost=0.0),
        ]
        scenario, snapshot = make_scenario(nodes, edges)

        route = scenario.route_task(snapshot, target, task_index=0)

        self.assertTrue(route["access_success"])
        self.assertFalse(route["route_success"])
        self.assertEqual(route["rls_node"], "RLS1")
        self.assertEqual(route["selected_cls"], "CLS1")

    def test_route_task_enforces_minimum_cls_relay_hops(self) -> None:
        target = make_node(0, "target1", NODE_TARGET)
        rls = make_node(1, "RLS1", NODE_RLS)
        access_cls = make_node(2, "CLS1", NODE_CLS, local_id=0)
        relay_cls = make_node(3, "CLS2", NODE_CLS, local_id=1)
        es = make_node(4, "ES1", NODE_ES)
        nodes = [target, rls, access_cls, relay_cls, es]
        edges = [
            make_edge(target, rls, LINK_OBSERVATION, delay_ms=1.0, energy_cost=0.0),
            make_edge(rls, access_cls, LINK_ACCESS, delay_ms=1.0, energy_cost=0.0),
            make_edge(access_cls, es, LINK_DOWNLINK, delay_ms=1.0, energy_cost=0.0),
            make_edge(access_cls, relay_cls, LINK_ISL, delay_ms=2.0, energy_cost=0.0),
            make_edge(relay_cls, es, LINK_DOWNLINK, delay_ms=2.0, energy_cost=0.0),
        ]
        config = RemoteSensingScenarioConfig(cls_total_sats=2, min_cls_relay_hops=1)
        scenario, snapshot = make_scenario(nodes, edges, config=config)

        route = scenario.route_task(snapshot, target, task_index=0)

        self.assertTrue(route["route_success"])
        self.assertEqual(route["path"], "target1 -> RLS1 -> CLS1 -> CLS2 -> ES1")
        self.assertEqual(route["hop_count"], 4)
        self.assertEqual(route["cls_relay_hops"], 1)

    def test_build_access_edges_limits_candidates_per_rls(self) -> None:
        rls = make_node(1, "RLS1", NODE_RLS)
        cls1 = make_node(2, "CLS1", NODE_CLS, local_id=0)
        cls2 = make_node(3, "CLS2", NODE_CLS, local_id=1)
        cls3 = make_node(4, "CLS3", NODE_CLS, local_id=2)
        config = RemoteSensingScenarioConfig(
            cls_total_sats=3,
            max_access_cls_per_rls=2,
            rls_access_policy="nearest",
        )
        scenario = RemoteSensingScenario(config)
        scenario._space_link_available = lambda *_args: True  # type: ignore[method-assign]

        edges = scenario._build_access_edges(0, [rls, cls1, cls2, cls3])

        self.assertEqual([edge.dst for edge in edges], ["CLS1", "CLS2"])


if __name__ == "__main__":
    unittest.main()
