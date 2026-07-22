"""Dijkstra baseline using the shared multi-source simulation core."""

from __future__ import annotations

from typing import Any

from env.remote_sensing_random_baseline import (
    RandomBaselineConfig,
    run_random_baseline,
)
from env.remote_sensing_scenario import (
    LINK_DOWNLINK,
    RemoteSensingScenarioConfig,
    ScenarioEdge,
    ScenarioSnapshot,
)
from env.remote_sensing_task_core import shortest_isl_path


def dijkstra_cls_path_to_es(
    snapshot: ScenarioSnapshot,
    start_cls_id: int,
    link_config: Any = None,
) -> list[int]:
    """Return the minimum-delay CLS path to an exit visible in this snapshot."""

    del link_config
    exits = {
        edge.src_id
        for edge in snapshot.edges
        if edge.link_type == LINK_DOWNLINK and edge.available and edge.is_data_flow_allowed
    }
    path, _delay_ms = shortest_isl_path(snapshot, start_cls_id, exits)
    return path


def run_dijkstra_baseline(
    scenario_config: RemoteSensingScenarioConfig,
    num_time_slots: int,
    *,
    random_seed: int = 42,
    ttl_cap: int = 10,
    drain_slots: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ScenarioEdge]]:
    """Run Dijkstra with the same tasks, fixed GS, queues and delivery rules."""

    return run_random_baseline(
        scenario_config,
        num_time_slots,
        RandomBaselineConfig(
            seed=random_seed,
            ttl_cap=ttl_cap,
            drain_slots=drain_slots,
            variant="dijkstra",
        ),
    )


__all__ = ["dijkstra_cls_path_to_es", "run_dijkstra_baseline"]
