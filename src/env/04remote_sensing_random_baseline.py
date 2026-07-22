"""Random04 compatibility layer backed by the shared concurrent simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace

from env.remote_sensing_random_baseline import (
    RANDOM_FAIL_REASONS,
    RetryRandomBaselineConfig as SharedRetryRandomBaselineConfig,
    build_random_metrics_summary,
    run_random_baseline as run_shared_baseline,
    write_random_outputs,
)


@dataclass(frozen=True)
class RetryRandomBaselineConfig(SharedRetryRandomBaselineConfig):
    variant: str = "random04"


RandomBaselineConfig = RetryRandomBaselineConfig


def run_random_baseline(scenario_config, num_time_slots, random_config=None):
    config = random_config or RetryRandomBaselineConfig()
    return run_shared_baseline(
        scenario_config,
        num_time_slots,
        replace(config, variant="random04"),
    )


__all__ = [
    "RANDOM_FAIL_REASONS",
    "RandomBaselineConfig",
    "RetryRandomBaselineConfig",
    "build_random_metrics_summary",
    "run_random_baseline",
    "write_random_outputs",
]
