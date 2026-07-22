"""Random02 compatibility layer backed by the shared concurrent simulator."""

from __future__ import annotations

from dataclasses import dataclass, replace

from env.remote_sensing_random_baseline import (
    RANDOM_FAIL_REASONS,
    RandomBaselineConfig as SharedRandomBaselineConfig,
    build_random_metrics_summary,
    run_random_baseline as run_shared_baseline,
    write_random_outputs,
)


@dataclass(frozen=True)
class RandomBaselineConfig(SharedRandomBaselineConfig):
    variant: str = "random02"


def run_random_baseline(scenario_config, num_time_slots, random_config=None):
    config = random_config or RandomBaselineConfig()
    return run_shared_baseline(
        scenario_config,
        num_time_slots,
        replace(config, variant="random02"),
    )


__all__ = [
    "RANDOM_FAIL_REASONS",
    "RandomBaselineConfig",
    "build_random_metrics_summary",
    "run_random_baseline",
    "write_random_outputs",
]
