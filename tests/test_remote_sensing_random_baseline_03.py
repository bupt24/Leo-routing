from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.remote_sensing_random_baseline import (  # noqa: E402
    RandomBaselineConfig,
    RandomBaselineSimulator,
)


def load_numbered(name: str):
    path = REPO_ROOT / "src" / "env" / f"{name}remote_sensing_random_baseline.py"
    spec = importlib.util.spec_from_file_location(f"test_random_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RetryRandomBaselineTest(unittest.TestCase):
    def test_numbered_sources_use_shared_core_directly(self) -> None:
        random02 = load_numbered("02")
        random03 = load_numbered("03")
        self.assertEqual(random02.RandomBaselineConfig().variant, "random02")
        self.assertEqual(random03.RetryRandomBaselineConfig().variant, "random03")
        for name in ("02", "03", "04"):
            source = (REPO_ROOT / "src" / "env" / f"{name}remote_sensing_random_baseline.py").read_text()
            self.assertNotIn("importlib.util", source)

    def test_retry_accumulates_failed_attempt_delay_and_energy(self) -> None:
        simulator = object.__new__(RandomBaselineSimulator)
        simulator.random_config = RandomBaselineConfig(
            variant="random03", max_attempts=3
        )
        final_plan = SimpleNamespace(
            route_delay_ms=20.0,
            energy_j=3.0,
            attempt_count=0,
            failed_attempts=0,
            attempted_delay_ms=0.0,
            attempted_energy_j=0.0,
        )
        outcomes = iter(
            [
                (None, "no_available_next_hop", 12.0, 2.0),
                (final_plan, "", 20.0, 3.0),
            ]
        )
        simulator._plan_once = lambda *_args: next(outcomes)
        task = SimpleNamespace(deadline_s=100.0)

        plan, reason, stats = simulator._plan_with_retries(
            task,
            SimpleNamespace(),
            [],
            {},
            [],
            SimpleNamespace(),
        )

        self.assertIs(plan, final_plan)
        self.assertEqual(reason, "no_available_next_hop")
        self.assertEqual(plan.attempt_count, 2)
        self.assertEqual(plan.failed_attempts, 1)
        self.assertEqual(plan.attempted_delay_ms, 32.0)
        self.assertEqual(plan.attempted_energy_j, 5.0)
        self.assertEqual(stats["attempt_fail_reasons"], "no_available_next_hop")


if __name__ == "__main__":
    unittest.main()
