from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.link_model import LinkModelConfig, compute_fspl, compute_snr  # noqa: E402
from env.remote_sensing_scenario import (  # noqa: E402
    RemoteSensingScenario,
    _load_simple_yaml,
    load_scenario_config,
    scenario_config_from_dict,
)


class RemoteSensingLinkBudgetTest(unittest.TestCase):
    def test_legacy_channel_gain_still_scales_power(self) -> None:
        base = LinkModelConfig()
        fspl = compute_fspl(torch.tensor(4_000.0), base.carrier_frequency_hz)
        reference = float(compute_snr(fspl, base).item())

        doubled_amplitude = replace(base, channel_gain_linear=2.0)

        self.assertAlmostEqual(
            float(compute_snr(fspl, doubled_amplitude).item()) / reference,
            4.0,
            places=6,
        )

    def test_antenna_gains_and_losses_enter_snr_in_db(self) -> None:
        distance = torch.tensor(4_000.0)
        base = LinkModelConfig()
        fspl = compute_fspl(distance, base.carrier_frequency_hz)
        reference = float(compute_snr(fspl, base).item())
        enhanced = replace(
            base,
            tx_antenna_gain_dbi=10.0,
            rx_antenna_gain_dbi=20.0,
            additional_loss_db=3.0,
        )

        ratio = float(compute_snr(fspl, enhanced).item()) / reference

        self.assertAlmostEqual(ratio, 10.0 ** (27.0 / 10.0), places=6)

    def test_noise_temperature_sets_noise_density(self) -> None:
        base = LinkModelConfig(noise_spectral_density_w_per_hz=1.380649e-23 * 500.0)
        thermal = replace(base, system_noise_temperature_k=500.0)
        fspl = compute_fspl(torch.tensor(4_000.0), base.carrier_frequency_hz)

        self.assertAlmostEqual(
            float(compute_snr(fspl, base).item()),
            float(compute_snr(fspl, thermal).item()),
            places=12,
        )

    def test_yaml_uses_distinct_access_isl_and_downlink_budgets(self) -> None:
        config_path = REPO_ROOT / "configs" / "remote_sensing_scenario.yaml"
        config = load_scenario_config(config_path)
        fallback_config = scenario_config_from_dict(_load_simple_yaml(config_path))
        scenario = RemoteSensingScenario(config)

        self.assertEqual(config.link_budgets.access.tx_antenna_gain_dbi, 25.0)
        self.assertEqual(config.link_budgets.isl.rx_antenna_gain_dbi, 25.0)
        self.assertEqual(config.link_budgets.downlink.rx_antenna_gain_dbi, 35.0)
        self.assertEqual(config.link_budgets.access.system_noise_temperature_k, 500.0)
        self.assertEqual(fallback_config.link_budgets, config.link_budgets)
        self.assertEqual(scenario.link_config_for("rls_backup"), config.link_budgets.isl)
        self.assertGreater(
            scenario._communication_metrics(4_000.0, "downlink")[1],
            scenario._communication_metrics(4_000.0, "access")[1],
        )

    def test_rejects_inconsistent_packet_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same packet_size_bits"):
            scenario_config_from_dict(
                {
                    "link_budgets": [
                        {"link_type": "access", "packet_size_bits": 12_000},
                        {"link_type": "isl", "packet_size_bits": 8_000},
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
