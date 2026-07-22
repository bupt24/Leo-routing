"""
Link-budget helpers for the paper-aligned LEO routing environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EPSILON = 1e-9
SECONDS_TO_MS = 1000.0
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
SPEED_OF_LIGHT_KM_PER_MS = SPEED_OF_LIGHT_M_PER_S / 1_000_000.0


@dataclass(frozen=True)
class LinkModelConfig:
    """
    Default constants are engineering placeholders so the environment can run
    end-to-end. For strict paper reproduction, override them with the exact
    values used in your experiments.
    """

    carrier_frequency_hz: float = 30.0e9
    bandwidth_hz: float = 10.0e9
    tx_power_w: float = 50.0
    rx_power_w: float = 1.0
    noise_spectral_density_w_per_hz: float = 1.0e-24
    channel_gain_linear: float = 1.0
    packet_size_bits: float = 1500.0 * 8.0


def compute_fspl(distance_km: torch.Tensor, carrier_frequency_hz: float) -> torch.Tensor:
    distance_m = torch.as_tensor(distance_km, dtype=torch.float64) * 1000.0
    safe_distance_m = distance_m.clamp_min(EPSILON)
    numerator = 4.0 * math.pi * safe_distance_m * float(carrier_frequency_hz)
    return (numerator / SPEED_OF_LIGHT_M_PER_S).pow(2)


def compute_snr(fspl: torch.Tensor, config: LinkModelConfig) -> torch.Tensor:
    total_noise_power = max(
        float(config.noise_spectral_density_w_per_hz) * float(config.bandwidth_hz),
        torch.finfo(torch.float64).tiny,
    )
    gain_power = float(config.channel_gain_linear) ** 2 * float(config.tx_power_w)
    return gain_power / (fspl * total_noise_power).clamp_min(torch.finfo(fspl.dtype).tiny)


def compute_link_rates_bps(
    distance_km: torch.Tensor,
    availability_mask: torch.Tensor,
    config: LinkModelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fspl = compute_fspl(distance_km, config.carrier_frequency_hz)
    snr = compute_snr(fspl, config)
    rates = float(config.bandwidth_hz) * torch.log1p(snr) / math.log(2.0)
    rates = torch.where(availability_mask.bool(), rates, torch.zeros_like(rates))
    return fspl.to(torch.float32), snr.to(torch.float32), rates.to(torch.float32)


def compute_propagation_delay_ms(
    distance_km: torch.Tensor,
    availability_mask: torch.Tensor,
) -> torch.Tensor:
    delay_ms = torch.as_tensor(distance_km, dtype=torch.float32) / SPEED_OF_LIGHT_KM_PER_MS
    return torch.where(availability_mask.bool(), delay_ms, torch.zeros_like(delay_ms))


def compute_transmission_duration_s(
    packet_counts: torch.Tensor,
    rate_bps: torch.Tensor,
    packet_size_bits: float,
) -> torch.Tensor:
    packet_counts = torch.as_tensor(packet_counts, dtype=torch.float32)
    rate_bps = torch.as_tensor(rate_bps, dtype=torch.float32)
    safe_rate = torch.clamp(rate_bps, min=EPSILON)
    duration_s = packet_counts * float(packet_size_bits) / safe_rate
    return torch.where(
        (packet_counts > 0.0) & (rate_bps > 0.0),
        duration_s,
        torch.zeros_like(duration_s),
    )


def compute_transmission_delay_ms(
    packet_counts: torch.Tensor,
    rate_bps: torch.Tensor,
    packet_size_bits: float,
) -> torch.Tensor:
    return compute_transmission_duration_s(packet_counts, rate_bps, packet_size_bits) * SECONDS_TO_MS


def compute_transmission_energy_j(
    packet_counts: torch.Tensor,
    rate_bps: torch.Tensor,
    availability_mask: torch.Tensor,
    config: LinkModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    duration_s = compute_transmission_duration_s(packet_counts, rate_bps, config.packet_size_bits)
    duration_s = torch.where(availability_mask.bool(), duration_s, torch.zeros_like(duration_s))
    return float(config.tx_power_w) * duration_s, duration_s


def compute_reception_energy_j(
    link_duration_s: torch.Tensor,
    availability_mask: torch.Tensor,
    config: LinkModelConfig,
) -> torch.Tensor:
    duration_s = torch.where(availability_mask.bool(), link_duration_s, torch.zeros_like(link_duration_s))
    return float(config.rx_power_w) * duration_s
