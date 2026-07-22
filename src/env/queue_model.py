"""
Queueing helpers for both the legacy single-queue proxy and the
paper-aligned G/G/1/K routing environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EPSILON = 1e-9
SECONDS_TO_MS = 1000.0


def compute_queue_delay(
    queue_length: torch.Tensor | float,
    service_rate: float = 1.0,
    arrival_rate: float = 0.5,
) -> torch.Tensor:
    """
    Legacy M/M/1-style queue-delay proxy kept for backward compatibility.
    """
    del arrival_rate
    queue_tensor = torch.as_tensor(queue_length, dtype=torch.float32)
    safe_service_rate = max(float(service_rate), EPSILON)
    return queue_tensor / safe_service_rate


def compute_total_delay(
    propagation_delay: torch.Tensor | float,
    queue_length: torch.Tensor | float,
    service_rate: float = 1.0,
) -> torch.Tensor:
    """
    Legacy helper kept for existing training/evaluation scripts.
    """
    prop_delay = torch.as_tensor(propagation_delay, dtype=torch.float32)
    queue_delay = compute_queue_delay(queue_length, service_rate=service_rate)
    return prop_delay + queue_delay


@dataclass(frozen=True)
class QueueUpdateResult:
    next_queue_lengths: torch.Tensor
    overflow_drops: torch.Tensor


def compute_node_service_rate_packets(
    link_rates_bps: torch.Tensor,
    packet_size_bits: float,
) -> torch.Tensor:
    """
    Paper Eq. (6): mu_i(t) = sum_j R_ij(t) / S, in packets/s.
    """
    packet_size_bits = max(float(packet_size_bits), EPSILON)
    return link_rates_bps.sum(dim=1) / packet_size_bits


def compute_served_packets(
    queue_lengths: torch.Tensor,
    service_rates_pps: torch.Tensor,
    slot_duration_s: float,
) -> torch.Tensor:
    """
    Paper Eq. (9): x_i,m(t) = min(q_i,m(t), mu_i,m(t) * dt).
    """
    service_capacity = service_rates_pps * float(slot_duration_s)
    return torch.minimum(queue_lengths, service_capacity)


def update_queue_lengths(
    queue_lengths: torch.Tensor,
    arrivals: torch.Tensor,
    served_packets: torch.Tensor,
    queue_capacities: torch.Tensor,
) -> QueueUpdateResult:
    """
    Paper Eq. (8): q_i,m(t+1) = min(max(q_i,m(t) + r_i,m(t) - x_i,m(t), 0), K_m).
    """
    capacities = queue_capacities.unsqueeze(0).to(queue_lengths.device, queue_lengths.dtype)
    unclipped = torch.clamp(queue_lengths + arrivals - served_packets, min=0.0)
    next_queue_lengths = torch.minimum(unclipped, capacities)
    overflow_drops = torch.clamp(unclipped - capacities, min=0.0)
    return QueueUpdateResult(
        next_queue_lengths=next_queue_lengths,
        overflow_drops=overflow_drops,
    )


def compute_per_class_queueing_delay_ms(
    queue_lengths: torch.Tensor,
    service_rates_pps: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Paper Eq. (10): D_queue_i,m(t) ~= q_i,m(t) / mu_i,m(t), reported in ms.
    """
    safe_rates = torch.clamp(service_rates_pps, min=EPSILON)
    delays_ms = queue_lengths / safe_rates * SECONDS_TO_MS
    return torch.where(active_mask, delays_ms, torch.zeros_like(delays_ms))


def compute_aggregate_queueing_delay_ms(
    per_class_queue_delay_ms: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Paper Eq. (11): aggregate node queueing delay is the max active per-class delay.
    """
    masked = torch.where(
        active_mask,
        per_class_queue_delay_ms,
        torch.zeros_like(per_class_queue_delay_ms),
    )
    return masked.max(dim=1).values
