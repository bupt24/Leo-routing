"""
AQM and WPQ helpers for the paper-aligned LEO routing environment.
"""

from __future__ import annotations

import torch


EPSILON = 1e-9


def normalize_traffic_distribution(
    traffic: torch.Tensor,
    valid_neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Project routing ratios onto the valid-neighbor simplex.
    """
    traffic = torch.as_tensor(traffic, dtype=torch.float32)
    valid_neighbor_mask = valid_neighbor_mask.bool()

    if traffic.shape != valid_neighbor_mask.shape:
        raise ValueError(
            "Traffic distribution shape does not match the valid-neighbor mask."
        )

    masked = torch.where(valid_neighbor_mask, torch.clamp(traffic, min=0.0), 0.0)
    row_sum = masked.sum(dim=1, keepdim=True)

    uniform = valid_neighbor_mask.float()
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = torch.where(
        row_sum > EPSILON,
        masked / row_sum.clamp_min(EPSILON),
        uniform,
    )
    return normalized * valid_neighbor_mask.float()


def sanitize_wpq_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=torch.float32)
    weights = torch.clamp(weights, min=0.0)
    all_zero = weights.sum(dim=1, keepdim=True) <= EPSILON
    return torch.where(all_zero, torch.ones_like(weights), weights)


def sanitize_aqm_params(aqm_params: torch.Tensor, min_gap: float = 1e-3) -> torch.Tensor:
    aqm_params = torch.as_tensor(aqm_params, dtype=torch.float32)
    if aqm_params.dim() != 2 or aqm_params.shape[1] != 3:
        raise ValueError("AQM parameters must have shape [num_nodes, 3].")

    alpha = aqm_params[:, 0].clamp(0.0, 1.0)
    beta = aqm_params[:, 1].clamp(0.0, 1.0)
    pmax = aqm_params[:, 2].clamp(0.0, 1.0)

    lower = torch.minimum(alpha, beta)
    upper = torch.maximum(alpha, beta)
    lower = lower.clamp(max=1.0 - min_gap)
    upper = torch.maximum(upper, lower + min_gap).clamp(max=1.0)

    return torch.stack([lower, upper, pmax], dim=1)


def compute_highest_nonempty_priority_mask(
    queue_lengths: torch.Tensor,
    priority_levels: torch.Tensor,
) -> torch.Tensor:
    """
    Paper definition of B_i(t): backlogged queues in the highest nonempty priority.
    Lower priority-level integers mean higher priority.
    """
    queue_lengths = torch.as_tensor(queue_lengths, dtype=torch.float32)
    priority_levels = torch.as_tensor(
        priority_levels,
        dtype=torch.float32,
        device=queue_lengths.device,
    )
    if queue_lengths.shape[1] != priority_levels.numel():
        raise ValueError("Priority levels must match the number of queues.")

    nonempty = queue_lengths > EPSILON
    priority_grid = priority_levels.unsqueeze(0).expand_as(queue_lengths)
    inf_grid = torch.full_like(priority_grid, float("inf"))
    masked_priorities = torch.where(nonempty, priority_grid, inf_grid)
    highest_priority = masked_priorities.min(dim=1).values
    has_active = torch.isfinite(highest_priority)

    return (
        nonempty
        & has_active.unsqueeze(1)
        & (priority_grid == highest_priority.unsqueeze(1))
    )


def allocate_wpq_service(
    total_service_rate_pps: torch.Tensor,
    queue_lengths: torch.Tensor,
    priority_levels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Paper Eq. (7): strict priority across levels, weighted sharing within the active level.
    """
    total_service_rate_pps = torch.as_tensor(total_service_rate_pps, dtype=torch.float32)
    queue_lengths = torch.as_tensor(queue_lengths, dtype=torch.float32)
    weights = sanitize_wpq_weights(weights).to(queue_lengths.device)

    active_mask = compute_highest_nonempty_priority_mask(queue_lengths, priority_levels)
    active_weights = weights * active_mask.float()
    denom = active_weights.sum(dim=1, keepdim=True)

    service = torch.zeros_like(queue_lengths)
    rows = denom.squeeze(1) > EPSILON
    if rows.any():
        service[rows] = (
            total_service_rate_pps[rows].unsqueeze(1)
            * active_weights[rows]
            / denom[rows]
        )
    return service, active_mask


def compute_aqm_drop_probabilities(
    queue_lengths: torch.Tensor,
    queue_capacities: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pmax: torch.Tensor,
) -> torch.Tensor:
    """
    Paper Eq. (13): queue-length-based quadratic AQM drop profile.
    """
    queue_lengths = torch.as_tensor(queue_lengths, dtype=torch.float32)
    queue_capacities = torch.as_tensor(
        queue_capacities,
        dtype=torch.float32,
        device=queue_lengths.device,
    )

    d1 = alpha.unsqueeze(1) * queue_capacities.unsqueeze(0)
    d2 = beta.unsqueeze(1) * queue_capacities.unsqueeze(0)
    ratio = (queue_lengths - d1) / (d2 - d1).clamp_min(EPSILON)
    quadratic = pmax.unsqueeze(1) * torch.clamp(ratio, min=0.0, max=1.0).pow(2)

    below = queue_lengths < d1
    above = queue_lengths >= d2

    probs = quadratic
    probs = torch.where(below, torch.zeros_like(probs), probs)
    probs = torch.where(above, pmax.unsqueeze(1), probs)
    return probs.clamp(0.0, 1.0)
