"""
Utilities for converting MAPPO-style policy outputs into valid environment actions.
"""

from __future__ import annotations

import torch

from env.aqm_wpq import sanitize_aqm_params


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = torch.as_tensor(logits, dtype=torch.float32)
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(masked_logits, dim=-1)
    fallback = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    invalid_rows = ~mask.any(dim=-1, keepdim=True)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    return torch.where(invalid_rows, torch.zeros_like(probs), torch.where(mask.any(dim=-1, keepdim=True), probs, fallback))


def build_joint_actions_from_logits(
    traffic_logits: torch.Tensor,
    wpq_logits: torch.Tensor,
    aqm_logits: torch.Tensor,
    valid_neighbor_mask: torch.Tensor,
    pmax_upper_bound: float = 0.1,
) -> dict[str, torch.Tensor]:
    """
    Convert raw actor outputs to the environment action structure:
    - traffic distribution over valid neighbors
    - positive WPQ weights
    - ordered AQM parameters alpha < beta and bounded pmax
    """
    traffic = masked_softmax(traffic_logits, valid_neighbor_mask)
    weights = torch.nn.functional.softplus(torch.as_tensor(wpq_logits, dtype=torch.float32)) + 1e-6

    aqm_logits = torch.as_tensor(aqm_logits, dtype=torch.float32)
    alpha_raw = torch.sigmoid(aqm_logits[:, 0]) * 0.7
    beta_raw = 0.2 + torch.sigmoid(aqm_logits[:, 1]) * 0.8
    pmax_raw = torch.sigmoid(aqm_logits[:, 2]) * float(pmax_upper_bound)
    aqm = torch.stack([alpha_raw, beta_raw, pmax_raw], dim=1)
    aqm = sanitize_aqm_params(aqm)

    return {
        "traffic": traffic,
        "weights": weights,
        "aqm": aqm,
    }
