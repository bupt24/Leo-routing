"""Engineering beam model for remote-sensing target-to-RLS observation edges."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EPSILON = 1e-9


@dataclass(frozen=True)
class BeamConfig:
    enabled: bool = True
    pointing_mode: str = "nadir"
    half_angle_deg: float = 20.0
    max_off_nadir_deg: float = 35.0
    max_observation_range_km: float = 3000.0
    gain_max_db: float = 35.0
    three_db_width_deg: float = 6.0
    side_lobe_attenuation_db: float = 25.0
    max_observation_beams_per_rls: int = 1
    max_rls_per_target: int = 1
    base_observation_delay_ms: float = 50.0
    base_observation_energy_j: float = 0.1
    quality_penalty_mode: str = "linear"
    quality_delay_penalty: float = 2.0
    quality_energy_penalty: float = 2.0
    min_quality_for_penalty: float = 0.2


@dataclass(frozen=True)
class BeamResult:
    available: bool
    elevation_deg: float
    off_nadir_angle_deg: float
    slant_range_km: float
    beam_gain_db: float
    beam_gain_linear: float
    observation_quality: float
    observation_delay_ms: float
    observation_energy_j: float
    reason: str = ""


def _as_float_tensor(position: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(position)
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.norm(vector).clamp_min(EPSILON)


def _angle_deg(unit_a: torch.Tensor, unit_b: torch.Tensor) -> float:
    cosine = torch.dot(unit_a, unit_b).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).item())


def compute_off_nadir_angle_deg(
    rls_position_ecef_km: torch.Tensor,
    target_position_ecef_km: torch.Tensor,
) -> float:
    rls_pos = _as_float_tensor(rls_position_ecef_km)
    target_pos = _as_float_tensor(target_position_ecef_km)
    slant_vec = target_pos - rls_pos
    look_vec = _normalize(slant_vec)
    nadir_vec = _normalize(-rls_pos)
    return _angle_deg(look_vec, nadir_vec)


def compute_elevation_angle_deg(
    rls_position_ecef_km: torch.Tensor,
    target_position_ecef_km: torch.Tensor,
) -> float:
    rls_pos = _as_float_tensor(rls_position_ecef_km)
    target_pos = _as_float_tensor(target_position_ecef_km)
    ground_normal = _normalize(target_pos)
    to_sat = _normalize(rls_pos - target_pos)
    sine_elevation = torch.dot(to_sat, ground_normal).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.asin(sine_elevation)).item())


def compute_beam_gain_db(off_nadir_angle_deg: float, beam_config: BeamConfig) -> float:
    width = max(float(beam_config.three_db_width_deg), EPSILON)
    attenuation_db = min(
        12.0 * (float(off_nadir_angle_deg) / width) ** 2,
        float(beam_config.side_lobe_attenuation_db),
    )
    return float(beam_config.gain_max_db) - attenuation_db


def compute_observation_delay_energy(
    observation_quality: float,
    beam_config: BeamConfig,
) -> tuple[float, float]:
    """
    observation_quality is a normalized quality indicator based on beam gain
    relative to maximum gain. It is not a physical delay unit; it describes
    the quality difference between targets near the beam center and edge.
    By default, use a mild linear penalty to avoid unrealistically large
    delays for low-quality targets.
    """
    mode = str(beam_config.quality_penalty_mode).lower()
    base_delay_ms = float(beam_config.base_observation_delay_ms)
    base_energy_j = float(beam_config.base_observation_energy_j)

    if mode == "none":
        return base_delay_ms, base_energy_j

    effective_quality = max(
        float(observation_quality),
        float(beam_config.min_quality_for_penalty),
    )

    if mode == "linear":
        delay_penalty = 1.0 + float(beam_config.quality_delay_penalty) * (1.0 - effective_quality)
        energy_penalty = 1.0 + float(beam_config.quality_energy_penalty) * (1.0 - effective_quality)
        return base_delay_ms * delay_penalty, base_energy_j * energy_penalty

    if mode == "inverse":
        safe_quality = max(effective_quality, EPSILON)
        return base_delay_ms / safe_quality, base_energy_j / safe_quality

    raise ValueError(
        f"Unsupported quality_penalty_mode={beam_config.quality_penalty_mode!r}. "
        "Expected one of {'none', 'linear', 'inverse'}."
    )


def compute_rls_beam_to_target(
    rls_position_ecef_km: torch.Tensor,
    target_position_ecef_km: torch.Tensor,
    min_elevation_deg: float,
    beam_config: BeamConfig,
) -> BeamResult:
    rls_pos = _as_float_tensor(rls_position_ecef_km)
    target_pos = _as_float_tensor(target_position_ecef_km)
    slant_range_km = float(torch.norm(target_pos - rls_pos).item())
    off_nadir_angle_deg = compute_off_nadir_angle_deg(rls_pos, target_pos)
    elevation_deg = compute_elevation_angle_deg(rls_pos, target_pos)
    pointing_mode = str(beam_config.pointing_mode).lower()
    # In steerable mode the satellite slews the beam center to the target;
    # off_nadir_angle_deg still records the required slew angle.
    beam_offset_deg = 0.0 if pointing_mode == "steerable" else off_nadir_angle_deg
    beam_gain_db = compute_beam_gain_db(beam_offset_deg, beam_config)
    beam_gain_linear = 10.0 ** (beam_gain_db / 10.0)
    max_gain_linear = 10.0 ** (float(beam_config.gain_max_db) / 10.0)
    observation_quality = beam_gain_linear / max(max_gain_linear, EPSILON)
    observation_delay_ms, observation_energy_j = compute_observation_delay_energy(
        observation_quality,
        beam_config,
    )

    if not beam_config.enabled:
        available = elevation_deg >= float(min_elevation_deg)
        reason = "legacy_elevation_only" if available else "elevation_below_min"
        return BeamResult(
            available=available,
            elevation_deg=elevation_deg,
            off_nadir_angle_deg=off_nadir_angle_deg,
            slant_range_km=slant_range_km,
            beam_gain_db=beam_gain_db,
            beam_gain_linear=beam_gain_linear,
            observation_quality=observation_quality,
            observation_delay_ms=observation_delay_ms,
            observation_energy_j=observation_energy_j,
            reason=reason,
        )

    if pointing_mode not in {"nadir", "steerable"}:
        return BeamResult(
            available=False,
            elevation_deg=elevation_deg,
            off_nadir_angle_deg=off_nadir_angle_deg,
            slant_range_km=slant_range_km,
            beam_gain_db=beam_gain_db,
            beam_gain_linear=beam_gain_linear,
            observation_quality=observation_quality,
            observation_delay_ms=observation_delay_ms,
            observation_energy_j=observation_energy_j,
            reason="unsupported_pointing_mode",
        )

    reason = "available"
    available = True
    if elevation_deg < float(min_elevation_deg):
        available = False
        reason = "elevation_below_min"
    elif pointing_mode == "nadir" and off_nadir_angle_deg > float(beam_config.half_angle_deg):
        available = False
        reason = "outside_beam_half_angle"
    elif off_nadir_angle_deg > float(beam_config.max_off_nadir_deg):
        available = False
        reason = "exceed_max_off_nadir"
    elif slant_range_km > float(beam_config.max_observation_range_km):
        available = False
        reason = "exceed_max_observation_range"

    return BeamResult(
        available=available,
        elevation_deg=elevation_deg,
        off_nadir_angle_deg=off_nadir_angle_deg,
        slant_range_km=slant_range_km,
        beam_gain_db=beam_gain_db,
        beam_gain_linear=beam_gain_linear,
        observation_quality=observation_quality,
        observation_delay_ms=observation_delay_ms,
        observation_energy_j=observation_energy_j,
        reason=reason,
    )
