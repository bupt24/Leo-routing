"""Coordinate and ground-motion helpers for scenario-level modeling."""

from __future__ import annotations

import math

import torch


EARTH_ROTATION_RATE_RAD_PER_S = 7.2921159e-5
EARTH_RADIUS_KM = 6371.0


def rotation_z(angle_rad: float) -> torch.Tensor:
    """Return a right-handed rotation matrix around the z axis."""
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return torch.tensor(
        [
            [cos_angle, -sin_angle, 0.0],
            [sin_angle, cos_angle, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _rotate_position(position: torch.Tensor, angle_rad: float) -> torch.Tensor:
    position_tensor = torch.as_tensor(position)
    if not torch.is_floating_point(position_tensor):
        position_tensor = position_tensor.to(dtype=torch.float32)
    rotation = rotation_z(angle_rad).to(dtype=position_tensor.dtype, device=position_tensor.device)
    return position_tensor @ rotation.T


def eci_to_ecef(position_eci: torch.Tensor, time_sec: float) -> torch.Tensor:
    """Convert ECI coordinates to ECEF coordinates at ``time_sec``."""
    earth_angle = EARTH_ROTATION_RATE_RAD_PER_S * time_sec
    return _rotate_position(position_eci, -earth_angle)


def ecef_to_eci(position_ecef: torch.Tensor, time_sec: float) -> torch.Tensor:
    """Convert ECEF coordinates to ECI coordinates at ``time_sec``."""
    earth_angle = EARTH_ROTATION_RATE_RAD_PER_S * time_sec
    return _rotate_position(position_ecef, earth_angle)


def latlon_to_ecef_km(lat_deg: float, lon_deg: float, altitude_km: float = 0.0) -> torch.Tensor:
    radius = EARTH_RADIUS_KM + altitude_km
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    return torch.tensor(
        [
            radius * math.cos(lat_rad) * math.cos(lon_rad),
            radius * math.cos(lat_rad) * math.sin(lon_rad),
            radius * math.sin(lat_rad),
        ],
        dtype=torch.float32,
    )


def update_ground_mobile_latlon(
    lat_deg: float,
    lon_deg: float,
    speed_mps: float,
    heading_deg: float,
    dt_s: float,
) -> tuple[float, float]:
    """Advance a ground point along a spherical-Earth great-circle path."""
    if speed_mps == 0.0 or dt_s == 0.0:
        return float(lat_deg), _normalize_lon_deg(float(lon_deg))

    angular_distance = (speed_mps * dt_s / 1000.0) / EARTH_RADIUS_KM
    bearing = math.radians(heading_deg)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_delta = math.sin(angular_distance)
    cos_delta = math.cos(angular_distance)

    lat2 = math.asin(sin_lat1 * cos_delta + cos_lat1 * sin_delta * math.cos(bearing))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * sin_delta * cos_lat1,
        cos_delta - sin_lat1 * math.sin(lat2),
    )

    return math.degrees(lat2), _normalize_lon_deg(math.degrees(lon2))


def _normalize_lon_deg(lon_deg: float) -> float:
    return ((lon_deg + 180.0) % 360.0) - 180.0
