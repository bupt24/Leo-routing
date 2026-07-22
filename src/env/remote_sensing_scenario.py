"""
Remote-sensing-task-driven CLS-layer routing scenario.

This module intentionally does not modify LEORoutingEnv. It builds a separate
layered scenario for the first-stage deterministic workflow:

Observation Target -> RLS -> CLS -> CLS -> Ground Station

RLS-RLS links are retained in the topology as backup/control links, but they
are not used by default for observation packet forwarding. The main data path
is Observation Target -> RLS -> CLS -> CLS -> Ground Station.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from env.aqm_wpq import allocate_wpq_service, compute_aqm_drop_probabilities
from env.beam_model import BeamConfig, BeamResult, compute_rls_beam_to_target
from env.coordinate_utils import (
    eci_to_ecef,
    latlon_to_ecef_km,
    update_ground_mobile_latlon,
)
from env.link_model import (
    LinkModelConfig,
    compute_link_rates_bps,
    compute_propagation_delay_ms,
    compute_reception_energy_j,
    compute_transmission_delay_ms,
    compute_transmission_energy_j,
)
from env.queue_model import (
    compute_aggregate_queueing_delay_ms,
    compute_per_class_queueing_delay_ms,
    compute_served_packets,
    update_queue_lengths,
)
from env.topology import (
    check_elevation_constraint,
    check_line_of_sight,
    compute_grid_plus_isl,
    generate_walker_constellation,
)


NODE_TARGET = "target"
NODE_RLS = "rls"
NODE_CLS = "cls"
NODE_ES = "es"

LINK_OBSERVATION = "observation"
LINK_ACCESS = "access"
LINK_ISL = "isl"
LINK_DOWNLINK = "downlink"
LINK_RLS_BACKUP = "rls_backup"

ACCESS_POLICIES = {"nearest", "min_delay", "best_quality"}


@dataclass(frozen=True)
class GroundPoint:
    name: str
    lat_deg: float
    lon_deg: float
    altitude_km: float = 0.0
    motion_type: str = "fixed"
    speed_mps: float = 0.0
    heading_deg: float = 0.0


@dataclass(frozen=True)
class RoutingWeights:
    delay: float = 0.4
    energy: float = 0.2
    queue: float = 0.3
    loss: float = 0.1

    def normalized(self) -> "RoutingWeights":
        total = self.delay + self.energy + self.queue + self.loss
        if total <= 0.0:
            return RoutingWeights()
        return RoutingWeights(
            delay=self.delay / total,
            energy=self.energy / total,
            queue=self.queue / total,
            loss=self.loss / total,
        )


@dataclass(frozen=True)
class TaskProfile:
    name: str
    min_data_mb: float
    max_data_mb: float
    traffic_class: int
    priority: int
    deadline_s: float
    probability: float


@dataclass(frozen=True)
class GSSelectionWeights:
    path_delay: float = 0.35
    exit_reachability: float = 0.20
    visibility_window: float = 0.20
    assigned_load: float = 0.25

    def normalized(self) -> "GSSelectionWeights":
        total = self.path_delay + self.exit_reachability + self.visibility_window + self.assigned_load
        if total <= 0.0:
            return GSSelectionWeights()
        return GSSelectionWeights(
            path_delay=self.path_delay / total,
            exit_reachability=self.exit_reachability / total,
            visibility_window=self.visibility_window / total,
            assigned_load=self.assigned_load / total,
        )


DEFAULT_TASK_PROFILES = (
    TaskProfile("urgent", 5.0, 10.0, 0, 0, 60.0, 0.2),
    TaskProfile("normal", 20.0, 50.0, 1, 1, 180.0, 0.5),
    TaskProfile("bulk", 80.0, 150.0, 2, 2, 600.0, 0.3),
)


@dataclass(frozen=True)
class RemoteSensingScenarioConfig:
    time_slot_seconds: float = 1.0
    num_targets: int = 3
    targets: tuple[GroundPoint, ...] = field(default_factory=tuple)

    rls_total_sats: int = 18
    rls_num_planes: int = 3
    rls_sats_per_plane: int = 6
    rls_altitude_km: float = 580.0
    rls_inclination_deg: float = 97.7
    rls_phase_offset: int = 1
    rls_is_agent: bool = False
    allow_rls_to_rls_edge: bool = True
    allow_rls_to_rls_data_flow: bool = False

    cls_total_sats: int = 15
    cls_num_planes: int = 5
    cls_sats_per_plane: int = 3
    cls_altitude_km: float = 1150.0
    cls_inclination_deg: float = 53.0
    cls_phase_offset: int = 1
    cls_is_agent: bool = True

    num_es: int = 3
    es_points: tuple[GroundPoint, ...] = field(default_factory=tuple)
    es_is_agent: bool = False

    allow_rls_to_es: bool = False
    allow_cls_to_rls: bool = False
    allow_es_to_any: bool = False

    rls_access_policy: str = "best_quality"
    max_access_cls_per_rls: int = 0
    min_cls_relay_hops: int = 0
    routing_weights: RoutingWeights = field(default_factory=RoutingWeights)

    min_elevation_deg: float = 10.0
    beam: BeamConfig = field(default_factory=BeamConfig)
    max_rls_isl_distance_km: float = 12_000.0
    max_cls_isl_distance_km: float = 16_000.0
    max_cross_layer_distance_km: float = 20_000.0
    max_downlink_distance_km: float = 20_000.0

    packet_count_per_task: float = 16.0
    packet_class: int = 0
    num_traffic_classes: int = 3
    queue_capacities: tuple[int, ...] = (100, 100, 100)
    priority_levels: tuple[int, ...] = (0, 1, 2)
    default_aqm: tuple[float, float, float] = (0.4, 0.8, 0.1)
    background_packets_per_cls: float = 0.0
    background_packet_class: int = 0
    max_concurrent_tasks: int = 4
    task_profiles: tuple[TaskProfile, ...] = DEFAULT_TASK_PROFILES
    gs_selection_weights: GSSelectionWeights = field(default_factory=GSSelectionWeights)
    gs_lookahead_slots: int = 20
    drain_slots: int = 20
    downlink_queue_capacities: tuple[int, ...] = (20_000, 50_000, 100_000)
    downlink_wpq_weights: tuple[float, ...] = (1.0, 1.0, 1.0)

    normalizer_delay_ms: float = 120.0
    normalizer_energy_j: float = 1.0
    normalizer_queue_packets: float = 100.0
    normalizer_loss: float = 1.0
    reward_cost_clip: float = 3.0
    reward_success_bonus: float = 1.0
    reward_failure_penalty: float = 1.0
    reward_terminal_cls_delay_weight: float = 1.0
    reward_terminal_cls_delay_norm_ms: float = 30.0


@dataclass
class ScenarioNode:
    node_id: int
    label: str
    node_type: str
    position_km: torch.Tensor
    local_id: int = -1
    plane_id: int | None = None
    sat_id_in_plane: int | None = None
    is_agent: bool = False
    has_queue: bool = False
    can_relay: bool = False
    lat_deg: float | None = None
    lon_deg: float | None = None
    frame: str = "ECEF"
    motion_type: str = ""


@dataclass
class ScenarioEdge:
    time_slot: int
    src_id: int
    dst_id: int
    src: str
    dst: str
    src_type: str
    dst_type: str
    link_type: str
    distance_km: float
    delay_ms: float
    capacity_bps: float
    energy_cost: float
    queue_length: float
    loss_risk: float
    available: bool
    is_data_flow_allowed: bool
    elevation_deg: float = 0.0
    off_nadir_angle_deg: float = 0.0
    slant_range_km: float = 0.0
    beam_gain_db: float = 0.0
    observation_quality: float = 0.0
    beam_available: bool = False
    beam_reason: str = ""

    def csv_row(self) -> dict[str, Any]:
        return {
            "time_slot": self.time_slot,
            "src": self.src,
            "dst": self.dst,
            "src_type": self.src_type,
            "dst_type": self.dst_type,
            "link_type": self.link_type,
            "distance_km": self.distance_km,
            "delay_ms": self.delay_ms,
            "capacity_bps": self.capacity_bps,
            "energy_cost": self.energy_cost,
            "queue_length": self.queue_length,
            "loss_risk": self.loss_risk,
            "available": self.available,
            "is_data_flow_allowed": self.is_data_flow_allowed,
            "elevation_deg": self.elevation_deg,
            "off_nadir_angle_deg": self.off_nadir_angle_deg,
            "slant_range_km": self.slant_range_km,
            "beam_gain_db": self.beam_gain_db,
            "observation_quality": self.observation_quality,
            "beam_available": self.beam_available,
            "beam_reason": self.beam_reason,
        }


@dataclass(frozen=True)
class ScenarioNodeRecord:
    time_slot: int
    node: str
    node_type: str
    lat_deg: float
    lon_deg: float
    x_km: float
    y_km: float
    z_km: float
    frame: str
    motion_type: str

    def csv_row(self) -> dict[str, Any]:
        return {
            "time_slot": self.time_slot,
            "node": self.node,
            "node_type": self.node_type,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "x_km": self.x_km,
            "y_km": self.y_km,
            "z_km": self.z_km,
            "frame": self.frame,
            "motion_type": self.motion_type,
        }


@dataclass
class ScenarioSnapshot:
    time_slot: int
    time_sec: float
    nodes: list[ScenarioNode]
    edges: list[ScenarioEdge]


@dataclass
class ScenarioResult:
    routes: list[dict[str, Any]]
    topology_edges: list[ScenarioEdge]
    metrics_summary: dict[str, Any]
    topology_nodes: list[ScenarioNodeRecord] = field(default_factory=list)


DEFAULT_TARGET_POINTS = (
    GroundPoint("agricultural_land_monitoring", 35.0, 115.0, motion_type="fixed"),
    GroundPoint("forest_monitoring", 45.0, 100.0, motion_type="fixed"),
    GroundPoint("maritime_monitoring", 10.0, 140.0, motion_type="fixed"),
    GroundPoint("mobile_user", 30.0, 120.0, motion_type="moving", speed_mps=15.0, heading_deg=90.0),
)

DEFAULT_ES_POINTS = (
    GroundPoint("ES1", 40.0, 116.0, motion_type="fixed"),
    GroundPoint("ES2", 31.2, 121.5, motion_type="fixed"),
    GroundPoint("ES3", 22.3, 114.2, motion_type="fixed"),
)

GROUND_MOTION_TYPES = {"fixed", "moving"}



def _ground_point_from_config(item: Any, default: GroundPoint) -> GroundPoint:
    if isinstance(item, dict):
        motion_type = str(item.get("motion_type", default.motion_type)).lower()
        return GroundPoint(
            name=str(item.get("name", default.name)),
            lat_deg=float(item.get("lat_deg", default.lat_deg)),
            lon_deg=float(item.get("lon_deg", default.lon_deg)),
            altitude_km=float(item.get("altitude_km", default.altitude_km)),
            motion_type=motion_type,
            speed_mps=float(item.get("speed_mps", default.speed_mps)),
            heading_deg=float(item.get("heading_deg", default.heading_deg)),
        )

    name = str(item or default.name)
    return GroundPoint(
        name=name,
        lat_deg=default.lat_deg,
        lon_deg=default.lon_deg,
        altitude_km=default.altitude_km,
        motion_type=default.motion_type,
        speed_mps=default.speed_mps,
        heading_deg=default.heading_deg,
    )


def _ground_latlon_at_time(point: GroundPoint, time_sec: float) -> tuple[float, float]:
    if point.motion_type == "moving":
        return update_ground_mobile_latlon(
            point.lat_deg,
            point.lon_deg,
            point.speed_mps,
            point.heading_deg,
            time_sec,
        )
    return point.lat_deg, point.lon_deg


def _ecef_to_latlon_deg(position_km: torch.Tensor) -> tuple[float, float]:
    x_km, y_km, z_km = (float(value) for value in position_km.detach().cpu().tolist())
    lat_deg = math.degrees(math.atan2(z_km, math.hypot(x_km, y_km)))
    lon_deg = math.degrees(math.atan2(y_km, x_km))
    return lat_deg, lon_deg


def _node_record(time_slot: int, node: ScenarioNode) -> ScenarioNodeRecord:
    x_km, y_km, z_km = (float(value) for value in node.position_km.detach().cpu().tolist())
    lat_deg, lon_deg = (
        (node.lat_deg, node.lon_deg)
        if node.lat_deg is not None and node.lon_deg is not None
        else _ecef_to_latlon_deg(node.position_km)
    )
    return ScenarioNodeRecord(
        time_slot=time_slot,
        node=node.label,
        node_type=node.node_type,
        lat_deg=float(lat_deg),
        lon_deg=float(lon_deg),
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        frame=node.frame,
        motion_type=node.motion_type,
    )


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_tuple(raw: Any, default: tuple[Any, ...], cast: type) -> tuple[Any, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        values = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        values = list(raw)
    return tuple(cast(value) for value in values)


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = _strip_comment(raw_line).rstrip()
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0:
            current_item = None
            if line.endswith(":"):
                current_key = line[:-1].strip()
                root[current_key] = {}
            else:
                key, value = line.split(":", 1)
                root[key.strip()] = _parse_scalar(value)
                current_key = None
            continue

        if current_key is None:
            continue

        if line.startswith("- "):
            if not isinstance(root.get(current_key), list):
                root[current_key] = []
            rest = line[2:].strip()
            if ":" in rest:
                key, value = rest.split(":", 1)
                current_item = {key.strip(): _parse_scalar(value)}
                root[current_key].append(current_item)
            else:
                current_item = None
                root[current_key].append(_parse_scalar(rest))
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        container = root.get(current_key)
        if isinstance(container, list) and current_item is not None:
            current_item[key.strip()] = _parse_scalar(value)
        else:
            if not isinstance(container, dict):
                container = {}
                root[current_key] = container
            container[key.strip()] = _parse_scalar(value)

    return root


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return loaded or {}
    except ImportError:
        return _load_simple_yaml(config_path)


def scenario_config_from_dict(raw: dict[str, Any]) -> RemoteSensingScenarioConfig:
    target_items = raw.get("targets") or []
    targets: list[GroundPoint] = []
    for idx in range(int(raw.get("num_targets", len(target_items) or 3))):
        item = target_items[idx] if idx < len(target_items) else {}
        default = DEFAULT_TARGET_POINTS[idx % len(DEFAULT_TARGET_POINTS)]
        targets.append(_ground_point_from_config(item, default))

    es_items = raw.get("earth_stations") or raw.get("es") or []
    es_points: list[GroundPoint] = []
    for idx in range(int(raw.get("num_es", len(es_items) or 3))):
        item = es_items[idx] if idx < len(es_items) else {}
        default = DEFAULT_ES_POINTS[idx % len(DEFAULT_ES_POINTS)]
        es_points.append(_ground_point_from_config(item, default))

    weights_raw = raw.get("routing_weights") or {}
    weights = RoutingWeights(
        delay=float(weights_raw.get("delay", 0.4)),
        energy=float(weights_raw.get("energy", 0.2)),
        queue=float(weights_raw.get("queue", 0.3)),
        loss=float(weights_raw.get("loss", 0.1)),
    ).normalized()

    gs_weights_raw = raw.get("gs_selection_weights") or {}
    gs_selection_weights = GSSelectionWeights(
        path_delay=float(gs_weights_raw.get("path_delay", 0.35)),
        exit_reachability=float(gs_weights_raw.get("exit_reachability", 0.20)),
        visibility_window=float(gs_weights_raw.get("visibility_window", 0.20)),
        assigned_load=float(gs_weights_raw.get("assigned_load", 0.25)),
    ).normalized()

    task_profile_items = raw.get("task_profiles") or []
    task_profiles: list[TaskProfile] = []
    for idx, item in enumerate(task_profile_items):
        if not isinstance(item, dict):
            raise ValueError(f"task_profiles[{idx}] must be a mapping.")
        default = DEFAULT_TASK_PROFILES[idx % len(DEFAULT_TASK_PROFILES)]
        task_profiles.append(
            TaskProfile(
                name=str(item.get("name", default.name)),
                min_data_mb=float(item.get("min_data_mb", default.min_data_mb)),
                max_data_mb=float(item.get("max_data_mb", default.max_data_mb)),
                traffic_class=int(item.get("traffic_class", default.traffic_class)),
                priority=int(item.get("priority", default.priority)),
                deadline_s=float(item.get("deadline_s", default.deadline_s)),
                probability=float(item.get("probability", default.probability)),
            )
        )
    if not task_profiles:
        task_profiles = list(DEFAULT_TASK_PROFILES)

    access_policy = str(raw.get("rls_access_policy", "best_quality"))
    if access_policy not in ACCESS_POLICIES:
        raise ValueError(f"Unsupported rls_access_policy={access_policy!r}. Expected one of {sorted(ACCESS_POLICIES)}.")

    beam_raw = raw.get("beam") or {}
    beam = BeamConfig(
        enabled=bool(beam_raw.get("enabled", True)),
        pointing_mode=str(beam_raw.get("pointing_mode", "nadir")),
        half_angle_deg=float(beam_raw.get("half_angle_deg", 20.0)),
        max_off_nadir_deg=float(beam_raw.get("max_off_nadir_deg", 35.0)),
        max_observation_range_km=float(beam_raw.get("max_observation_range_km", 3000.0)),
        gain_max_db=float(beam_raw.get("gain_max_db", 35.0)),
        three_db_width_deg=float(beam_raw.get("three_db_width_deg", 6.0)),
        side_lobe_attenuation_db=float(beam_raw.get("side_lobe_attenuation_db", 25.0)),
        max_observation_beams_per_rls=int(beam_raw.get("max_observation_beams_per_rls", 1)),
        max_rls_per_target=int(beam_raw.get("max_rls_per_target", 1)),
        base_observation_delay_ms=float(beam_raw.get("base_observation_delay_ms", 50.0)),
        base_observation_energy_j=float(beam_raw.get("base_observation_energy_j", 0.1)),
        quality_penalty_mode=str(beam_raw.get("quality_penalty_mode", "linear")),
        quality_delay_penalty=float(beam_raw.get("quality_delay_penalty", 2.0)),
        quality_energy_penalty=float(beam_raw.get("quality_energy_penalty", 2.0)),
        min_quality_for_penalty=float(beam_raw.get("min_quality_for_penalty", 0.2)),
    )

    cfg = RemoteSensingScenarioConfig(
        time_slot_seconds=float(raw.get("time_slot_seconds", 1.0)),
        num_targets=int(raw.get("num_targets", len(targets))),
        targets=tuple(targets),
        rls_total_sats=int(raw.get("rls_total_sats", raw.get("num_rls", 18))),
        rls_num_planes=int(raw.get("rls_num_planes", 3)),
        rls_sats_per_plane=int(raw.get("rls_sats_per_plane", 6)),
        rls_altitude_km=float(raw.get("rls_altitude_km", 580.0)),
        rls_inclination_deg=float(raw.get("rls_inclination_deg", 97.7)),
        rls_phase_offset=int(raw.get("rls_phase_offset", 1)),
        rls_is_agent=bool(raw.get("rls_is_agent", False)),
        allow_rls_to_rls_edge=bool(raw.get("allow_rls_to_rls_edge", True)),
        allow_rls_to_rls_data_flow=bool(raw.get("allow_rls_to_rls_data_flow", False)),
        cls_total_sats=int(raw.get("cls_total_sats", raw.get("num_cls", 15))),
        cls_num_planes=int(raw.get("cls_num_planes", 5)),
        cls_sats_per_plane=int(raw.get("cls_sats_per_plane", 3)),
        cls_altitude_km=float(raw.get("cls_altitude_km", 1150.0)),
        cls_inclination_deg=float(raw.get("cls_inclination_deg", 53.0)),
        cls_phase_offset=int(raw.get("cls_phase_offset", 1)),
        cls_is_agent=bool(raw.get("cls_is_agent", True)),
        num_es=int(raw.get("num_es", len(es_points))),
        es_points=tuple(es_points),
        es_is_agent=bool(raw.get("es_is_agent", False)),
        allow_rls_to_es=bool(raw.get("allow_rls_to_es", False)),
        allow_cls_to_rls=bool(raw.get("allow_cls_to_rls", False)),
        allow_es_to_any=bool(raw.get("allow_es_to_any", False)),
        rls_access_policy=access_policy,
        max_access_cls_per_rls=int(raw.get("max_access_cls_per_rls", 0)),
        min_cls_relay_hops=int(raw.get("min_cls_relay_hops", 0)),
        routing_weights=weights,
        min_elevation_deg=float(raw.get("min_elevation_deg", 10.0)),
        beam=beam,
        max_rls_isl_distance_km=float(raw.get("max_rls_isl_distance_km", 12_000.0)),
        max_cls_isl_distance_km=float(raw.get("max_cls_isl_distance_km", 16_000.0)),
        max_cross_layer_distance_km=float(raw.get("max_cross_layer_distance_km", 20_000.0)),
        max_downlink_distance_km=float(raw.get("max_downlink_distance_km", 20_000.0)),
        packet_count_per_task=float(raw.get("packet_count_per_task", 16.0)),
        packet_class=int(raw.get("packet_class", 0)),
        num_traffic_classes=int(raw.get("num_traffic_classes", 3)),
        queue_capacities=_parse_tuple(raw.get("queue_capacities"), (100, 100, 100), int),
        priority_levels=_parse_tuple(raw.get("priority_levels"), (0, 1, 2), int),
        default_aqm=_parse_tuple(raw.get("default_aqm"), (0.4, 0.8, 0.1), float),
        background_packets_per_cls=float(raw.get("background_packets_per_cls", 0.0)),
        background_packet_class=int(raw.get("background_packet_class", raw.get("packet_class", 0))),
        max_concurrent_tasks=int(raw.get("max_concurrent_tasks", 4)),
        task_profiles=tuple(task_profiles),
        gs_selection_weights=gs_selection_weights,
        gs_lookahead_slots=int(raw.get("gs_lookahead_slots", 20)),
        drain_slots=int(raw.get("drain_slots", 20)),
        downlink_queue_capacities=_parse_tuple(
            raw.get("downlink_queue_capacities"),
            (20_000, 50_000, 100_000),
            int,
        ),
        downlink_wpq_weights=_parse_tuple(raw.get("downlink_wpq_weights"), (1.0, 1.0, 1.0), float),
        normalizer_delay_ms=float(raw.get("normalizer_delay_ms", 120.0)),
        normalizer_energy_j=float(raw.get("normalizer_energy_j", 1.0)),
        normalizer_queue_packets=float(raw.get("normalizer_queue_packets", 100.0)),
        normalizer_loss=float(raw.get("normalizer_loss", 1.0)),
        reward_cost_clip=float(raw.get("reward_cost_clip", 3.0)),
        reward_success_bonus=float(raw.get("reward_success_bonus", 1.0)),
        reward_failure_penalty=float(raw.get("reward_failure_penalty", 1.0)),
        reward_terminal_cls_delay_weight=float(raw.get("reward_terminal_cls_delay_weight", 1.0)),
        reward_terminal_cls_delay_norm_ms=float(raw.get("reward_terminal_cls_delay_norm_ms", 30.0)),
    )
    _validate_config(cfg)
    return cfg


def load_scenario_config(path: str | Path) -> RemoteSensingScenarioConfig:
    return scenario_config_from_dict(load_config_file(path))


def _validate_config(cfg: RemoteSensingScenarioConfig) -> None:
    if cfg.rls_num_planes * cfg.rls_sats_per_plane != cfg.rls_total_sats:
        raise ValueError("RLS plane/satellite counts must multiply to rls_total_sats.")
    if cfg.cls_num_planes * cfg.cls_sats_per_plane != cfg.cls_total_sats:
        raise ValueError("CLS plane/satellite counts must multiply to cls_total_sats.")
    if cfg.rls_is_agent:
        raise ValueError("RLS must not be configured as an agent in this first-stage scenario.")
    if not cfg.cls_is_agent:
        raise ValueError("CLS must be configured as agent-capable in this first-stage scenario.")
    if cfg.es_is_agent:
        raise ValueError("ES must not be configured as an agent.")
    if cfg.allow_rls_to_es:
        raise ValueError("RLS->ES main access is forbidden by the scenario definition.")
    if cfg.allow_cls_to_rls:
        raise ValueError("CLS->RLS forwarding is forbidden by the scenario definition.")
    if cfg.allow_es_to_any:
        raise ValueError("ES outgoing links are forbidden by the scenario definition.")
    if cfg.max_access_cls_per_rls < 0:
        raise ValueError("max_access_cls_per_rls must be non-negative; use 0 for unlimited access.")
    if cfg.min_cls_relay_hops < 0:
        raise ValueError("min_cls_relay_hops must be non-negative.")
    if not 0 <= cfg.packet_class < cfg.num_traffic_classes:
        raise ValueError("packet_class must be within num_traffic_classes.")
    if not 0 <= cfg.background_packet_class < cfg.num_traffic_classes:
        raise ValueError("background_packet_class must be within num_traffic_classes.")
    if cfg.background_packets_per_cls < 0.0:
        raise ValueError("background_packets_per_cls must be non-negative.")
    if cfg.max_concurrent_tasks <= 0 or cfg.max_concurrent_tasks > 4:
        raise ValueError("max_concurrent_tasks must be between 1 and 4.")
    if len(cfg.queue_capacities) != cfg.num_traffic_classes:
        raise ValueError("queue_capacities length must match num_traffic_classes.")
    if len(cfg.priority_levels) != cfg.num_traffic_classes:
        raise ValueError("priority_levels length must match num_traffic_classes.")
    if len(cfg.default_aqm) != 3:
        raise ValueError("default_aqm must contain alpha, beta, and pmax.")
    if any(capacity <= 0 for capacity in cfg.queue_capacities):
        raise ValueError("queue_capacities values must be positive.")
    if len(cfg.downlink_queue_capacities) != cfg.num_traffic_classes:
        raise ValueError("downlink_queue_capacities length must match num_traffic_classes.")
    if len(cfg.downlink_wpq_weights) != cfg.num_traffic_classes:
        raise ValueError("downlink_wpq_weights length must match num_traffic_classes.")
    if any(capacity <= 0 for capacity in cfg.downlink_queue_capacities):
        raise ValueError("downlink_queue_capacities values must be positive.")
    if cfg.gs_lookahead_slots <= 0 or cfg.drain_slots < 0:
        raise ValueError("gs_lookahead_slots must be positive and drain_slots must be non-negative.")
    if not cfg.task_profiles:
        raise ValueError("At least one task profile is required.")
    probability_sum = sum(profile.probability for profile in cfg.task_profiles)
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError("task profile probabilities must sum to 1.0.")
    for profile in cfg.task_profiles:
        if profile.min_data_mb <= 0.0 or profile.max_data_mb < profile.min_data_mb:
            raise ValueError(f"Invalid data range for task profile {profile.name!r}.")
        if not 0 <= profile.traffic_class < cfg.num_traffic_classes:
            raise ValueError(f"Invalid traffic_class for task profile {profile.name!r}.")
        if profile.deadline_s <= 0.0 or profile.probability < 0.0:
            raise ValueError(f"Invalid deadline/probability for task profile {profile.name!r}.")
    if cfg.beam.pointing_mode.lower() not in {"nadir", "steerable"}:
        raise ValueError("beam.pointing_mode must be one of: nadir, steerable.")
    if cfg.beam.half_angle_deg < 0.0:
        raise ValueError("beam.half_angle_deg must be non-negative.")
    if cfg.beam.max_off_nadir_deg < 0.0:
        raise ValueError("beam.max_off_nadir_deg must be non-negative.")
    if cfg.beam.max_observation_range_km <= 0.0:
        raise ValueError("beam.max_observation_range_km must be positive.")
    if cfg.beam.three_db_width_deg <= 0.0:
        raise ValueError("beam.three_db_width_deg must be positive.")
    if cfg.beam.max_observation_beams_per_rls < 0:
        raise ValueError("beam.max_observation_beams_per_rls must be non-negative.")
    if cfg.beam.max_rls_per_target < 0:
        raise ValueError("beam.max_rls_per_target must be non-negative.")
    if cfg.beam.base_observation_delay_ms < 0.0 or cfg.beam.base_observation_energy_j < 0.0:
        raise ValueError("beam base observation delay/energy must be non-negative.")
    if cfg.beam.quality_penalty_mode.lower() not in {"none", "linear", "inverse"}:
        raise ValueError("beam.quality_penalty_mode must be one of: none, linear, inverse.")
    if cfg.beam.quality_delay_penalty < 0.0 or cfg.beam.quality_energy_penalty < 0.0:
        raise ValueError("beam quality delay/energy penalties must be non-negative.")
    if not 0.0 <= cfg.beam.min_quality_for_penalty <= 1.0:
        raise ValueError("beam.min_quality_for_penalty must be in [0, 1].")
    for point in (*cfg.targets, *cfg.es_points):
        if point.motion_type not in GROUND_MOTION_TYPES:
            raise ValueError(
                f"Unsupported motion_type={point.motion_type!r} for {point.name}. "
                f"Expected one of {sorted(GROUND_MOTION_TYPES)}."
            )
        if point.speed_mps < 0.0:
            raise ValueError(f"speed_mps must be non-negative for {point.name}.")


class RemoteSensingScenario:
    def __init__(self, config: RemoteSensingScenarioConfig):
        self.config = config
        self.link_config = LinkModelConfig()
        self.device = torch.device("cpu")
        self.queue_capacities = torch.tensor(config.queue_capacities, dtype=torch.float32)
        self.priority_levels = torch.tensor(config.priority_levels, dtype=torch.float32)
        self.default_wpq_weights = torch.ones((config.cls_total_sats, config.num_traffic_classes), dtype=torch.float32)
        self.default_aqm = torch.tensor(config.default_aqm, dtype=torch.float32).unsqueeze(0).repeat(
            config.cls_total_sats,
            1,
        )
        self.queue_lengths = torch.zeros((config.cls_total_sats, config.num_traffic_classes), dtype=torch.float32)
        self.cls_node_ids: list[int] = []
        self.node_by_id: dict[int, ScenarioNode] = {}
        self.cls_queue_delay_ms: dict[int, float] = {}
        self.cls_loss_risk: dict[int, float] = {}
        self.cls_queue_length: dict[int, float] = {}
        self.cls_service_rates_pps = torch.zeros(config.cls_total_sats, dtype=torch.float32)

    def run(
        self,
        num_time_slots: int = 1,
        *,
        random_seed: int = 42,
        drain_slots: int | None = None,
        ttl_cap: int = 10,
    ) -> ScenarioResult:
        """Run the upgraded multi-source scenario with deterministic Dijkstra routing."""

        from env.remote_sensing_random_baseline import (  # Local import avoids a module cycle.
            RandomBaselineConfig,
            run_random_baseline,
        )

        effective_drain = self.config.drain_slots if drain_slots is None else int(drain_slots)
        routes, summary, all_edges = run_random_baseline(
            self.config,
            num_time_slots,
            RandomBaselineConfig(
                seed=int(random_seed),
                ttl_cap=int(ttl_cap),
                drain_slots=effective_drain,
                variant="dijkstra",
            ),
        )
        summary.update(
            {
                "num_tasks": summary["task_count"],
                "success_tasks": summary["success_count"],
                "access_failed_tasks": sum(
                    1 for route in routes if not route.get("access_success")
                ),
                "route_failed_tasks": sum(
                    1
                    for route in routes
                    if route.get("access_success") and not route.get("route_success")
                ),
            }
        )
        node_scenario = RemoteSensingScenario(self.config)
        all_node_records: list[ScenarioNodeRecord] = []
        for time_slot in range(max(1, int(num_time_slots)) + effective_drain):
            snapshot = node_scenario.build_snapshot(time_slot)
            all_node_records.extend(
                _node_record(time_slot, node) for node in snapshot.nodes
            )
        return ScenarioResult(
            routes=routes,
            topology_edges=all_edges,
            metrics_summary=summary,
            topology_nodes=all_node_records,
        )

    def build_snapshot(self, time_slot: int) -> ScenarioSnapshot:
        cfg = self.config
        time_sec = time_slot * cfg.time_slot_seconds
        nodes = self._build_nodes(time_sec)
        self.node_by_id = {node.node_id: node for node in nodes}
        self.cls_node_ids = [node.node_id for node in nodes if node.node_type == NODE_CLS]

        edges: list[ScenarioEdge] = []
        edges.extend(self._build_observation_edges(time_slot, nodes))
        edges.extend(self._build_rls_backup_edges(time_slot, nodes))
        edges.extend(self._build_access_edges(time_slot, nodes))
        edges.extend(self._build_cls_isl_edges(time_slot, nodes))
        edges.extend(self._build_downlink_edges(time_slot, nodes))

        self._refresh_cls_queue_metrics(edges)
        for edge in edges:
            owner = self._edge_queue_owner(edge)
            if owner is not None:
                edge.queue_length = self.cls_queue_length.get(owner, 0.0)
                edge.loss_risk = self.cls_loss_risk.get(owner, 0.0)

        return ScenarioSnapshot(time_slot=time_slot, time_sec=time_sec, nodes=nodes, edges=edges)

    def _build_nodes(self, time_sec: float) -> list[ScenarioNode]:
        cfg = self.config
        nodes: list[ScenarioNode] = []
        next_id = 0

        for idx, target in enumerate(cfg.targets):
            lat_deg, lon_deg = _ground_latlon_at_time(target, time_sec)
            nodes.append(
                ScenarioNode(
                    node_id=next_id,
                    label=target.name,
                    node_type=NODE_TARGET,
                    position_km=latlon_to_ecef_km(lat_deg, lon_deg, target.altitude_km),
                    local_id=idx,
                    is_agent=False,
                    has_queue=False,
                    can_relay=False,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    frame="ECEF",
                    motion_type=target.motion_type,
                )
            )
            next_id += 1

        rls_positions_eci = generate_walker_constellation(
            num_planes=cfg.rls_num_planes,
            sats_per_plane=cfg.rls_sats_per_plane,
            altitude_km=cfg.rls_altitude_km,
            inclination_deg=cfg.rls_inclination_deg,
            phase_offset=cfg.rls_phase_offset,
            time_sec=time_sec,
        )
        rls_positions = eci_to_ecef(rls_positions_eci, time_sec)
        for local_id, position in enumerate(rls_positions):
            lat_deg, lon_deg = _ecef_to_latlon_deg(position)
            nodes.append(
                ScenarioNode(
                    node_id=next_id,
                    label=f"RLS{local_id + 1}",
                    node_type=NODE_RLS,
                    position_km=position,
                    local_id=local_id,
                    plane_id=local_id // cfg.rls_sats_per_plane,
                    sat_id_in_plane=local_id % cfg.rls_sats_per_plane,
                    is_agent=False,
                    has_queue=False,
                    can_relay=False,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    frame="ECEF",
                    motion_type="orbital",
                )
            )
            next_id += 1

        cls_positions_eci = generate_walker_constellation(
            num_planes=cfg.cls_num_planes,
            sats_per_plane=cfg.cls_sats_per_plane,
            altitude_km=cfg.cls_altitude_km,
            inclination_deg=cfg.cls_inclination_deg,
            phase_offset=cfg.cls_phase_offset,
            time_sec=time_sec,
        )
        cls_positions = eci_to_ecef(cls_positions_eci, time_sec)
        for local_id, position in enumerate(cls_positions):
            lat_deg, lon_deg = _ecef_to_latlon_deg(position)
            nodes.append(
                ScenarioNode(
                    node_id=next_id,
                    label=f"CLS{local_id + 1}",
                    node_type=NODE_CLS,
                    position_km=position,
                    local_id=local_id,
                    plane_id=local_id // cfg.cls_sats_per_plane,
                    sat_id_in_plane=local_id % cfg.cls_sats_per_plane,
                    is_agent=True,
                    has_queue=True,
                    can_relay=True,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    frame="ECEF",
                    motion_type="orbital",
                )
            )
            next_id += 1

        for idx, es in enumerate(cfg.es_points):
            lat_deg, lon_deg = _ground_latlon_at_time(es, time_sec)
            nodes.append(
                ScenarioNode(
                    node_id=next_id,
                    label=es.name,
                    node_type=NODE_ES,
                    position_km=latlon_to_ecef_km(lat_deg, lon_deg, es.altitude_km),
                    local_id=idx,
                    is_agent=False,
                    has_queue=False,
                    can_relay=False,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    frame="ECEF",
                    motion_type=es.motion_type,
                )
            )
            next_id += 1

        return nodes

    def _build_observation_edges(self, time_slot: int, nodes: list[ScenarioNode]) -> list[ScenarioEdge]:
        targets = [node for node in nodes if node.node_type == NODE_TARGET]
        rls_nodes = [node for node in nodes if node.node_type == NODE_RLS]
        candidates: list[tuple[ScenarioNode, ScenarioNode, BeamResult]] = []
        for target in targets:
            for rls in rls_nodes:
                beam_result = compute_rls_beam_to_target(
                    rls_position_ecef_km=rls.position_km,
                    target_position_ecef_km=target.position_km,
                    min_elevation_deg=self.config.min_elevation_deg,
                    beam_config=self.config.beam,
                )
                if beam_result.available:
                    candidates.append((target, rls, beam_result))

        candidates.sort(
            key=lambda item: (
                -item[2].observation_quality,
                item[2].off_nadir_angle_deg,
                item[2].slant_range_km,
                item[0].label,
                item[1].label,
            )
        )

        max_beams_per_rls = max(int(self.config.beam.max_observation_beams_per_rls), 0)
        max_rls_per_target = max(int(self.config.beam.max_rls_per_target), 0)
        rls_counts: dict[int, int] = {}
        target_counts: dict[int, int] = {}
        edges: list[ScenarioEdge] = []
        for target, rls, beam_result in candidates:
            if rls_counts.get(rls.node_id, 0) >= max_beams_per_rls:
                continue
            if target_counts.get(target.node_id, 0) >= max_rls_per_target:
                continue
            edges.append(
                self._make_edge(
                    time_slot=time_slot,
                    src=target,
                    dst=rls,
                    link_type=LINK_OBSERVATION,
                    is_data_flow_allowed=True,
                    communication_metric=False,
                    beam_result=beam_result,
                )
            )
            rls_counts[rls.node_id] = rls_counts.get(rls.node_id, 0) + 1
            target_counts[target.node_id] = target_counts.get(target.node_id, 0) + 1
        return edges

    def _build_rls_backup_edges(self, time_slot: int, nodes: list[ScenarioNode]) -> list[ScenarioEdge]:
        cfg = self.config
        if not cfg.allow_rls_to_rls_edge:
            return []
        rls_nodes = [node for node in nodes if node.node_type == NODE_RLS]
        src_local, dst_local = compute_grid_plus_isl(cfg.rls_num_planes, cfg.rls_sats_per_plane, torch.stack([n.position_km for n in rls_nodes]))
        edges: list[ScenarioEdge] = []
        for src_idx, dst_idx in zip(src_local, dst_local):
            src = rls_nodes[src_idx]
            dst = rls_nodes[dst_idx]
            if self._space_link_available(src, dst, cfg.max_rls_isl_distance_km):
                edges.append(
                    self._make_edge(
                        time_slot=time_slot,
                        src=src,
                        dst=dst,
                        link_type=LINK_RLS_BACKUP,
                        is_data_flow_allowed=cfg.allow_rls_to_rls_data_flow,
                    )
                )
        return edges

    def _build_access_edges(self, time_slot: int, nodes: list[ScenarioNode]) -> list[ScenarioEdge]:
        rls_nodes = [node for node in nodes if node.node_type == NODE_RLS]
        cls_nodes = [node for node in nodes if node.node_type == NODE_CLS]
        edges: list[ScenarioEdge] = []
        for rls in rls_nodes:
            candidates: list[ScenarioEdge] = []
            for cls in cls_nodes:
                if self._space_link_available(rls, cls, self.config.max_cross_layer_distance_km):
                    candidates.append(
                        self._make_edge(
                            time_slot=time_slot,
                            src=rls,
                            dst=cls,
                            link_type=LINK_ACCESS,
                            is_data_flow_allowed=True,
                        )
                    )
            candidates.sort(key=self._access_candidate_sort_key)
            limit = self.config.max_access_cls_per_rls
            if limit > 0:
                candidates = candidates[:limit]
            edges.extend(candidates)
        return edges

    def _build_cls_isl_edges(self, time_slot: int, nodes: list[ScenarioNode]) -> list[ScenarioEdge]:
        cfg = self.config
        cls_nodes = [node for node in nodes if node.node_type == NODE_CLS]
        src_local, dst_local = compute_grid_plus_isl(cfg.cls_num_planes, cfg.cls_sats_per_plane, torch.stack([n.position_km for n in cls_nodes]))
        edges: list[ScenarioEdge] = []
        for src_idx, dst_idx in zip(src_local, dst_local):
            src = cls_nodes[src_idx]
            dst = cls_nodes[dst_idx]
            if self._space_link_available(src, dst, cfg.max_cls_isl_distance_km):
                edges.append(
                    self._make_edge(
                        time_slot=time_slot,
                        src=src,
                        dst=dst,
                        link_type=LINK_ISL,
                        is_data_flow_allowed=True,
                    )
                )
        return edges

    def _build_downlink_edges(self, time_slot: int, nodes: list[ScenarioNode]) -> list[ScenarioEdge]:
        cls_nodes = [node for node in nodes if node.node_type == NODE_CLS]
        es_nodes = [node for node in nodes if node.node_type == NODE_ES]
        edges: list[ScenarioEdge] = []
        for cls in cls_nodes:
            for es in es_nodes:
                if self._satellite_can_see_ground(cls, es):
                    distance = float(torch.norm(cls.position_km - es.position_km).item())
                    if distance <= self.config.max_downlink_distance_km:
                        edges.append(
                            self._make_edge(
                                time_slot=time_slot,
                                src=cls,
                                dst=es,
                                link_type=LINK_DOWNLINK,
                                is_data_flow_allowed=True,
                            )
                        )
        return edges

    def _make_edge(
        self,
        time_slot: int,
        src: ScenarioNode,
        dst: ScenarioNode,
        link_type: str,
        is_data_flow_allowed: bool,
        communication_metric: bool = True,
        beam_result: BeamResult | None = None,
    ) -> ScenarioEdge:
        distance_km = float(torch.norm(src.position_km - dst.position_km).item())
        if link_type == LINK_OBSERVATION and beam_result is not None:
            delay_ms = beam_result.observation_delay_ms
            capacity_bps = 0.0
            energy_cost = beam_result.observation_energy_j
        elif communication_metric:
            delay_ms, capacity_bps, energy_cost = self._communication_metrics(distance_km)
        else:
            delay_ms = 0.0
            capacity_bps = 0.0
            energy_cost = 0.0
        return ScenarioEdge(
            time_slot=time_slot,
            src_id=src.node_id,
            dst_id=dst.node_id,
            src=src.label,
            dst=dst.label,
            src_type=src.node_type,
            dst_type=dst.node_type,
            link_type=link_type,
            distance_km=distance_km,
            delay_ms=delay_ms,
            capacity_bps=capacity_bps,
            energy_cost=energy_cost,
            queue_length=0.0,
            loss_risk=0.0,
            available=True,
            is_data_flow_allowed=is_data_flow_allowed,
            elevation_deg=beam_result.elevation_deg if beam_result is not None else 0.0,
            off_nadir_angle_deg=beam_result.off_nadir_angle_deg if beam_result is not None else 0.0,
            slant_range_km=beam_result.slant_range_km if beam_result is not None else 0.0,
            beam_gain_db=beam_result.beam_gain_db if beam_result is not None else 0.0,
            observation_quality=beam_result.observation_quality if beam_result is not None else 0.0,
            beam_available=beam_result.available if beam_result is not None else False,
            beam_reason=beam_result.reason if beam_result is not None else "",
        )

    def _communication_metrics(self, distance_km: float) -> tuple[float, float, float]:
        distance = torch.tensor(distance_km, dtype=torch.float32)
        available = torch.tensor(True)
        _, _, rate_bps = compute_link_rates_bps(distance, available, self.link_config)
        prop_delay_ms = compute_propagation_delay_ms(distance, available)
        tx_delay_ms = compute_transmission_delay_ms(
            torch.tensor(self.config.packet_count_per_task, dtype=torch.float32),
            rate_bps,
            self.link_config.packet_size_bits,
        )
        tx_energy, duration_s = compute_transmission_energy_j(
            torch.tensor(self.config.packet_count_per_task, dtype=torch.float32),
            rate_bps,
            available,
            self.link_config,
        )
        rx_energy = compute_reception_energy_j(duration_s, available, self.link_config)
        return (
            float((prop_delay_ms + tx_delay_ms).item()),
            float(rate_bps.item()),
            float((tx_energy + rx_energy).item()),
        )

    def _space_link_available(self, src: ScenarioNode, dst: ScenarioNode, max_distance_km: float) -> bool:
        distance = float(torch.norm(src.position_km - dst.position_km).item())
        if distance > max_distance_km:
            return False
        visible = check_line_of_sight(src.position_km, dst.position_km)
        return bool(torch.as_tensor(visible).item())

    def _satellite_can_see_ground(self, sat: ScenarioNode, ground: ScenarioNode) -> bool:
        visible = check_elevation_constraint(
            sat.position_km,
            ground.position_km,
            min_elevation_deg=self.config.min_elevation_deg,
        )
        return bool(torch.as_tensor(visible).item())

    def _refresh_cls_queue_metrics(self, edges: list[ScenarioEdge]) -> None:
        rates = torch.zeros(self.config.cls_total_sats, dtype=torch.float32)
        for edge in edges:
            if edge.src_type == NODE_CLS and edge.is_data_flow_allowed:
                cls_idx = self.node_by_id[edge.src_id].local_id
                rates[cls_idx] += float(edge.capacity_bps)

        service_by_class, active_mask = allocate_wpq_service(
            rates,
            self.queue_lengths,
            self.priority_levels,
            self.default_wpq_weights,
        )
        queue_delay_by_class = compute_per_class_queueing_delay_ms(
            self.queue_lengths,
            service_by_class,
            active_mask,
        )
        aggregate_delay = compute_aggregate_queueing_delay_ms(queue_delay_by_class, active_mask)
        drop_probs = compute_aqm_drop_probabilities(
            self.queue_lengths,
            self.queue_capacities,
            self.default_aqm[:, 0],
            self.default_aqm[:, 1],
            self.default_aqm[:, 2],
        )

        self.cls_service_rates_pps = rates
        self.cls_queue_delay_ms = {}
        self.cls_queue_length = {}
        self.cls_loss_risk = {}
        for node_id in self.cls_node_ids:
            local_id = self.node_by_id[node_id].local_id
            self.cls_queue_delay_ms[node_id] = float(aggregate_delay[local_id].item())
            self.cls_queue_length[node_id] = float(self.queue_lengths[local_id].sum().item())
            self.cls_loss_risk[node_id] = float(drop_probs[local_id].max().item())

    def _edge_queue_owner(self, edge: ScenarioEdge) -> int | None:
        if edge.src_type == NODE_CLS:
            return edge.src_id
        if edge.link_type == LINK_ACCESS and edge.dst_type == NODE_CLS:
            return edge.dst_id
        return None

    def route_task(self, snapshot: ScenarioSnapshot, target: ScenarioNode, task_index: int) -> dict[str, Any]:
        observation_edges = [
            edge for edge in snapshot.edges
            if edge.src_id == target.node_id and edge.link_type == LINK_OBSERVATION
        ]
        access_attempts: list[tuple[float, ScenarioEdge, ScenarioEdge]] = []
        route_candidates: list[tuple[float, ScenarioEdge, ScenarioEdge, list[int], list[int], dict[str, float]]] = []
        for obs_edge in observation_edges:
            access_edges = [
                edge for edge in snapshot.edges
                if edge.src_id == obs_edge.dst_id and edge.link_type == LINK_ACCESS and edge.is_data_flow_allowed
            ]
            for access_edge in access_edges:
                attempt_cost = self._edge_route_cost(obs_edge) + self._access_selection_cost(access_edge)
                access_attempts.append((attempt_cost, obs_edge, access_edge))
                cls_path = self._shortest_cls_path_to_es(snapshot, access_edge.dst_id)
                if not cls_path:
                    continue
                full_node_ids = [target.node_id, obs_edge.dst_id] + cls_path
                full_edges = self._resolve_route_edges(snapshot, full_node_ids)
                route_metrics = self._route_metrics(full_edges)
                route_candidates.append(
                    (
                        route_metrics["cost"],
                        obs_edge,
                        access_edge,
                        cls_path,
                        full_node_ids,
                        route_metrics,
                    )
                )

        if not access_attempts:
            return self._empty_route(
                snapshot=snapshot,
                target=target,
                task_index=task_index,
                access_success=False,
                route_success=False,
            )

        if not route_candidates:
            _, observation_edge, access_edge = min(access_attempts, key=lambda item: item[0])
            return self._empty_route(
                snapshot=snapshot,
                target=target,
                task_index=task_index,
                access_success=True,
                route_success=False,
                rls_node=observation_edge.dst,
                selected_cls=access_edge.dst,
            )

        _, observation_edge, access_edge, cls_path, full_node_ids, route_metrics = min(
            route_candidates,
            key=lambda item: item[0],
        )
        used_rls_rls = self._route_uses_rls_rls(full_node_ids)
        cls_relay_hops = self._count_cls_relay_hops(full_node_ids)
        selected_es = self.node_by_id[cls_path[-1]].label

        return {
            "time_slot": snapshot.time_slot,
            "task_id": f"task_{task_index}",
            "target_name": target.label,
            "rls_node": observation_edge.dst,
            "selected_cls": access_edge.dst,
            "selected_es": selected_es,
            "path": self._format_path(full_node_ids),
            "access_success": True,
            "route_success": True,
            "used_rls_rls_data_flow": used_rls_rls,
            "total_delay_ms": route_metrics["delay_ms"],
            "total_energy": route_metrics["energy"],
            "total_queue_cost": route_metrics["queue"],
            "total_loss_risk": route_metrics["loss"],
            "total_cost": route_metrics["cost"],
            "hop_count": max(len(full_node_ids) - 1, 0),
            "cls_relay_hops": cls_relay_hops,
            "_path_node_ids": full_node_ids,
        }

    def _empty_route(
        self,
        snapshot: ScenarioSnapshot,
        target: ScenarioNode,
        task_index: int,
        access_success: bool,
        route_success: bool,
        rls_node: str = "",
        selected_cls: str = "",
    ) -> dict[str, Any]:
        return {
            "time_slot": snapshot.time_slot,
            "task_id": f"task_{task_index}",
            "target_name": target.label,
            "rls_node": rls_node,
            "selected_cls": selected_cls,
            "selected_es": "",
            "path": target.label,
            "access_success": access_success,
            "route_success": route_success,
            "used_rls_rls_data_flow": False,
            "total_delay_ms": 0.0,
            "total_energy": 0.0,
            "total_queue_cost": 0.0,
            "total_loss_risk": 0.0,
            "total_cost": 0.0,
            "hop_count": 0,
            "cls_relay_hops": 0,
            "_path_node_ids": [target.node_id],
        }

    def _select_access_edge(self, access_edges: list[ScenarioEdge]) -> ScenarioEdge | None:
        if not access_edges:
            return None
        if self.config.rls_access_policy == "nearest":
            return min(access_edges, key=lambda edge: edge.distance_km)
        if self.config.rls_access_policy == "min_delay":
            return min(access_edges, key=lambda edge: edge.delay_ms)
        return min(access_edges, key=self._edge_route_cost)

    def _access_selection_cost(self, edge: ScenarioEdge) -> float:
        if self.config.rls_access_policy == "nearest":
            return edge.distance_km
        if self.config.rls_access_policy == "min_delay":
            return edge.delay_ms
        return self._edge_route_cost(edge)

    def _access_candidate_sort_key(self, edge: ScenarioEdge) -> tuple[float, float, str]:
        return (
            self._access_selection_cost(edge),
            edge.distance_km,
            edge.dst,
        )

    def _shortest_cls_path_to_es(self, snapshot: ScenarioSnapshot, start_cls_id: int) -> list[int]:
        adjacency: dict[int, list[tuple[int, float, ScenarioEdge]]] = {}
        for edge in snapshot.edges:
            if not edge.is_data_flow_allowed:
                continue
            if edge.src_type != NODE_CLS:
                continue
            if edge.dst_type not in {NODE_CLS, NODE_ES}:
                continue
            adjacency.setdefault(edge.src_id, []).append((edge.dst_id, self._edge_route_cost(edge), edge))

        min_relays = self.config.min_cls_relay_hops
        start_path = (start_cls_id,)
        start_state = (start_cls_id, 0, start_path)
        distances = {start_state: 0.0}
        heap: list[tuple[float, int, int, tuple[int, ...]]] = [(0.0, start_cls_id, 0, start_path)]
        while heap:
            current_dist, node_id, relay_hops, path = heapq.heappop(heap)
            current_state = (node_id, relay_hops, path)
            if current_dist > distances.get(current_state, float("inf")):
                continue
            if self.node_by_id[node_id].node_type == NODE_ES:
                return list(path)
            for next_id, edge_cost, _ in adjacency.get(node_id, []):
                next_type = self.node_by_id[next_id].node_type
                if next_type == NODE_ES and relay_hops < min_relays:
                    continue
                next_relay_hops = relay_hops
                if next_type == NODE_CLS:
                    if next_id in path:
                        continue
                    next_relay_hops = min(min_relays, relay_hops + 1)
                next_path = path + (next_id,)
                next_state = (next_id, next_relay_hops, next_path)
                candidate = current_dist + edge_cost
                if candidate < distances.get(next_state, float("inf")):
                    distances[next_state] = candidate
                    heapq.heappush(heap, (candidate, next_id, next_relay_hops, next_path))

        return []

    def _edge_route_cost(self, edge: ScenarioEdge) -> float:
        weights = self.config.routing_weights
        queue_owner = self._edge_queue_owner(edge)
        queue_delay = self.cls_queue_delay_ms.get(queue_owner, 0.0) if queue_owner is not None else 0.0
        queue_length = self.cls_queue_length.get(queue_owner, 0.0) if queue_owner is not None else 0.0
        loss_risk = self.cls_loss_risk.get(queue_owner, 0.0) if queue_owner is not None else 0.0

        delay_norm = min((edge.delay_ms + queue_delay) / max(self.config.normalizer_delay_ms, 1e-9), 1.0)
        energy_norm = min(edge.energy_cost / max(self.config.normalizer_energy_j, 1e-9), 1.0)
        queue_norm = min(queue_length / max(self.config.normalizer_queue_packets, 1e-9), 1.0)
        loss_norm = min(loss_risk / max(self.config.normalizer_loss, 1e-9), 1.0)
        return (
            weights.delay * delay_norm
            + weights.energy * energy_norm
            + weights.queue * queue_norm
            + weights.loss * loss_norm
        )

    def _resolve_route_edges(self, snapshot: ScenarioSnapshot, node_ids: list[int]) -> list[ScenarioEdge]:
        resolved: list[ScenarioEdge] = []
        for src_id, dst_id in zip(node_ids, node_ids[1:]):
            candidates = [
                edge for edge in snapshot.edges
                if edge.src_id == src_id and edge.dst_id == dst_id
            ]
            if not candidates:
                continue
            resolved.append(candidates[0])
        return resolved

    def _route_metrics(self, edges: list[ScenarioEdge]) -> dict[str, float]:
        delay_ms = 0.0
        energy = 0.0
        queue_cost = 0.0
        loss_risk = 0.0
        total_cost = 0.0
        for edge in edges:
            queue_owner = self._edge_queue_owner(edge)
            delay_ms += edge.delay_ms
            if queue_owner is not None:
                delay_ms += self.cls_queue_delay_ms.get(queue_owner, 0.0)
                queue_cost += self.cls_queue_length.get(queue_owner, 0.0)
                loss_risk += self.cls_loss_risk.get(queue_owner, 0.0)
            energy += edge.energy_cost
            total_cost += self._edge_route_cost(edge)
        return {
            "delay_ms": delay_ms,
            "energy": energy,
            "queue": queue_cost,
            "loss": loss_risk,
            "cost": total_cost,
        }

    def _route_uses_rls_rls(self, node_ids: list[int]) -> bool:
        for src_id, dst_id in zip(node_ids, node_ids[1:]):
            if self.node_by_id[src_id].node_type == NODE_RLS and self.node_by_id[dst_id].node_type == NODE_RLS:
                return True
        return False

    def _count_cls_relay_hops(self, node_ids: list[int]) -> int:
        relay_hops = 0
        for src_id, dst_id in zip(node_ids, node_ids[1:]):
            if self.node_by_id[src_id].node_type == NODE_CLS and self.node_by_id[dst_id].node_type == NODE_CLS:
                relay_hops += 1
        return relay_hops

    def _format_path(self, node_ids: list[int]) -> str:
        return " -> ".join(self.node_by_id[node_id].label for node_id in node_ids)

    def _apply_background_traffic(self) -> None:
        background_packets = float(self.config.background_packets_per_cls)
        if background_packets <= 0.0:
            return
        class_idx = self.config.background_packet_class
        self.queue_lengths[:, class_idx] += background_packets
        capacities = self.queue_capacities.unsqueeze(0)
        self.queue_lengths = torch.minimum(self.queue_lengths, capacities)

    def _apply_successful_route(self, route: dict[str, Any]) -> None:
        packet_count = float(self.config.packet_count_per_task)
        cls_nodes_on_path = [
            self.node_by_id[node_id]
            for node_id in route.get("_path_node_ids", [])
            if self.node_by_id[node_id].node_type == NODE_CLS
        ]
        for node in cls_nodes_on_path:
            class_idx = self.config.packet_class
            self.queue_lengths[node.local_id, class_idx] += packet_count
        capacities = self.queue_capacities.unsqueeze(0)
        self.queue_lengths = torch.minimum(self.queue_lengths, capacities)

    def _service_cls_queues(self, snapshot: ScenarioSnapshot) -> None:
        self._refresh_cls_queue_metrics(snapshot.edges)
        service_by_class, active_mask = allocate_wpq_service(
            self.cls_service_rates_pps,
            self.queue_lengths,
            self.priority_levels,
            self.default_wpq_weights,
        )
        drop_probs = compute_aqm_drop_probabilities(
            self.queue_lengths,
            self.queue_capacities,
            self.default_aqm[:, 0],
            self.default_aqm[:, 1],
            self.default_aqm[:, 2],
        )
        after_aqm = self.queue_lengths * (1.0 - drop_probs)
        served = compute_served_packets(after_aqm, service_by_class, self.config.time_slot_seconds)
        zero_arrivals = torch.zeros_like(self.queue_lengths)
        update = update_queue_lengths(after_aqm, zero_arrivals, served, self.queue_capacities)
        self.queue_lengths = update.next_queue_lengths

    def _build_metrics_summary(self, routes: list[dict[str, Any]], edges: list[ScenarioEdge]) -> dict[str, Any]:
        success_routes = [route for route in routes if route["access_success"] and route["route_success"]]
        access_failed = [route for route in routes if not route["access_success"]]
        route_failed = [route for route in routes if route["access_success"] and not route["route_success"]]
        used_rls_rls_count = sum(1 for route in routes if route["used_rls_rls_data_flow"])
        used_cls_relay_count = sum(1 for route in success_routes if int(route.get("cls_relay_hops", 0)) > 0)

        def average(key: str) -> float:
            if not success_routes:
                return 0.0
            return float(sum(float(route[key]) for route in success_routes) / len(success_routes))

        return {
            "num_tasks": len(routes),
            "success_tasks": len(success_routes),
            "access_failed_tasks": len(access_failed),
            "route_failed_tasks": len(route_failed),
            "average_delay_ms": average("total_delay_ms"),
            "average_energy": average("total_energy"),
            "average_hop_count": average("hop_count"),
            "average_cls_relay_hops": average("cls_relay_hops"),
            "average_queue_cost": average("total_queue_cost"),
            "average_loss_risk": average("total_loss_risk"),
            "used_cls_relay_count": used_cls_relay_count,
            "num_rls_rls_edges": sum(1 for edge in edges if edge.link_type == LINK_RLS_BACKUP),
            "allow_rls_to_rls_data_flow": self.config.allow_rls_to_rls_data_flow,
            "used_rls_rls_data_flow_count": used_rls_rls_count,
            "max_access_cls_per_rls": self.config.max_access_cls_per_rls,
            "min_cls_relay_hops": self.config.min_cls_relay_hops,
            "background_packets_per_cls": self.config.background_packets_per_cls,
        }

def write_scenario_outputs(result: ScenarioResult, output_dir: str | Path) -> None:
    from env.remote_sensing_random_baseline import write_random_outputs

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_random_outputs(
        out_dir,
        result.routes,
        result.metrics_summary,
        result.topology_edges,
    )

    route_fieldnames = [
        "time_slot",
        "task_id",
        "target_name",
        "rls_node",
        "selected_cls",
        "selected_es",
        "path",
        "access_success",
        "route_success",
        "used_rls_rls_data_flow",
        "total_delay_ms",
        "total_energy",
        "total_queue_cost",
        "total_loss_risk",
        "total_cost",
        "hop_count",
        "cls_relay_hops",
    ]
    with (out_dir / "routes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=route_fieldnames)
        writer.writeheader()
        for route in result.routes:
            writer.writerow({field: route.get(field, "") for field in route_fieldnames})

    edge_fieldnames = [
        "time_slot",
        "src",
        "dst",
        "src_type",
        "dst_type",
        "link_type",
        "distance_km",
        "delay_ms",
        "capacity_bps",
        "energy_cost",
        "queue_length",
        "loss_risk",
        "available",
        "is_data_flow_allowed",
        "elevation_deg",
        "off_nadir_angle_deg",
        "slant_range_km",
        "beam_gain_db",
        "observation_quality",
        "beam_available",
        "beam_reason",
    ]
    with (out_dir / "topology_edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fieldnames)
        writer.writeheader()
        for edge in result.topology_edges:
            writer.writerow(edge.csv_row())

    node_fieldnames = [
        "time_slot",
        "node",
        "node_type",
        "lat_deg",
        "lon_deg",
        "x_km",
        "y_km",
        "z_km",
        "frame",
        "motion_type",
    ]
    with (out_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fieldnames)
        writer.writeheader()
        for node in result.topology_nodes:
            writer.writerow(node.csv_row())

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {key: value for key, value in result.metrics_summary.items() if not key.startswith("_")},
            f,
            indent=2,
        )

def _parse_cli_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the deterministic remote-sensing-driven CLS-layer routing scenario."
    )
    parser.add_argument(
        "--config",
        default=str(repo_root / "configs" / "remote_sensing_scenario.yaml"),
        help="Path to the remote sensing scenario YAML config.",
    )
    parser.add_argument(
        "--time-slot",
        "--time-slots",
        dest="time_slots",
        type=int,
        default=1,
        help="Number of deterministic time slots to simulate.",
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "outputs" / "remote_sensing_scenario"),
        help="Root output directory. A timestamped subdirectory will be created.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ttl-cap", type=int, default=10)
    parser.add_argument("--drain-slots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    config_path = Path(args.config)
    config = load_scenario_config(config_path) if config_path.exists() else scenario_config_from_dict({})
    scenario = RemoteSensingScenario(config)
    result = scenario.run(
        num_time_slots=args.time_slots,
        random_seed=args.random_seed,
        ttl_cap=args.ttl_cap,
        drain_slots=args.drain_slots,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    write_scenario_outputs(result, output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Routes: {output_dir / 'routes.csv'}")
    print(f"Topology edges: {output_dir / 'topology_edges.csv'}")
    print(f"Nodes: {output_dir / 'nodes.csv'}")
    print(f"Metrics: {output_dir / 'metrics_summary.json'}")
    print(
        "Summary: "
        f"tasks={result.metrics_summary['task_count']} "
        f"success={result.metrics_summary['success_count']} "
        f"success_rate={result.metrics_summary['success_rate']:.4f} "
        f"deadline_rate={result.metrics_summary['deadline_meeting_rate']:.4f} "
        f"throughput_mbps={result.metrics_summary['throughput_mbps']:.4f}"
    )


if __name__ == "__main__":
    main()
