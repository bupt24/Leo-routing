import math
import random

import torch

SPEED_OF_LIGHT_KM_PER_MS = 299.792458
EARTH_RADIUS_KM = 6371.0
MIN_ELEVATION_DEG = 25.0  # 最小仰角约束


# =============================================================================
# Walker-Delta 星座模型
# =============================================================================

def generate_walker_constellation(
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    phase_offset: int = 1,
    time_sec: float = 0.0,
) -> torch.Tensor:
    """
    生成 Walker-Delta 星座的卫星位置。
    
    :param num_planes: 轨道面数量
    :param sats_per_plane: 每个轨道面的卫星数
    :param altitude_km: 轨道高度 (km)
    :param inclination_deg: 轨道倾角 (度)
    :param phase_offset: 相邻轨道面的相位差因子 F (Walker T/P/F 中的 F)
    :param time_sec: 当前仿真时间 (秒)，用于计算卫星运动
    :return: 卫星位置张量 [num_sats, 3] (x, y, z in km)
    """
    radius = EARTH_RADIUS_KM + altitude_km
    inclination = math.radians(inclination_deg)
    
    # 轨道周期 (秒)，近似计算：T = 2π * sqrt(r³ / μ)
    # μ = 398600.4418 km³/s² (地球引力常数)
    mu = 398600.4418
    orbital_period = 2 * math.pi * math.sqrt(radius**3 / mu)
    
    # 角速度 (rad/s)
    omega = 2 * math.pi / orbital_period
    
    positions = []
    total_sats = num_planes * sats_per_plane
    
    for plane_idx in range(num_planes):
        # 升交点赤经 (RAAN)：均匀分布在 360 度
        raan = 2 * math.pi * plane_idx / num_planes
        
        for sat_idx in range(sats_per_plane):
            # 初始真近点角：均匀分布 + 相位偏移
            phase_shift = 2 * math.pi * phase_offset * plane_idx / total_sats
            initial_anomaly = 2 * math.pi * sat_idx / sats_per_plane + phase_shift
            
            # 当前真近点角（随时间演化）
            true_anomaly = initial_anomaly + omega * time_sec
            
            # 轨道坐标系转地心惯性坐标系 (ECI)
            # 先计算轨道平面内的位置
            x_orbital = radius * math.cos(true_anomaly)
            y_orbital = radius * math.sin(true_anomaly)
            
            # 旋转到 ECI 坐标系
            # R = Rz(-RAAN) * Rx(-i) * [x_orbital, y_orbital, 0]
            cos_raan = math.cos(raan)
            sin_raan = math.sin(raan)
            cos_inc = math.cos(inclination)
            sin_inc = math.sin(inclination)
            
            x = (cos_raan * x_orbital - sin_raan * cos_inc * y_orbital)
            y = (sin_raan * x_orbital + cos_raan * cos_inc * y_orbital)
            z = sin_inc * y_orbital
            
            positions.append([x, y, z])
    
    return torch.tensor(positions, dtype=torch.float32)


def generate_ground_stations(
    num_stations: int = 6,
    preset: str = "global",
) -> torch.Tensor:
    """
    生成地面站位置。
    
    :param num_stations: 地面站数量
    :param preset: 预设分布 ("global" 全球均匀, "starlink" 模拟真实分布)
    :return: 地面站位置 [num_stations, 3] (x, y, z in km)
    """
    if preset == "starlink":
        # 模拟 Starlink 地面站分布（北美、欧洲为主）
        lat_lon_list = [
            (47.6, -122.3),   # Seattle
            (39.0, -77.5),    # Washington DC
            (51.5, -0.1),     # London
            (48.9, 2.3),      # Paris
            (35.7, 139.7),    # Tokyo
            (-33.9, 151.2),   # Sydney
        ]
    else:
        # 全球均匀分布
        lat_lon_list = []
        for i in range(num_stations):
            lat = math.degrees(math.asin(2 * (i + 0.5) / num_stations - 1))
            lon = (i * 137.5) % 360 - 180  # 黄金角分布
            lat_lon_list.append((lat, lon))
    
    positions = []
    for lat, lon in lat_lon_list[:num_stations]:
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        x = EARTH_RADIUS_KM * math.cos(lat_rad) * math.cos(lon_rad)
        y = EARTH_RADIUS_KM * math.cos(lat_rad) * math.sin(lon_rad)
        z = EARTH_RADIUS_KM * math.sin(lat_rad)
        positions.append([x, y, z])
    
    return torch.tensor(positions, dtype=torch.float32)


# =============================================================================
# 链路可见性判断
# =============================================================================

def check_line_of_sight(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    """
    检查两点之间的连线是否被地球遮挡。
    
    :param pos1: 点1位置 [N, 3] 或 [3]
    :param pos2: 点2位置 [M, 3] 或 [3]
    :return: 可见性矩阵 [N, M] 或标量，True 表示可见
    """
    if pos1.dim() == 1:
        pos1 = pos1.unsqueeze(0)
    if pos2.dim() == 1:
        pos2 = pos2.unsqueeze(0)
    
    # 计算连线上最近地球表面的点
    # 使用参数化方法：P(t) = pos1 + t * (pos2 - pos1), t ∈ [0, 1]
    # 最近点：t* = -dot(pos1, pos2-pos1) / |pos2-pos1|²
    
    N, M = pos1.shape[0], pos2.shape[0]
    pos1_exp = pos1.unsqueeze(1).expand(N, M, 3)  # [N, M, 3]
    pos2_exp = pos2.unsqueeze(0).expand(N, M, 3)  # [N, M, 3]
    
    direction = pos2_exp - pos1_exp  # [N, M, 3]
    dir_length_sq = (direction ** 2).sum(dim=2)  # [N, M]
    
    # t* = -dot(pos1, direction) / |direction|²
    t_star = -(pos1_exp * direction).sum(dim=2) / (dir_length_sq + 1e-10)
    t_star = t_star.clamp(0, 1)  # 限制在线段上
    
    # 最近点位置
    closest_point = pos1_exp + t_star.unsqueeze(2) * direction  # [N, M, 3]
    closest_dist = torch.norm(closest_point, dim=2)  # [N, M]
    
    # 如果最近距离大于地球半径，则可见
    visible = closest_dist > EARTH_RADIUS_KM
    
    return visible.squeeze()


def check_elevation_constraint(
    sat_pos: torch.Tensor,
    ground_pos: torch.Tensor,
    min_elevation_deg: float = MIN_ELEVATION_DEG,
) -> torch.Tensor:
    """
    检查卫星相对于地面站的仰角是否满足最小约束。
    
    :param sat_pos: 卫星位置 [N, 3]
    :param ground_pos: 地面站位置 [M, 3]
    :param min_elevation_deg: 最小仰角 (度)
    :return: 满足约束的矩阵 [N, M]
    """
    if sat_pos.dim() == 1:
        sat_pos = sat_pos.unsqueeze(0)
    if ground_pos.dim() == 1:
        ground_pos = ground_pos.unsqueeze(0)
    
    N, M = sat_pos.shape[0], ground_pos.shape[0]
    sat_exp = sat_pos.unsqueeze(1).expand(N, M, 3)
    ground_exp = ground_pos.unsqueeze(0).expand(N, M, 3)
    
    # 从地面站指向卫星的向量
    to_sat = sat_exp - ground_exp  # [N, M, 3]
    
    # 地面站的法向量（指向天顶）
    ground_normal = ground_exp / torch.norm(ground_exp, dim=2, keepdim=True)
    
    # 计算仰角：elevation = 90° - arccos(dot(to_sat, normal) / |to_sat|)
    to_sat_norm = to_sat / (torch.norm(to_sat, dim=2, keepdim=True) + 1e-10)
    cos_angle = (to_sat_norm * ground_normal).sum(dim=2)
    elevation_rad = torch.asin(cos_angle.clamp(-1, 1))
    elevation_deg = torch.rad2deg(elevation_rad)
    
    return elevation_deg >= min_elevation_deg


# =============================================================================
# Grid+ ISL 拓扑模型
# =============================================================================

def compute_grid_plus_isl(
    num_planes: int,
    sats_per_plane: int,
    sat_positions: torch.Tensor,
) -> tuple[list, list]:
    """
    计算 Grid+ ISL 配置。
    
    Grid+ 模型定义：
    - 每个卫星与同一轨道平面内的前后邻居建立 2 条平面内链路 (Intra-plane ISL)
    - 每个卫星与相邻轨道面的卫星建立 2 条平面间链路 (Inter-plane ISL)
    - 每个卫星最多维持 4 条 ISL
    
    :param num_planes: 轨道面数量
    :param sats_per_plane: 每轨道卫星数
    :param sat_positions: 卫星位置 [num_sats, 3]
    :return: (edge_sources, edge_targets) 边列表
    """
    num_sats = num_planes * sats_per_plane
    edge_sources = []
    edge_targets = []
    
    def sat_index(plane, sat_in_plane):
        """计算卫星在一维数组中的索引"""
        return plane * sats_per_plane + sat_in_plane
    
    for plane in range(num_planes):
        for sat in range(sats_per_plane):
            current_idx = sat_index(plane, sat)
            
            # ===== 平面内链路 (Intra-plane ISL) =====
            # 与同一轨道内的前后卫星连接（环形）
            prev_sat = (sat - 1) % sats_per_plane
            next_sat = (sat + 1) % sats_per_plane
            
            prev_idx = sat_index(plane, prev_sat)
            next_idx = sat_index(plane, next_sat)
            
            # 平面内前向链路
            edge_sources.append(current_idx)
            edge_targets.append(next_idx)
            
            # 平面内后向链路
            edge_sources.append(current_idx)
            edge_targets.append(prev_idx)
            
            # ===== 平面间链路 (Inter-plane ISL) =====
            # 与相邻轨道面的卫星连接
            left_plane = (plane - 1) % num_planes
            right_plane = (plane + 1) % num_planes
            
            # 与左侧轨道面同位置卫星连接
            left_idx = sat_index(left_plane, sat)
            edge_sources.append(current_idx)
            edge_targets.append(left_idx)
            
            # 与右侧轨道面同位置卫星连接
            right_idx = sat_index(right_plane, sat)
            edge_sources.append(current_idx)
            edge_targets.append(right_idx)
    
    return edge_sources, edge_targets


def compute_grid_plus_edges(
    sat_positions: torch.Tensor,
    ground_positions: torch.Tensor,
    num_planes: int,
    sats_per_plane: int,
    max_isl_distance_km: float = 10000.0,  # 增加到 10000 km 以支持平面间链路
    use_visibility: bool = True,
) -> tuple:
    """
    计算 Grid+ ISL 配置的边索引，并添加星地链路。
    
    :param sat_positions: 卫星位置 [num_sats, 3]
    :param ground_positions: 地面站位置 [num_gateways, 3]
    :param num_planes: 轨道面数量
    :param sats_per_plane: 每轨道卫星数
    :param max_isl_distance_km: ISL 最大距离约束
    :param use_visibility: 是否检查视线遮挡
    :return: (edge_index, edge_delays, neighbor_indices, neighbor_delays)
    """
    num_sats = sat_positions.shape[0]
    num_ground = ground_positions.shape[0]
    positions = torch.cat([sat_positions, ground_positions], dim=0)
    num_nodes = positions.shape[0]
    
    distances = torch.cdist(positions, positions, p=2)
    delays = distances / SPEED_OF_LIGHT_KM_PER_MS
    
    # 获取 Grid+ ISL 边
    isl_sources, isl_targets = compute_grid_plus_isl(num_planes, sats_per_plane, sat_positions)
    
    # 过滤无效链路（距离过远或被地球遮挡）
    valid_sources = []
    valid_targets = []
    
    if use_visibility:
        visibility = check_line_of_sight(sat_positions, sat_positions)
    
    for src, tgt in zip(isl_sources, isl_targets):
        dist = distances[src, tgt].item()
        
        # 距离约束
        if dist > max_isl_distance_km:
            continue
        
        # 视线约束
        if use_visibility and not visibility[src, tgt]:
            continue
        
        valid_sources.append(src)
        valid_targets.append(tgt)
    
    # 添加星地链路
    if num_ground > 0:
        elevation_ok = check_elevation_constraint(sat_positions, ground_positions)
        for sat_idx in range(num_sats):
            for gnd_idx in range(num_ground):
                if elevation_ok[sat_idx, gnd_idx]:
                    node_gnd = num_sats + gnd_idx
                    # 双向链路
                    valid_sources.append(sat_idx)
                    valid_targets.append(node_gnd)
                    valid_sources.append(node_gnd)
                    valid_targets.append(sat_idx)
    
    # 构建 edge_index
    if len(valid_sources) > 0:
        edge_index = torch.tensor([valid_sources, valid_targets], dtype=torch.long)
        edge_delays = delays[edge_index[0], edge_index[1]]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_delays = torch.tensor([], dtype=torch.float32)
    
    # 构建邻居索引（用于动作空间）
    max_neighbors = 4  # Grid+ 模型每个卫星最多 4 条 ISL
    neighbor_indices = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long)
    neighbor_delays = torch.zeros((num_nodes, max_neighbors), dtype=torch.float32)
    
    # 统计每个节点的邻居
    from collections import defaultdict
    neighbors_dict = defaultdict(list)
    for src, tgt in zip(valid_sources, valid_targets):
        neighbors_dict[src].append(tgt)
    
    for node_id in range(num_nodes):
        neighbors = neighbors_dict[node_id]
        if len(neighbors) > 0:
            # 按距离排序
            neighbor_dists = [(n, distances[node_id, n].item()) for n in neighbors]
            neighbor_dists.sort(key=lambda x: x[1])
            neighbors = [n for n, _ in neighbor_dists[:max_neighbors]]
            
            neighbor_indices[node_id, :len(neighbors)] = torch.tensor(neighbors)
            neighbor_delays[node_id, :len(neighbors)] = delays[node_id, torch.tensor(neighbors)]
    
    return edge_index, edge_delays, neighbor_indices, neighbor_delays


# =============================================================================
# 拓扑构建（保留旧接口兼容性）
# =============================================================================

def _random_point_on_sphere(radius_km):
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(2 * random.random() - 1)
    x = radius_km * math.sin(phi) * math.cos(theta)
    y = radius_km * math.sin(phi) * math.sin(theta)
    z = radius_km * math.cos(phi)
    return x, y, z


def generate_satellite_positions(num_sats, altitude_km=550.0):
    """旧接口：随机生成卫星位置（已弃用，建议使用 generate_walker_constellation）"""
    radius = EARTH_RADIUS_KM + altitude_km
    positions = [_random_point_on_sphere(radius) for _ in range(num_sats)]
    return torch.tensor(positions, dtype=torch.float32)


def generate_gateway_positions(num_gateways):
    """旧接口：随机生成地面站位置（已弃用，建议使用 generate_ground_stations）"""
    positions = [_random_point_on_sphere(EARTH_RADIUS_KM) for _ in range(num_gateways)]
    return torch.tensor(positions, dtype=torch.float32)


def compute_edge_index_and_delay(positions, max_neighbors=4):
    num_nodes = positions.shape[0]
    distances = torch.cdist(positions, positions, p=2)
    delays = distances / SPEED_OF_LIGHT_KM_PER_MS

    edge_sources = []
    edge_targets = []
    for i in range(num_nodes):
        neighbors = torch.argsort(distances[i])
        selected = [int(idx) for idx in neighbors[1 : max_neighbors + 1]]
        for j in selected:
            edge_sources.append(i)
            edge_targets.append(j)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    # 添加反向边确保双向连通
    edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    edge_delays = delays[edge_index[0], edge_index[1]]

    neighbor_indices = torch.full(
        (num_nodes, max_neighbors), -1, dtype=torch.long
    )
    neighbor_delays = torch.zeros((num_nodes, max_neighbors), dtype=torch.float32)
    for node_id in range(num_nodes):
        neighbors = torch.argsort(distances[node_id])[1 : max_neighbors + 1]
        neighbor_indices[node_id, : neighbors.numel()] = neighbors
        neighbor_delays[node_id, : neighbors.numel()] = delays[node_id, neighbors]

    return edge_index, edge_delays, neighbor_indices, neighbor_delays


def compute_visible_edges(
    sat_positions: torch.Tensor,
    ground_positions: torch.Tensor,
    max_isl_distance_km: float = 5000.0,
    max_neighbors: int = 4,
) -> tuple:
    """
    计算考虑可见性约束的边索引和时延。
    
    :param sat_positions: 卫星位置 [num_sats, 3]
    :param ground_positions: 地面站位置 [num_gateways, 3]
    :param max_isl_distance_km: 星间链路最大距离 (km)
    :param max_neighbors: 每个节点的最大邻居数
    :return: (edge_index, edge_delays, neighbor_indices, neighbor_delays)
    """
    num_sats = sat_positions.shape[0]
    num_ground = ground_positions.shape[0]
    positions = torch.cat([sat_positions, ground_positions], dim=0)
    num_nodes = positions.shape[0]
    
    distances = torch.cdist(positions, positions, p=2)
    delays = distances / SPEED_OF_LIGHT_KM_PER_MS
    
    # 检查视线可见性
    visibility = check_line_of_sight(positions, positions)
    
    # 星间链路：距离约束 + 视线约束
    isl_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    isl_mask[:num_sats, :num_sats] = (distances[:num_sats, :num_sats] < max_isl_distance_km)
    isl_mask[:num_sats, :num_sats] &= visibility[:num_sats, :num_sats]
    
    # 星地链路：仰角约束
    if num_ground > 0:
        elevation_ok = check_elevation_constraint(sat_positions, ground_positions)
        isl_mask[:num_sats, num_sats:] = elevation_ok
        isl_mask[num_sats:, :num_sats] = elevation_ok.T
    
    # 自连接去除
    isl_mask.fill_diagonal_(False)
    
    # 构建边索引
    edge_sources = []
    edge_targets = []
    for i in range(num_nodes):
        valid_neighbors = torch.where(isl_mask[i])[0]
        if len(valid_neighbors) > max_neighbors:
            # 选择最近的 max_neighbors 个
            neighbor_dists = distances[i, valid_neighbors]
            topk_idx = torch.argsort(neighbor_dists)[:max_neighbors]
            valid_neighbors = valid_neighbors[topk_idx]
        for j in valid_neighbors:
            edge_sources.append(i)
            edge_targets.append(int(j))
    
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    if edge_index.numel() > 0:
        edge_delays = delays[edge_index[0], edge_index[1]]
    else:
        edge_delays = torch.tensor([], dtype=torch.float32)
    
    # 邻居索引
    neighbor_indices = torch.full((num_nodes, max_neighbors), -1, dtype=torch.long)
    neighbor_delays = torch.zeros((num_nodes, max_neighbors), dtype=torch.float32)
    for node_id in range(num_nodes):
        valid = torch.where(isl_mask[node_id])[0]
        if len(valid) > 0:
            dists = distances[node_id, valid]
            sorted_idx = torch.argsort(dists)[:max_neighbors]
            neighbors = valid[sorted_idx]
            neighbor_indices[node_id, :len(neighbors)] = neighbors
            neighbor_delays[node_id, :len(neighbors)] = delays[node_id, neighbors]
    
    return edge_index, edge_delays, neighbor_indices, neighbor_delays


def build_leo_topology(
    num_sats,
    num_gateways,
    altitude_km=550.0,
    max_neighbors=4,
):
    """旧接口：随机拓扑（兼容旧代码）"""
    sat_pos = generate_satellite_positions(num_sats, altitude_km=altitude_km)
    gate_pos = generate_gateway_positions(num_gateways)
    positions = torch.cat([sat_pos, gate_pos], dim=0)

    edge_index, edge_delays, neighbor_indices, neighbor_delays = compute_edge_index_and_delay(
        positions, max_neighbors=max_neighbors
    )
    return positions, edge_index, edge_delays, neighbor_indices, neighbor_delays


def build_walker_topology(
    num_planes: int = 22,
    sats_per_plane: int = 72,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    num_gateways: int = 6,
    max_neighbors: int = 4,
    time_sec: float = 0.0,
    use_visibility: bool = True,
    use_grid_plus: bool = True,  # 新增：使用 Grid+ ISL 配置
) -> tuple:
    """
    构建 Walker-Delta 星座拓扑。
    
    :param num_planes: 轨道面数量（Starlink 第一代为 22）
    :param sats_per_plane: 每轨道卫星数（Starlink 第一代为 72）
    :param altitude_km: 轨道高度 (km)
    :param inclination_deg: 轨道倾角 (度)
    :param num_gateways: 地面站数量
    :param max_neighbors: 每节点最大邻居数
    :param time_sec: 仿真时间 (秒)
    :param use_visibility: 是否启用可见性约束
    :param use_grid_plus: 是否使用 Grid+ ISL 配置（每卫星 4 条固定链路）
    :return: (positions, edge_index, edge_delays, neighbor_indices, neighbor_delays)
    """
    sat_pos = generate_walker_constellation(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        time_sec=time_sec,
    )
    ground_pos = generate_ground_stations(num_gateways, preset="starlink")
    positions = torch.cat([sat_pos, ground_pos], dim=0)
    
    if use_grid_plus:
        # 使用 Grid+ ISL 配置：2 条平面内 + 2 条平面间链路
        edge_index, edge_delays, neighbor_indices, neighbor_delays = compute_grid_plus_edges(
            sat_pos, ground_pos,
            num_planes=num_planes,
            sats_per_plane=sats_per_plane,
            use_visibility=use_visibility,
        )
    elif use_visibility:
        edge_index, edge_delays, neighbor_indices, neighbor_delays = compute_visible_edges(
            sat_pos, ground_pos, max_neighbors=max_neighbors
        )
    else:
        edge_index, edge_delays, neighbor_indices, neighbor_delays = compute_edge_index_and_delay(
            positions, max_neighbors=max_neighbors
        )
    
    return positions, edge_index, edge_delays, neighbor_indices, neighbor_delays
