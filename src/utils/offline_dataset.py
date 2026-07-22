import torch

from env.topology import build_leo_topology, build_walker_topology
from env.queue_model import compute_total_delay


class SyntheticOfflineDataset:
    def __init__(
        self,
        num_nodes,
        node_feature_dim,
        action_dim,
        batch_size,
        num_edges,
        device,
        total_batches,
    ):
        self.num_nodes = num_nodes
        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.num_edges = num_edges
        self.device = device
        self.total_batches = total_batches

    def __iter__(self):
        for _ in range(self.total_batches):
            state_x = torch.randn(
                self.num_nodes, self.node_feature_dim, device=self.device
            )
            next_x = torch.randn(
                self.num_nodes, self.node_feature_dim, device=self.device
            )

            edge_index = torch.randint(
                0, self.num_nodes, (2, self.num_edges), device=self.device
            )
            next_edge_index = torch.randint(
                0, self.num_nodes, (2, self.num_edges), device=self.device
            )

            curr_idx = torch.randint(
                0, self.num_nodes, (self.batch_size,), device=self.device
            )
            dest_idx = torch.randint(
                0, self.num_nodes, (self.batch_size,), device=self.device
            )
            next_curr_idx = torch.randint(
                0, self.num_nodes, (self.batch_size,), device=self.device
            )
            next_dest_idx = torch.randint(
                0, self.num_nodes, (self.batch_size,), device=self.device
            )

            actions = torch.randint(
                0, self.action_dim, (self.batch_size,), device=self.device
            )
            rewards = torch.randn(self.batch_size, device=self.device)
            dones = torch.randint(0, 2, (self.batch_size,), device=self.device).float()

            yield {
                "state": (state_x, edge_index, curr_idx, dest_idx),
                "next_state": (next_x, next_edge_index, next_curr_idx, next_dest_idx),
                "action": actions,
                "reward": rewards,
                "done": dones,
            }


class LEOOfflineDataset:
    def __init__(
        self,
        num_sats,
        num_gateways,
        node_feature_dim,
        action_dim,
        batch_size,
        max_neighbors,
        device,
        total_batches,
        altitude_km=550.0,
    ):
        self.num_sats = num_sats
        self.num_gateways = num_gateways
        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_neighbors = max_neighbors
        self.device = device
        self.total_batches = total_batches
        self.altitude_km = altitude_km

    def __iter__(self):
        for _ in range(self.total_batches):
            positions, edge_index, edge_delays, neighbor_indices, neighbor_delays = build_leo_topology(
                num_sats=self.num_sats,
                num_gateways=self.num_gateways,
                altitude_km=self.altitude_km,
                max_neighbors=self.max_neighbors,
            )
            positions = positions.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_delays = edge_delays.to(self.device)
            neighbor_indices = neighbor_indices.to(self.device)
            neighbor_delays = neighbor_delays.to(self.device)

            num_nodes = positions.shape[0]
            # 随机生成队列长度模拟拥塞
            queue_lengths = torch.rand(num_nodes, device=self.device) * 10.0
            
            base_features = torch.randn(
                num_nodes, self.node_feature_dim - 2, device=self.device
            )
            mean_delay = edge_delays.mean().unsqueeze(0).repeat(num_nodes, 1)
            queue_feat = queue_lengths.unsqueeze(1)
            state_x = torch.cat([base_features, mean_delay, queue_feat], dim=1)
            next_queue = queue_lengths + torch.randn_like(queue_lengths) * 0.5
            next_queue = next_queue.clamp(min=0.0)
            next_x = torch.cat([
                base_features + 0.01 * torch.randn_like(base_features),
                mean_delay,
                next_queue.unsqueeze(1),
            ], dim=1)

            curr_idx = torch.randint(
                0, num_nodes, (self.batch_size,), device=self.device
            )
            dest_idx = torch.randint(
                0, num_nodes, (self.batch_size,), device=self.device
            )
            next_curr_idx = curr_idx.clone()
            next_dest_idx = dest_idx.clone()

            action_mask = torch.zeros(
                self.batch_size, self.action_dim, device=self.device
            )
            next_action_mask = torch.zeros(
                self.batch_size, self.action_dim, device=self.device
            )
            valid_counts = torch.zeros(
                self.batch_size, device=self.device, dtype=torch.long
            )
            next_valid_counts = torch.zeros(
                self.batch_size, device=self.device, dtype=torch.long
            )
            for i, node_id in enumerate(curr_idx.tolist()):
                neighbors = neighbor_indices[node_id]
                valid = (neighbors >= 0).nonzero(as_tuple=False).squeeze(-1)
                count = min(valid.numel(), self.action_dim)
                valid_counts[i] = count
                if count > 0:
                    action_mask[i, :count] = 1.0

            for i, node_id in enumerate(next_curr_idx.tolist()):
                neighbors = neighbor_indices[node_id]
                valid = (neighbors >= 0).nonzero(as_tuple=False).squeeze(-1)
                count = min(valid.numel(), self.action_dim)
                next_valid_counts[i] = count
                if count > 0:
                    next_action_mask[i, :count] = 1.0

            actions = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            for i in range(self.batch_size):
                if valid_counts[i] > 0:
                    actions[i] = torch.randint(0, valid_counts[i], (1,), device=self.device)

            chosen_delays = torch.zeros(self.batch_size, device=self.device)
            for i, node_id in enumerate(curr_idx.tolist()):
                if valid_counts[i] > 0:
                    prop_delay = neighbor_delays[node_id, actions[i]]
                    q_len = queue_lengths[node_id]
                    chosen_delays[i] = compute_total_delay(prop_delay, q_len)
                else:
                    chosen_delays[i] = neighbor_delays.mean() + queue_lengths.mean()

            rewards = -chosen_delays
            dones = torch.zeros(self.batch_size, device=self.device)

            yield {
                "state": (state_x, edge_index, curr_idx, dest_idx),
                "next_state": (next_x, edge_index, next_curr_idx, next_dest_idx),
                "action": actions,
                "action_mask": action_mask,
                "next_action_mask": next_action_mask,
                "reward": rewards,
                "done": dones,
            }


class WalkerOfflineDataset:
    """
    基于 Walker-Delta 星座模型的离线数据集。
    支持时间演化和可见性约束。
    """
    def __init__(
        self,
        num_planes: int = 22,
        sats_per_plane: int = 72,
        num_gateways: int = 6,
        node_feature_dim: int = 10,
        action_dim: int = 4,
        batch_size: int = 128,
        max_neighbors: int = 4,
        device: str = "cpu",
        total_batches: int = 10,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        time_step_sec: float = 10.0,
        use_visibility: bool = True,
    ):
        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.num_gateways = num_gateways
        self.node_feature_dim = node_feature_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_neighbors = max_neighbors
        self.device = device
        self.total_batches = total_batches
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.time_step_sec = time_step_sec
        self.use_visibility = use_visibility
        self.current_time = 0.0

    def __iter__(self):
        for batch_idx in range(self.total_batches):
            # 当前时间的拓扑
            positions, edge_index, edge_delays, neighbor_indices, neighbor_delays = build_walker_topology(
                num_planes=self.num_planes,
                sats_per_plane=self.sats_per_plane,
                altitude_km=self.altitude_km,
                inclination_deg=self.inclination_deg,
                num_gateways=self.num_gateways,
                max_neighbors=self.max_neighbors,
                time_sec=self.current_time,
                use_visibility=self.use_visibility,
            )
            
            # 下一时刻的拓扑
            next_time = self.current_time + self.time_step_sec
            next_positions, next_edge_index, next_edge_delays, next_neighbor_indices, next_neighbor_delays = build_walker_topology(
                num_planes=self.num_planes,
                sats_per_plane=self.sats_per_plane,
                altitude_km=self.altitude_km,
                inclination_deg=self.inclination_deg,
                num_gateways=self.num_gateways,
                max_neighbors=self.max_neighbors,
                time_sec=next_time,
                use_visibility=self.use_visibility,
            )
            
            positions = positions.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_delays = edge_delays.to(self.device) if edge_delays.numel() > 0 else torch.tensor([1.0], device=self.device)
            neighbor_indices = neighbor_indices.to(self.device)
            neighbor_delays = neighbor_delays.to(self.device)
            
            next_edge_index = next_edge_index.to(self.device)
            next_neighbor_indices = next_neighbor_indices.to(self.device)
            
            num_nodes = positions.shape[0]
            
            # 队列长度模拟
            queue_lengths = torch.rand(num_nodes, device=self.device) * 10.0
            next_queue = (queue_lengths + torch.randn_like(queue_lengths) * 0.5).clamp(min=0.0)
            
            # 构建状态特征
            base_features = torch.randn(num_nodes, self.node_feature_dim - 2, device=self.device)
            mean_delay = edge_delays.mean().unsqueeze(0).repeat(num_nodes, 1)
            queue_feat = queue_lengths.unsqueeze(1)
            state_x = torch.cat([base_features, mean_delay, queue_feat], dim=1)
            next_x = torch.cat([
                base_features + 0.01 * torch.randn_like(base_features),
                mean_delay,
                next_queue.unsqueeze(1),
            ], dim=1)
            
            # 采样当前节点和目的节点
            curr_idx = torch.randint(0, num_nodes, (self.batch_size,), device=self.device)
            dest_idx = torch.randint(0, num_nodes, (self.batch_size,), device=self.device)
            
            # 动作掩码
            action_mask = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            next_action_mask = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            valid_counts = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            
            for i, node_id in enumerate(curr_idx.tolist()):
                neighbors = neighbor_indices[node_id]
                valid = (neighbors >= 0).nonzero(as_tuple=False).squeeze(-1)
                count = min(valid.numel(), self.action_dim)
                valid_counts[i] = count
                if count > 0:
                    action_mask[i, :count] = 1.0
            
            # 采样动作
            actions = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            next_curr_idx = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
            
            for i in range(self.batch_size):
                if valid_counts[i] > 0:
                    actions[i] = torch.randint(0, int(valid_counts[i]), (1,), device=self.device)
                    # 下一状态的当前节点 = 选择的邻居
                    next_curr_idx[i] = neighbor_indices[curr_idx[i], actions[i]]
                else:
                    next_curr_idx[i] = curr_idx[i]
            
            next_dest_idx = dest_idx.clone()
            
            # 下一状态的动作掩码
            for i, node_id in enumerate(next_curr_idx.tolist()):
                neighbors = next_neighbor_indices[node_id]
                valid = (neighbors >= 0).nonzero(as_tuple=False).squeeze(-1)
                count = min(valid.numel(), self.action_dim)
                if count > 0:
                    next_action_mask[i, :count] = 1.0
            
            # 计算奖励
            chosen_delays = torch.zeros(self.batch_size, device=self.device)
            dones = torch.zeros(self.batch_size, device=self.device)
            
            for i, node_id in enumerate(curr_idx.tolist()):
                if valid_counts[i] > 0:
                    prop_delay = neighbor_delays[node_id, actions[i]]
                    q_len = queue_lengths[node_id]
                    chosen_delays[i] = compute_total_delay(prop_delay, q_len)
                else:
                    chosen_delays[i] = 100.0  # 惩罚无法移动
                
                # 检查是否到达目的地
                if next_curr_idx[i] == dest_idx[i]:
                    dones[i] = 1.0
            
            rewards = -chosen_delays
            # 到达目的地给予正向奖励
            rewards = rewards + dones * 50.0
            
            # 更新时间
            self.current_time = next_time
            
            yield {
                "state": (state_x, edge_index, curr_idx, dest_idx),
                "next_state": (next_x, next_edge_index, next_curr_idx, next_dest_idx),
                "action": actions,
                "action_mask": action_mask,
                "next_action_mask": next_action_mask,
                "reward": rewards,
                "done": dones,
            }
