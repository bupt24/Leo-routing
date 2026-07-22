"""
简化的路由模型 - 基于几何信息

直接用位置差向量预测下一跳，不依赖复杂的 GNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricRouter(nn.Module):
    """
    基于几何信息的路由器
    
    输入：当前位置、目标位置、邻居位置
    输出：每个邻居的得分（选择最高分的邻居）
    """
    
    def __init__(self, hidden_dim=64, action_dim=6):
        super().__init__()
        self.action_dim = action_dim
        
        # 方向编码器
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 邻居评分器：比较每个邻居方向与目标方向的一致性
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, curr_pos, dest_pos, neighbor_pos, action_mask):
        """
        Args:
            curr_pos: [B, 3] 当前节点位置
            dest_pos: [B, 3] 目标节点位置
            neighbor_pos: [B, action_dim, 3] 邻居节点位置
            action_mask: [B, action_dim] 有效邻居掩码
            
        Returns:
            scores: [B, action_dim] 每个邻居的得分
        """
        B = curr_pos.shape[0]
        
        # 计算到目标的方向
        to_dest = dest_pos - curr_pos  # [B, 3]
        dest_feat = self.direction_encoder(to_dest)  # [B, hidden]
        
        # 计算到每个邻居的方向
        to_neighbors = neighbor_pos - curr_pos.unsqueeze(1)  # [B, action_dim, 3]
        neighbor_feat = self.direction_encoder(to_neighbors.view(-1, 3))  # [B*action_dim, hidden]
        neighbor_feat = neighbor_feat.view(B, self.action_dim, -1)  # [B, action_dim, hidden]
        
        # 拼接目标方向特征和邻居方向特征
        dest_feat_exp = dest_feat.unsqueeze(1).expand(-1, self.action_dim, -1)  # [B, action_dim, hidden]
        combined = torch.cat([dest_feat_exp, neighbor_feat], dim=-1)  # [B, action_dim, hidden*2]
        
        # 计算每个邻居的得分
        scores = self.scorer(combined.view(-1, combined.shape[-1]))  # [B*action_dim, 1]
        scores = scores.view(B, self.action_dim)  # [B, action_dim]
        
        # 掩码无效邻居
        scores = scores.masked_fill(action_mask == 0, -1e9)
        
        return scores


class PositionAwareRouter(nn.Module):
    """
    位置感知路由器 - 结合 GNN 和几何信息
    """
    
    def __init__(self, node_feature_dim=10, hidden_dim=128, action_dim=6):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 几何路由器
        self.geo_router = GeometricRouter(hidden_dim=hidden_dim // 2, action_dim=action_dim)
        
        # 额外的节点特征编码（如拥塞信息）
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim - 3, hidden_dim // 2),  # 除位置外的特征
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(1 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x_features, neighbor_indices, curr_idx, dest_idx, action_mask):
        """
        Args:
            x_features: [N, feat_dim] 节点特征
            neighbor_indices: [N, action_dim] 邻居索引
            curr_idx: [B] 当前节点索引
            dest_idx: [B] 目标节点索引
            action_mask: [B, action_dim] 有效邻居掩码
        """
        B = curr_idx.shape[0]
        
        # 提取位置 (前3维)
        positions = x_features[:, :3]
        curr_pos = positions[curr_idx]  # [B, 3]
        dest_pos = positions[dest_idx]  # [B, 3]
        
        # 获取邻居位置
        neighbor_idx = neighbor_indices[curr_idx]  # [B, action_dim]
        # 处理无效邻居（用当前节点位置填充）
        valid_neighbor_idx = neighbor_idx.clamp(min=0)
        neighbor_pos = positions[valid_neighbor_idx]  # [B, action_dim, 3]
        
        # 几何得分
        geo_scores = self.geo_router(curr_pos, dest_pos, neighbor_pos, action_mask)  # [B, action_dim]
        
        # 节点特征（拥塞等）
        other_features = x_features[:, 3:]  # 除位置外的特征
        curr_features = other_features[curr_idx]  # [B, feat-3]
        node_feat = self.node_encoder(curr_features)  # [B, hidden//4]
        
        # 融合
        node_feat_exp = node_feat.unsqueeze(1).expand(-1, self.action_dim, -1)  # [B, action_dim, hidden//4]
        geo_scores_exp = geo_scores.unsqueeze(-1)  # [B, action_dim, 1]
        combined = torch.cat([geo_scores_exp, node_feat_exp], dim=-1)  # [B, action_dim, 1+hidden//4]
        
        final_scores = self.fusion(combined.view(-1, combined.shape[-1])).view(B, self.action_dim)
        final_scores = final_scores.masked_fill(action_mask == 0, -1e9)
        
        return final_scores
