import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class LEO_Routing_GNN_CQL(nn.Module):
    """
    LEO Satellite Routing with GraphSAGE and Offline RL (CQL)
    
    Features:
    - GraphSAGE for topology encoding
    - Direction encoder for target-aware routing
    - Q-network for action selection
    """
    
    def __init__(self, node_feature_dim, hidden_dim, action_dim, dropout: float = 0.0):
        super(LEO_Routing_GNN_CQL, self).__init__()
        
        # 1. GraphSAGE topology encoder
        self.sage_layer1 = SAGEConv(node_feature_dim, hidden_dim)
        self.sage_layer2 = SAGEConv(hidden_dim, hidden_dim)
        
        # 2. Direction encoder: encode target direction
        self.direction_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        
        # 3. Q-Network MLP
        # Input: GNN embeddings (curr + dest) + direction features
        q_input_dim = hidden_dim * 2 + hidden_dim // 2
        self.q_value_net = nn.Sequential(
            nn.Linear(q_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x_features, dynamic_edge_index, current_node_idx, dest_node_idx):
        """
        Forward pass
        
        Args:
            x_features: Node features [N, feat_dim], first 3 dims are normalized position
            dynamic_edge_index: Edge connectivity [2, E]
            current_node_idx: Current node indices [B]
            dest_node_idx: Destination node indices [B]
            
        Returns:
            q_values: Q-values for each action [B, action_dim]
        """
        # GNN message passing
        h = self.sage_layer1(x_features, dynamic_edge_index)
        h = F.relu(h)
        h = self.sage_layer2(h, dynamic_edge_index)
        node_embeddings = F.relu(h)
        
        # Extract node embeddings
        curr_emb = node_embeddings[current_node_idx]
        dest_emb = node_embeddings[dest_node_idx]
        
        # Compute direction vector (using normalized position in first 3 dims)
        curr_pos = x_features[current_node_idx, 0:3]
        dest_pos = x_features[dest_node_idx, 0:3]
        direction = dest_pos - curr_pos  # Direction towards target
        
        # Encode direction
        direction_feat = self.direction_encoder(direction)
        
        # Fuse features
        state_representation = torch.cat([curr_emb, dest_emb, direction_feat], dim=-1)
        
        # Output Q-values
        q_values = self.q_value_net(state_representation)
        return q_values
