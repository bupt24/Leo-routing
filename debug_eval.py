"""调试评估函数"""
import torch
import pickle
import sys
sys.path.insert(0, 'src')
from agents.model import LEO_Routing_GNN_CQL

device = 'cpu'

# 加载训练数据
with open('data/offline_trajectories.pkl', 'rb') as f:
    train_data = pickle.load(f)

sample0 = train_data[0]
state_x = sample0['state_x'].to(device)
edge_index = sample0['edge_index'].to(device)
neighbor_indices = sample0['neighbor_indices'].to(device)  # 使用保存的邻居索引
neighbor_delays = sample0['neighbor_delays'].to(device)
num_nodes = state_x.shape[0]
action_dim = 6

# 加载模型
model = LEO_Routing_GNN_CQL(node_feature_dim=10, hidden_dim=128, action_dim=6).to(device)
model.load_state_dict(torch.load('checkpoints/leo_routing_best.pt', map_location=device, weights_only=True))
model.eval()

# 测试多个路径
test_pairs = [(0, 50), (10, 80), (20, 70), (30, 60), (40, 90)]
max_hops = 15

for src, dst in test_pairs:
    current = src
    visited = {src}
    path = [src]
    success = False
    
    with torch.no_grad():
        for hop in range(max_hops):
            if current == dst:
                success = True
                break
            
            valid_count = (neighbor_indices[current] >= 0).sum().item()
            if valid_count == 0:
                break
            
            action_mask = torch.zeros(1, 6, device=device)
            action_mask[0, :valid_count] = 1.0
            
            q_values = model(state_x, edge_index, torch.tensor([current]), torch.tensor([dst]))
            masked_q = q_values.masked_fill(action_mask == 0, -1e9)
            action = masked_q.argmax(dim=1).item()
            
            next_node = neighbor_indices[current, action].item()
            
            if next_node in visited:
                # 尝试其他动作
                found = False
                for alt in range(valid_count):
                    alt_next = neighbor_indices[current, alt].item()
                    if alt_next >= 0 and alt_next not in visited:
                        next_node = alt_next
                        found = True
                        break
                if not found:
                    break
            
            visited.add(next_node)
            path.append(next_node)
            current = next_node
    
    status = "成功" if success else "失败"
    print(f"{src} -> {dst}: {status}, 路径长度={len(path)}, 路径={path[:5]}...")

# 检查 Q 值分布
print("\nQ值分析:")
with torch.no_grad():
    for src in [0, 10, 20]:
        for dst in [50, 80]:
            q = model(state_x, edge_index, torch.tensor([src]), torch.tensor([dst]))
            print(f"  src={src}, dst={dst}: Q={q[0].tolist()[:6]}")
