"""
分析训练数据，理解专家策略的模式
"""
import pickle
import numpy as np
from collections import Counter

# 加载数据
with open('d:/lunwen/data/offline_trajectories.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"总样本数: {len(data)}")

# 从第一个样本获取拓扑
first = data[0]
positions = first['state_x'][:, :3].numpy()  # [N, 3]
neighbor_indices = first['neighbor_indices'].numpy()  # [N, 6]
N = positions.shape[0]
action_dim = neighbor_indices.shape[1]

print(f"节点数: {N}, 动作数: {action_dim}")

# 分析专家选择的模式
correct_greedy = 0
total = 0
action_dist = Counter()

for t in data:
    curr = t['curr_idx']
    dest = t['dest_idx']
    action = t['action']
    
    # 计算当前到目标的距离
    curr_pos = positions[curr]
    dest_pos = positions[dest]
    
    # 计算每个邻居到目标的距离
    neighbors = neighbor_indices[curr]
    neighbor_dists = []
    
    for i, ni in enumerate(neighbors):
        if ni >= 0 and ni < N:
            ni_pos = positions[ni]
            dist_to_dest = np.linalg.norm(ni_pos - dest_pos)
            neighbor_dists.append((i, dist_to_dest, ni))
        else:
            neighbor_dists.append((i, float('inf'), -1))
    
    # 贪婪选择：选最近目标的邻居
    neighbor_dists.sort(key=lambda x: x[1])
    greedy_action = neighbor_dists[0][0]
    
    if action == greedy_action:
        correct_greedy += 1
    
    action_dist[action] += 1
    total += 1

print(f"\n贪婪策略匹配率: {correct_greedy/total*100:.2f}%")
print(f"\n动作分布: {dict(action_dist)}")

# 分析专家选择第几近的邻居
rank_dist = Counter()
for t in data:
    curr = t['curr_idx']
    dest = t['dest_idx']
    action = t['action']
    
    neighbors = neighbor_indices[curr]
    neighbor_dists = []
    
    for i, ni in enumerate(neighbors):
        if ni >= 0 and ni < N:
            ni_pos = positions[ni]
            dist_to_dest = np.linalg.norm(ni_pos - dest_pos)
            neighbor_dists.append((i, dist_to_dest))
        else:
            neighbor_dists.append((i, float('inf')))
    
    # 排序
    sorted_neighbors = sorted(enumerate(neighbor_dists), key=lambda x: x[1][1])
    
    # 找到action在排序中的位置
    for rank, (orig_idx, (action_idx, _)) in enumerate(sorted_neighbors):
        if action_idx == action:
            rank_dist[rank] += 1
            break

print(f"\n专家选择第k近邻居的分布:")
for k in range(action_dim):
    pct = rank_dist[k] / total * 100
    print(f"  第{k+1}近: {rank_dist[k]:6d} ({pct:.1f}%)")
