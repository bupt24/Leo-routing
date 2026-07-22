"""
分析：专家选择的邻居在时延上的排名

Dijkstra 基于时延选择，所以应该看时延排名而不是距离排名
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
neighbor_indices = first['neighbor_indices'].numpy()  # [N, 6]
neighbor_delays = first['neighbor_delays'].numpy()   # [N, 6] 到邻居的时延
N, action_dim = neighbor_indices.shape

print(f"节点数: {N}, 动作数: {action_dim}")

# 分析专家选择：是否选择最小时延的邻居？
rank_dist = Counter()
total = 0

for t in data:
    curr = int(t['curr_idx'])
    action = int(t['action'])
    
    # 当前节点的邻居时延
    delays = neighbor_delays[curr].copy()
    valid_mask = neighbor_indices[curr] >= 0
    
    # 只看有效邻居
    valid_delays = [(i, delays[i]) for i in range(action_dim) if valid_mask[i]]
    if not valid_delays:
        continue
    
    # 按时延排序
    sorted_by_delay = sorted(valid_delays, key=lambda x: x[1])
    
    # 找到选择的动作在时延排序中的排名
    for rank, (action_idx, _) in enumerate(sorted_by_delay):
        if action_idx == action:
            rank_dist[rank] += 1
            break
    
    total += 1

print(f"\n专家选择（按时延排名）:")
for k in range(action_dim):
    pct = rank_dist[k] / total * 100
    print(f"  时延第{k+1}小: {rank_dist[k]:6d} ({pct:.1f}%)")

# 检查：专家是否总是选择最小时延？
min_delay_correct = rank_dist[0] / total * 100
print(f"\n选择最小时延邻居的比例: {min_delay_correct:.1f}%")
