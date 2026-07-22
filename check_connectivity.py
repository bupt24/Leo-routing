"""检查拓扑连通性"""
import torch
import pickle
import heapq
from collections import defaultdict

# 加载数据
with open('data/offline_trajectories.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data[0]
edge_index = sample['edge_index']

# 构建邻接表
adj = defaultdict(list)
for s, t in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    adj[s].append(t)

# BFS 测试连通性
def can_reach(src, dst):
    visited = set()
    queue = [src]
    while queue:
        node = queue.pop(0)
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append(neighbor)
    return False

# 测试
test_pairs = [(0, 50), (10, 80), (20, 70), (30, 60), (40, 90), (0, 10), (0, 20), (0, 30)]
for src, dst in test_pairs:
    reachable = can_reach(src, dst)
    status = "OK" if reachable else "FAIL"
    print(f"{src} -> {dst}: {status}")

# 统计整体连通性
print("\n整体连通性分析:")
num_sats = 96
total_pairs = 0
reachable_pairs = 0
for i in range(num_sats):
    for j in range(num_sats):
        if i != j:
            total_pairs += 1
            if can_reach(i, j):
                reachable_pairs += 1

print(f"总对数: {total_pairs}")
print(f"可达对数: {reachable_pairs}")
print(f"连通率: {100*reachable_pairs/total_pairs:.2f}%")
