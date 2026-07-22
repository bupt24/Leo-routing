"""
Dijkstra 最短路径基线算法，用于与 DRL 路由对比。
"""
import heapq
from typing import Dict, List, Tuple

import torch


def dijkstra_shortest_path(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    source: int,
    target: int,
) -> Tuple[float, List[int]]:
    """
    Dijkstra 算法求解单源最短路径。
    
    :param num_nodes: 节点总数
    :param edge_index: 边索引 [2, num_edges]
    :param edge_weights: 边权重（时延）[num_edges]
    :param source: 起点
    :param target: 终点
    :return: (最短时延, 路径节点列表)
    """
    # 构建邻接表
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for idx in range(edge_index.shape[1]):
        u = int(edge_index[0, idx])
        v = int(edge_index[1, idx])
        w = float(edge_weights[idx])
        adj[u].append((v, w))
    
    # Dijkstra
    dist = [float("inf")] * num_nodes
    prev = [-1] * num_nodes
    dist[source] = 0.0
    pq = [(0.0, source)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            break
        for v, w in adj[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
    
    # 回溯路径
    path = []
    node = target
    while node != -1:
        path.append(node)
        node = prev[node]
    path.reverse()
    
    if dist[target] == float("inf"):
        return float("inf"), []
    return dist[target], path


def compute_dijkstra_delays(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    sources: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    批量计算 Dijkstra 最短时延。
    
    :param sources: 起点索引 [batch_size]
    :param targets: 终点索引 [batch_size]
    :return: 最短时延 [batch_size]
    """
    batch_size = sources.shape[0]
    delays = torch.zeros(batch_size)
    
    for i in range(batch_size):
        delay, _ = dijkstra_shortest_path(
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            source=int(sources[i]),
            target=int(targets[i]),
        )
        delays[i] = delay
    
    return delays
