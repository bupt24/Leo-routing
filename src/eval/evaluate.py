"""
评估脚本：对比 DRL 路由与 Dijkstra 基线的平均时延。
"""
import argparse
import pathlib
import sys

import torch

CURRENT_DIR = pathlib.Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.model import LEO_Routing_GNN_CQL
from env.topology import build_leo_topology
from env.queue_model import compute_total_delay
from eval.dijkstra_baseline import compute_dijkstra_delays


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = LEO_Routing_GNN_CQL(
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
    ).to(device)
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 构建拓扑
    positions, edge_index, edge_delays, neighbor_indices, neighbor_delays = build_leo_topology(
        num_sats=args.num_sats,
        num_gateways=args.num_gateways,
        altitude_km=args.altitude_km,
        max_neighbors=args.max_neighbors,
    )
    positions = positions.to(device)
    edge_index = edge_index.to(device)
    edge_delays = edge_delays.to(device)
    neighbor_indices = neighbor_indices.to(device)
    neighbor_delays = neighbor_delays.to(device)

    num_nodes = positions.shape[0]
    queue_lengths = torch.rand(num_nodes, device=device) * 5.0

    # 随机生成测试样本
    num_samples = args.num_samples
    sources = torch.randint(0, num_nodes, (num_samples,), device=device)
    targets = torch.randint(0, num_nodes, (num_samples,), device=device)

    # 构建状态特征
    base_features = torch.randn(num_nodes, args.node_feature_dim - 2, device=device)
    mean_delay = edge_delays.mean().unsqueeze(0).repeat(num_nodes, 1)
    queue_feat = queue_lengths.unsqueeze(1)
    state_x = torch.cat([base_features, mean_delay, queue_feat], dim=1)

    # DRL 路由时延
    drl_delays = []
    with torch.no_grad():
        for i in range(num_samples):
            curr = sources[i : i + 1]
            dest = targets[i : i + 1]
            q_values = model(state_x, edge_index, curr, dest)
            
            # 应用动作掩码
            neighbors = neighbor_indices[int(curr)]
            valid_count = (neighbors >= 0).sum().item()
            if valid_count > 0:
                action = q_values[0, :valid_count].argmax().item()
                prop_delay = neighbor_delays[int(curr), action]
                q_len = queue_lengths[int(curr)]
                total_delay = compute_total_delay(prop_delay, q_len)
            else:
                total_delay = torch.tensor(float("inf"))
            drl_delays.append(total_delay.item())

    drl_avg = sum(drl_delays) / len(drl_delays)

    # Dijkstra 基线时延
    dijkstra_delays = compute_dijkstra_delays(
        num_nodes=num_nodes,
        edge_index=edge_index.cpu(),
        edge_weights=edge_delays.cpu(),
        sources=sources.cpu(),
        targets=targets.cpu(),
    )
    dijkstra_avg = dijkstra_delays.mean().item()

    print("=" * 50)
    print(f"评估样本数: {num_samples}")
    print(f"DRL 路由平均时延 (单跳): {drl_avg:.4f} ms")
    print(f"Dijkstra 端到端时延: {dijkstra_avg:.4f} ms")
    print("=" * 50)

    return drl_avg, dijkstra_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LEO routing")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-sats", type=int, default=120)
    parser.add_argument("--num-gateways", type=int, default=6)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--node-feature-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)

    evaluate(parser.parse_args())
