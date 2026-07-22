"""
监督学习训练 - 行为克隆 (Behavior Cloning)

直接学习专家策略，避免 Q-learning 的传播问题
"""
import argparse
import pickle
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = pathlib.Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.model import LEO_Routing_GNN_CQL


def train_supervised(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    data_path = SRC_DIR.parent / "data" / "offline_trajectories.pkl"
    with open(data_path, "rb") as f:
        all_data = pickle.load(f)
    print(f"加载 {len(all_data)} 个样本")
    
    # 模型
    model = LEO_Routing_GNN_CQL(
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 训练
    history = {"epoch": [], "loss": [], "accuracy": []}
    best_acc = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # 随机打乱
        import random
        random.shuffle(all_data)
        
        # 批次训练
        for batch_idx in range(args.batches_per_epoch):
            start = (batch_idx * args.batch_size) % len(all_data)
            batch = all_data[start:start + args.batch_size]
            
            if len(batch) < args.batch_size:
                batch = batch + all_data[:args.batch_size - len(batch)]
            
            # 使用第一个样本的图结构
            state_x = batch[0]["state_x"].to(device)
            edge_index = batch[0]["edge_index"].to(device)
            
            curr_idx = torch.tensor([s["curr_idx"] for s in batch], device=device)
            dest_idx = torch.tensor([s["dest_idx"] for s in batch], device=device)
            actions = torch.tensor([s["action"] for s in batch], device=device)
            action_mask = torch.stack([s["action_mask"] for s in batch]).to(device)
            
            # 前向传播
            logits = model(state_x, edge_index, curr_idx, dest_idx)
            
            # 掩码无效动作
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
            # 交叉熵损失
            loss = F.cross_entropy(logits, actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算准确率
            pred = logits.argmax(dim=1)
            epoch_correct += (pred == actions).sum().item()
            epoch_total += len(batch)
        
        scheduler.step()
        
        avg_loss = epoch_loss / args.batches_per_epoch
        accuracy = epoch_correct / epoch_total
        
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "checkpoints/leo_routing_best.pt")
    
    # 保存最终模型
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/leo_routing_supervised.pt")
    
    # 绘图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history["epoch"], history["loss"], 'b-', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history["epoch"], [a*100 for a in history["accuracy"]], 'g-', linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Action Prediction Accuracy")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("checkpoints/supervised_training.png", dpi=150)
    plt.close()
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2%}")
    print(f"模型已保存至 checkpoints/leo_routing_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--node-feature-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=6)
    
    train_supervised(parser.parse_args())
