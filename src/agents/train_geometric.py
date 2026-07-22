"""
训练几何路由模型 - 监督学习
"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.geometric_model import GeometricRouter


def load_training_data(data_path):
    """加载训练数据"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"加载数据: {len(data)} 条转移")
    return data


def create_dataloader(data, batch_size=256):
    """创建数据加载器"""
    # 数据是list格式，每个元素是一个transition字典
    transitions = data
    
    # 从第一个样本获取图结构
    first_sample = transitions[0]
    x_features = first_sample['state_x'].numpy()
    neighbor_indices = first_sample['neighbor_indices'].numpy()
    action_dim = neighbor_indices.shape[1]
    
    # 提取位置 (前3维)
    positions = x_features[:, :3]
    N = positions.shape[0]
    
    # 构建样本
    samples = []
    for t in transitions:
        curr = t['curr_idx']
        action = t['action']
        dest = t['dest_idx']
        
        # 当前位置、目标位置
        curr_pos = positions[curr]
        dest_pos = positions[dest]
        
        # 邻居位置
        neighbor_idx = neighbor_indices[curr]
        neighbor_pos = np.zeros((action_dim, 3))
        action_mask = np.zeros(action_dim)
        
        for i, ni in enumerate(neighbor_idx):
            if ni >= 0 and ni < N:
                neighbor_pos[i] = positions[ni]
                action_mask[i] = 1.0
        
        # 检查 action 是否有效
        if action < action_dim and action_mask[action] == 1.0:
            samples.append({
                'curr_pos': curr_pos,
                'dest_pos': dest_pos,
                'neighbor_pos': neighbor_pos,
                'action_mask': action_mask,
                'action': action
            })
    
    print(f"有效样本: {len(samples)}")
    
    class RoutingDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            s = self.samples[idx]
            return {
                'curr_pos': torch.tensor(s['curr_pos'], dtype=torch.float32),
                'dest_pos': torch.tensor(s['dest_pos'], dtype=torch.float32),
                'neighbor_pos': torch.tensor(s['neighbor_pos'], dtype=torch.float32),
                'action_mask': torch.tensor(s['action_mask'], dtype=torch.float32),
                'action': torch.tensor(s['action'], dtype=torch.long),
            }
    
    dataset = RoutingDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, action_dim


def train(args):
    # 加载数据
    data = load_training_data(args.data_path)
    loader, action_dim = create_dataloader(data, args.batch_size)
    
    print(f"动作空间: {action_dim}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeometricRouter(hidden_dim=args.hidden_dim, action_dim=action_dim).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {'loss': [], 'acc': []}
    best_acc = 0
    
    print(f"\n开始训练 (epochs={args.epochs}, batch_size={args.batch_size})")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        batches = 0
        
        for batch_idx, batch in enumerate(loader):
            if args.batches_per_epoch and batch_idx >= args.batches_per_epoch:
                break
            
            curr_pos = batch['curr_pos'].to(device)
            dest_pos = batch['dest_pos'].to(device)
            neighbor_pos = batch['neighbor_pos'].to(device)
            action_mask = batch['action_mask'].to(device)
            action = batch['action'].to(device)
            
            # 前向传播
            scores = model(curr_pos, dest_pos, neighbor_pos, action_mask)
            loss = criterion(scores, action)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            pred = scores.argmax(dim=1)
            correct += (pred == action).sum().item()
            total += action.shape[0]
            batches += 1
        
        avg_loss = epoch_loss / batches
        acc = correct / total
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state': model.state_dict(),
                'hidden_dim': args.hidden_dim,
                'action_dim': action_dim,
            }, os.path.join(args.checkpoint_dir, 'geo_router_best.pt'))
    
    print("-" * 60)
    print(f"最佳准确率: {best_acc*100:.2f}%")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot([a * 100 for a in history['acc']])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    ax2.axhline(y=16.7, color='r', linestyle='--', label='Random (16.7%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'geo_training.png'), dpi=150)
    print(f"图表已保存")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='../data/offline_trajectories.pkl')
    parser.add_argument('--checkpoint-dir', default='../checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--batches-per-epoch', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train(args)
