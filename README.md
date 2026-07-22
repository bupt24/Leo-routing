# LEO卫星网络离线强化学习路由

基于 **Conservative Q-Learning (CQL)** 和 **GraphSAGE** 的低轨卫星网络最小时延路由方案。支持 Walker-Delta 星座模型与可见性约束。

## 📁 项目结构

```
D:\lunwen\
├── requirements.txt          # Python 依赖
├── checkpoints/              # 模型权重与训练曲线
├── data/                     # 数据文件（如有）
└── src/
    ├── agents/
    │   ├── model.py          # GraphSAGE + Q网络 (2层SAGEConv)
    │   ├── cql_trainer.py    # CQL损失计算与目标网络软更新
    │   └── train_offline.py  # 训练入口（支持曲线绘制）
    ├── env/
    │   ├── topology.py       # Walker-Delta 星座拓扑生成 + 可见性约束
    │   └── queue_model.py    # M/M/1 排队时延模型
    ├── eval/
    │   ├── dijkstra_baseline.py  # Dijkstra 最短路径基线
    │   └── evaluate.py       # DRL vs Dijkstra 对比评估
    └── utils/
        └── offline_dataset.py    # 离线数据集生成器 (Synthetic/LEO/Walker)
```

## 🚀 快速开始

### 1. 创建虚拟环境 (首次运行)

```powershell
cd D:\lunwen
python -m venv venv
& D:\lunwen\venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
& D:\lunwen\venv\Scripts\python.exe -m pip install -r D:\lunwen\requirements.txt
```

> 需要：`torch`、`torch-geometric`、`matplotlib`

### 3. 训练模型

#### 方式一：基础 LEO 拓扑训练

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
    --epochs 50 `
    --batches-per-epoch 20 `
    --dataset leo `
    --num-sats 120 `
    --num-gateways 6
```

#### 方式二：Walker-Delta 星座训练（推荐）

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
    --epochs 50 `
    --batches-per-epoch 20 `
    --dataset walker `
    --num-planes 6 `
    --sats-per-plane 10 `
    --inclination-deg 53.0 `
    --use-visibility
```

### 4. 训练输出

训练完成后会在 `checkpoints/` 目录生成：

| 文件 | 说明 |
|------|------|
| `leo_routing_epoch50.pt` | 模型权重 |
| `training_history_epoch50.json` | 训练历史数据 (JSON) |
| `training_curves_epoch50.png` | Loss/Reward 曲线图 |

### 5. 评估对比

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\eval\evaluate.py `
    --checkpoint D:\lunwen\checkpoints\leo_routing_epoch50.pt `
    --num-samples 100
```

### 6. 查看训练曲线

```powershell
Start-Process D:\lunwen\checkpoints\training_curves_epoch50.png
```

## ⚙️ 完整参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 3 |
| `--batches-per-epoch` | 每轮批次数 | 10 |
| `--batch-size` | 批次大小 | 128 |
| `--dataset` | 数据集类型 (`synthetic`/`leo`/`walker`) | synthetic |
| `--num-gateways` | 地面站数量 | 6 |
| `--max-neighbors` | 最大邻居数（动作空间维度） | 4 |
| `--altitude-km` | 轨道高度 (km) | 550.0 |

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--node-feature-dim` | 节点特征维度 | 10 |
| `--hidden-dim` | 隐藏层维度 | 64 |
| `--action-dim` | 动作空间维度 | 4 |
| `--lr` | 学习率 | 1e-3 |
| `--gamma` | 折扣因子 | 0.99 |
| `--cql-alpha` | CQL 正则化权重 | 1.0 |
| `--tau` | 目标网络软更新系数 | 0.005 |
| `--grad-clip` | 梯度裁剪阈值 | 1.0 |

### LEO 数据集参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-sats` | 卫星数量 | 120 |

### Walker 星座参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-planes` | 轨道面数量 | 6 |
| `--sats-per-plane` | 每轨道卫星数 | 10 |
| `--inclination-deg` | 轨道倾角 (度) | 53.0 |
| `--use-visibility` | 启用可见性约束 (LOS + 仰角) | False |

## 🔬 技术细节

### 网络架构

- **图编码器**: 2层 GraphSAGE (SAGEConv)，捕获二跳邻域拥塞信息
- **Q网络**: MLP (hidden_dim → hidden_dim → action_dim)
- **离线算法**: Conservative Q-Learning (CQL)，通过 logsumexp 正则化防止 Q 值过估计

### Walker-Delta 星座模型

- 支持轨道力学计算（开普勒轨道）
- 星间链路 (ISL) 可见性约束：
  - 视线遮挡检测（地球遮挡）
  - 最小仰角约束（默认 25°）
- 时间演化拓扑

### 时延模型

- **传播时延**: 基于欧氏距离
- **排队时延**: M/M/1 模型，考虑链路利用率

## 📊 示例输出

```
Epoch 1 | Loss: 212.4336 | Avg Reward: -13.5543
Epoch 2 | Loss: 227.5920 | Avg Reward: -13.8444
...
Epoch 50 | Loss: 185.2102 | Avg Reward: -10.2315
模型已保存至: checkpoints/leo_routing_epoch50.pt
训练曲线图已保存至: checkpoints/training_curves_epoch50.png
```

## 📝 下一步建议

- [ ] 用真实仿真轨迹数据替换随机动作采样
- [ ] 实现多跳路由仿真评估
- [ ] 添加多算法对比曲线（GraphPR/MAFDR/POMAP）
- [ ] 引入图采样（邻居采样）以扩展到更大星座
- [ ] 添加星座拓扑可视化

## 📚 参考文献

- [CQL: Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
# LEO 卫星网络离线路由学习

基于 GraphSAGE + Conservative Q-Learning (CQL) 的低轨卫星路由实验工程，支持多种离线数据源：
- synthetic 随机图数据
- leo 随机卫星/地面站拓扑
- walker Walker-Delta 星座拓扑（可见性约束）
- real 真实 Starlink TLE 数据
- precomputed 预计算专家轨迹数据

## 项目结构

```text
D:\lunwen
├── README.md
├── requirements.txt
├── data/
│   ├── offline_trajectories.pkl
│   └── tle/
├── checkpoints/
└── src/
    ├── agents/
    │   ├── model.py
    │   ├── cql_trainer.py
    │   ├── train_offline.py
    │   ├── train_supervised.py
    │   ├── train_geometric.py
    │   └── evaluate.py
    ├── env/
    │   ├── topology.py
    │   └── queue_model.py
    ├── data/
    │   ├── download_tle.py
    │   ├── real_dataset.py
    │   └── precompute_dataset.py
    ├── eval/
    │   ├── dijkstra_baseline.py
    │   └── evaluate.py
    └── utils/
        └── offline_dataset.py
```

## 环境准备

```powershell
cd D:\lunwen
python -m venv venv
& D:\lunwen\venv\Scripts\Activate.ps1
& D:\lunwen\venv\Scripts\python.exe -m pip install -r D:\lunwen\requirements.txt
```

## 训练入口

### 1) CQL 离线训练（主入口）

脚本：`src/agents/train_offline.py`

LEO 示例：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
  --dataset leo `
  --epochs 50 `
  --batches-per-epoch 20 `
  --num-sats 120 `
  --num-gateways 6
```

Walker 示例：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
  --dataset walker `
  --epochs 50 `
  --batches-per-epoch 20 `
  --num-planes 6 `
  --sats-per-plane 10 `
  --inclination-deg 53.0 `
  --use-visibility
```

precomputed 示例（推荐做稳定训练）：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
  --dataset precomputed `
  --epochs 50 `
  --batches-per-epoch 50
```

real TLE 示例：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_offline.py `
  --dataset real `
  --num-sats 100 `
  --use-expert `
  --expert-ratio 0.7
```

### 2) 监督学习（行为克隆）

脚本：`src/agents/train_supervised.py`

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_supervised.py `
  --epochs 50 `
  --batch-size 256 `
  --batches-per-epoch 100
```

### 3) 几何模型训练

脚本：`src/agents/train_geometric.py`

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\agents\train_geometric.py `
  --data-path D:\lunwen\data\offline_trajectories.pkl `
  --checkpoint-dir D:\lunwen\checkpoints `
  --epochs 50
```

## 评估入口

脚本：`src/eval/evaluate.py`

```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\eval\evaluate.py `
  --checkpoint D:\lunwen\checkpoints\leo_routing_epoch50.pt `
  --num-samples 100
```

注意：当前评估脚本中的 DRL 指标是“单跳时延”，Dijkstra 是“端到端时延”，两者用于快速对照，不是严格同口径多跳比较。

## 数据准备

下载并解析 Starlink TLE：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\data\download_tle.py
```

预生成离线专家轨迹：
```powershell
& D:\lunwen\venv\Scripts\python.exe D:\lunwen\src\data\precompute_dataset.py
```

## 输出文件

`train_offline.py` 典型输出：
- `checkpoints/leo_routing_best.pt`
- `checkpoints/leo_routing_epoch{N}.pt`
- `checkpoints/training_history_epoch{N}.json`
- `checkpoints/training_curves_epoch{N}.png`

`train_supervised.py` 典型输出：
- `checkpoints/leo_routing_best.pt`
- `checkpoints/leo_routing_supervised.pt`
- `checkpoints/supervised_training.png`

`train_geometric.py` 典型输出：
- `checkpoints/geo_router_best.pt`
- `checkpoints/geo_training.png`

## 参数说明（train_offline.py）

### 核心参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--dataset` | `synthetic` | `synthetic/leo/walker/real/precomputed` |
| `--epochs` | `3` | 训练轮数 |
| `--batches-per-epoch` | `10` | 每轮 batch 数 |
| `--batch-size` | `128` | 批大小 |
| `--lr` | `3e-4` | 学习率 |
| `--gamma` | `0.99` | 折扣因子 |
| `--cql-alpha` | `0.5` | CQL 正则权重 |
| `--tau` | `0.005` | 目标网络软更新系数 |
| `--grad-clip` | `1.0` | 梯度裁剪阈值 |
| `--patience` | `15` | 早停耐心轮数 |

### 拓扑与模型参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--num-sats` | `120` | 卫星数量（leo/real） |
| `--num-gateways` | `6` | 地面站数量 |
| `--max-neighbors` | `4` | 最大邻居数 |
| `--altitude-km` | `550.0` | 轨道高度 |
| `--node-feature-dim` | `10` | 节点特征维度 |
| `--hidden-dim` | `64` | 隐层维度 |
| `--action-dim` | `4` | 动作维度 |
| `--num-planes` | `6` | Walker 轨道面数 |
| `--sats-per-plane` | `10` | Walker 每面卫星数 |
| `--inclination-deg` | `53.0` | Walker 倾角 |
| `--use-visibility` | `False` | Walker 是否启用可见性约束 |
| `--use-expert` | `False` | real 数据是否混入专家动作 |
| `--expert-ratio` | `0.7` | 专家动作占比 |

## 参数说明（eval/evaluate.py）

| 参数 | 默认值 |
|---|---:|
| `--checkpoint` | `None` |
| `--num-samples` | `100` |
| `--num-sats` | `120` |
| `--num-gateways` | `6` |
| `--max-neighbors` | `4` |
| `--altitude-km` | `550.0` |
| `--node-feature-dim` | `10` |
| `--hidden-dim` | `64` |
| `--action-dim` | `4` |

## 参考

- [CQL: Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)