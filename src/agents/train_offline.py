import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime

import torch
import torch.nn.utils as nn_utils

CURRENT_DIR = pathlib.Path(__file__).resolve()
SRC_DIR = CURRENT_DIR.parents[1]
REPO_ROOT = SRC_DIR.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.model import LEO_Routing_GNN_CQL
from agents.cql_trainer import compute_cql_loss, soft_update_target_network
from agents.evaluate import build_eval_env, evaluate_policy
from utils.offline_dataset import LEOOfflineDataset, SyntheticOfflineDataset, WalkerOfflineDataset
from data.precompute_dataset import PrecomputedOfflineDataset


def build_run_name(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"offline_{args.dataset}_{timestamp}"


def create_summary_writer(run_name: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("未检测到 tensorboard，跳过 TensorBoard 日志记录。")
        return None, None

    log_dir = CHECKPOINT_DIR / "tensorboard" / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard 日志目录: {log_dir}")
    return writer, log_dir


def create_csv_logger(run_name: str):
    csv_path = CHECKPOINT_DIR / f"{run_name}_metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    fieldnames = [
        "epoch",
        "loss",
        "learning_rate",
        "eval_return",
        "eval_delay",
        "eval_energy",
        "eval_loss",
        "eval_success_rate",
        "is_best",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    print(f"CSV 日志文件: {csv_path}")
    return csv_file, writer, csv_path


def log_tensorboard_metrics(summary_writer, epoch: int, avg_loss: float, current_lr: float, eval_metrics: dict[str, float]) -> None:
    if summary_writer is None:
        return

    summary_writer.add_scalar("train/loss", avg_loss, epoch)
    summary_writer.add_scalar("train/lr", current_lr, epoch)
    summary_writer.add_scalar("eval/return", eval_metrics["mean_return"], epoch)
    summary_writer.add_scalar("eval/delay", eval_metrics["mean_delay"], epoch)
    summary_writer.add_scalar("eval/energy", eval_metrics["mean_energy"], epoch)
    summary_writer.add_scalar("eval/loss", eval_metrics["mean_loss"], epoch)
    summary_writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], epoch)


def plot_training_curves(history, epochs):
    """绘制训练曲线图"""
    import matplotlib.pyplot as plt
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 曲线
    axes[0].plot(history["epoch"], history["loss"], 'b-', linewidth=2, label="CQL Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss over Epochs", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Return 曲线 - 显示环境 rollout 评估 return
    axes[1].plot(history["epoch"], history["eval_return"], 'r-', linewidth=2, label="Policy Mean Return")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Average Return", fontsize=12)
    axes[1].set_title("Policy Evaluation Return over Epochs", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = CHECKPOINT_DIR / f"training_curves_epoch{epochs}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线图已保存至: {save_path}")


def train_offline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(args)
    best_model_path = CHECKPOINT_DIR / "leo_routing_best.pt"
    best_metrics_path = CHECKPOINT_DIR / "leo_routing_best_metrics.json"
    summary_writer, tensorboard_dir = create_summary_writer(run_name)
    csv_file, csv_writer, csv_path = create_csv_logger(run_name)

    model = LEO_Routing_GNN_CQL(
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        dropout=args.dropout,
    ).to(device)
    target_model = LEO_Routing_GNN_CQL(
        node_feature_dim=args.node_feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        dropout=args.dropout,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.dataset == "precomputed":
        # 使用预计算数据集（推荐）
        dataset = PrecomputedOfflineDataset(
            batch_size=args.batch_size,
            device=device,
            total_batches=args.batches_per_epoch,
            action_dim=args.action_dim,
        )
    elif args.dataset == "real":
        # 使用真实 TLE 数据
        from data.real_dataset import RealTLEOfflineDataset
        dataset = RealTLEOfflineDataset(
            max_sats=args.num_sats,
            num_gateways=args.num_gateways,
            node_feature_dim=args.node_feature_dim,
            action_dim=args.action_dim,
            batch_size=args.batch_size,
            device=device,
            total_batches=args.batches_per_epoch,
            use_expert_policy=args.use_expert,
            expert_ratio=args.expert_ratio,
        )
    elif args.dataset == "walker":
        dataset = WalkerOfflineDataset(
            num_planes=args.num_planes,
            sats_per_plane=args.sats_per_plane,
            num_gateways=args.num_gateways,
            node_feature_dim=args.node_feature_dim,
            action_dim=args.action_dim,
            batch_size=args.batch_size,
            max_neighbors=args.max_neighbors,
            device=device,
            total_batches=args.batches_per_epoch,
            altitude_km=args.altitude_km,
            inclination_deg=args.inclination_deg,
            use_visibility=args.use_visibility,
        )
    elif args.dataset == "leo":
        dataset = LEOOfflineDataset(
            num_sats=args.num_sats,
            num_gateways=args.num_gateways,
            node_feature_dim=args.node_feature_dim,
            action_dim=args.action_dim,
            batch_size=args.batch_size,
            max_neighbors=args.max_neighbors,
            device=device,
            total_batches=args.batches_per_epoch,
            altitude_km=args.altitude_km,
        )
    else:
        dataset = SyntheticOfflineDataset(
            num_nodes=args.num_nodes,
            node_feature_dim=args.node_feature_dim,
            action_dim=args.action_dim,
            batch_size=args.batch_size,
            num_edges=args.num_edges,
            device=device,
            total_batches=args.batches_per_epoch,
        )

    model.train()
    eval_env = build_eval_env(
        args,
        device=device,
        max_steps=args.eval_max_steps,
    )
    
    # 记录训练指标
    history = {
        "epoch": [],
        "loss": [],
        "eval_return": [],
        "eval_delay": [],
        "eval_energy": [],
        "eval_loss": [],
        "eval_success_rate": [],
        "learning_rate": [],
    }
    
    # 早停参数
    best_mean_return = float('-inf')
    best_epoch = 0
    patience_counter = 0
    patience = args.patience if hasattr(args, 'patience') else 15

    try:
        for epoch in range(args.epochs):
            epoch_losses = []
            for batch in dataset:
                optimizer.zero_grad()
                loss = compute_cql_loss(
                    model,
                    target_model,
                    batch,
                    gamma=args.gamma,
                    cql_alpha=args.cql_alpha,
                )
                loss.backward()
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                soft_update_target_network(model, target_model, tau=args.tau)
                target_model.eval()
                
                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            # 在固定环境上做闭环 rollout 评估
            eval_metrics = evaluate_policy(
                model,
                eval_env,
                device=device,
                node_feature_dim=args.node_feature_dim,
                action_dim=args.action_dim,
                num_episodes=args.eval_episodes,
                seed=args.eval_seed,
            )
            current_mean_return = eval_metrics["mean_return"]
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            history["epoch"].append(epoch + 1)
            history["loss"].append(avg_loss)
            history["eval_return"].append(eval_metrics["mean_return"])
            history["eval_delay"].append(eval_metrics["mean_delay"])
            history["eval_energy"].append(eval_metrics["mean_energy"])
            history["eval_loss"].append(eval_metrics["mean_loss"])
            history["eval_success_rate"].append(eval_metrics["success_rate"])
            history["learning_rate"].append(current_lr)

            is_best = current_mean_return > best_mean_return
            if is_best:
                best_mean_return = current_mean_return
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                best_metrics = {
                    "run_name": run_name,
                    "epoch": best_epoch,
                    "mean_return": eval_metrics["mean_return"],
                    "mean_delay": eval_metrics["mean_delay"],
                    "mean_energy": eval_metrics["mean_energy"],
                    "mean_loss": eval_metrics["mean_loss"],
                    "success_rate": eval_metrics["success_rate"],
                    "checkpoint_path": str(best_model_path),
                }
                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=2)
            else:
                patience_counter += 1

            log_tensorboard_metrics(
                summary_writer,
                epoch + 1,
                avg_loss,
                current_lr,
                eval_metrics,
            )
            csv_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "eval_return": eval_metrics["mean_return"],
                    "eval_delay": eval_metrics["mean_delay"],
                    "eval_energy": eval_metrics["mean_energy"],
                    "eval_loss": eval_metrics["mean_loss"],
                    "eval_success_rate": eval_metrics["success_rate"],
                    "is_best": int(is_best),
                }
            )
            csv_file.flush()
            
            print(
                f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | "
                f"Return: {eval_metrics['mean_return']:.4f} | "
                f"Delay: {eval_metrics['mean_delay']:.4f} | "
                f"Energy: {eval_metrics['mean_energy']:.4f} | "
                f"Loss: {eval_metrics['mean_loss']:.4f} | "
                f"Success: {eval_metrics['success_rate']:.4f} | "
                f"LR: {current_lr:.2e} | BestReturn: {best_mean_return:.4f}"
            )
            
            if patience_counter >= patience:
                print(f"早停触发！最佳 epoch: {best_epoch}, 最佳 mean_return: {best_mean_return:.4f}")
                break

        # 保存模型
        save_path = CHECKPOINT_DIR / f"leo_routing_epoch{args.epochs}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")
        
        # 保存训练历史
        history_path = CHECKPOINT_DIR / f"training_history_epoch{args.epochs}.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"训练历史已保存至: {history_path}")
        print(f"最佳 checkpoint 指标已保存至: {best_metrics_path}")
        print(f"CSV 日志已保存至: {csv_path}")
        if tensorboard_dir is not None:
            print(f"TensorBoard 日志已保存至: {tensorboard_dir}")
        
        # 绘制训练曲线
        plot_training_curves(history, args.epochs)
    finally:
        if summary_writer is not None:
            summary_writer.flush()
            summary_writer.close()
        csv_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline CQL training for LEO routing")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset", choices=["synthetic", "leo", "walker", "real", "precomputed"], default="synthetic")
    parser.add_argument("--batches-per-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-nodes", type=int, default=128)
    parser.add_argument("--num-edges", type=int, default=512)
    parser.add_argument("--num-sats", type=int, default=120)
    parser.add_argument("--num-gateways", type=int, default=6)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--node-feature-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql-alpha", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15, help="早停的耐心值")
    parser.add_argument("--eval-episodes", type=int, default=16, help="每轮评估的环境 rollout episode 数")
    parser.add_argument("--eval-max-steps", type=int, default=32, help="每个评估 episode 的最大时隙数")
    parser.add_argument("--eval-seed", type=int, default=42, help="评估环境的随机种子起点")
    # Walker 星座参数
    parser.add_argument("--num-planes", type=int, default=6, help="轨道面数量")
    parser.add_argument("--sats-per-plane", type=int, default=10, help="每轨道卫星数")
    parser.add_argument("--inclination-deg", type=float, default=53.0, help="轨道倾角")
    parser.add_argument("--use-visibility", action="store_true", help="启用可见性约束")
    # 真实数据集参数
    parser.add_argument("--use-expert", action="store_true", help="使用 Dijkstra 专家策略")
    parser.add_argument("--expert-ratio", type=float, default=0.7, help="专家动作占比")

    train_offline(parser.parse_args())
