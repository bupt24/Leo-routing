import torch
import torch.nn.functional as F


def compute_cql_loss(q_net, target_q_net, batch_data, gamma=0.99, cql_alpha=1.0):
    """
    计算 CQL 损失（带数值稳定性处理）
    
    CQL Loss = TD Loss + alpha * (logsumexp(Q) - Q(s,a))
    
    当 cql_alpha=0 时退化为标准 DQN
    """
    state_x, state_edge, curr_idx, dest_idx = batch_data["state"]
    next_x, next_edge, next_curr_idx, next_dest_idx = batch_data["next_state"]
    actions_taken = batch_data["action"]
    rewards = batch_data["reward"]
    dones = batch_data["done"]
    action_mask = batch_data.get("action_mask")
    next_action_mask = batch_data.get("next_action_mask")

    # 当前状态的 Q 值
    current_q_values = q_net(state_x, state_edge, curr_idx, dest_idx)
    
    # 用较小的负值（-100）而非 -1e9，避免数值问题
    MASK_VALUE = -100.0
    
    if action_mask is not None:
        masked_current_q = current_q_values.masked_fill(action_mask == 0, MASK_VALUE)
    else:
        masked_current_q = current_q_values
    
    # 获取实际采取动作的 Q 值（使用原始 Q 值，不是 masked）
    q_value_for_action = current_q_values.gather(1, actions_taken.unsqueeze(1)).squeeze(1)

    # 计算目标 Q 值
    with torch.no_grad():
        next_q_values = target_q_net(next_x, next_edge, next_curr_idx, next_dest_idx)
        if next_action_mask is not None:
            masked_next_q = next_q_values.masked_fill(next_action_mask == 0, MASK_VALUE)
        else:
            masked_next_q = next_q_values
        max_next_q = masked_next_q.max(dim=1).values
        # 限制目标 Q 值范围，防止过大
        max_next_q = torch.clamp(max_next_q, min=-100.0, max=100.0)
        target_q = rewards + gamma * (1 - dones) * max_next_q

    # TD 损失（使用 Huber Loss 更稳定）
    bellman_td_loss = F.smooth_l1_loss(q_value_for_action, target_q)

    # 如果 cql_alpha > 0，加入 CQL 正则化
    if cql_alpha > 0:
        # CQL 正则化项：惩罚过高的 Q 值估计
        cql_logsumexp = torch.logsumexp(masked_current_q, dim=1)
        cql_regularization = (cql_logsumexp - q_value_for_action).mean()
        cql_regularization = torch.clamp(cql_regularization, min=-10.0, max=100.0)
        total_loss = bellman_td_loss + cql_alpha * cql_regularization
    else:
        total_loss = bellman_td_loss
    
    return total_loss


def soft_update_target_network(model, target_model, tau=0.005):
    for target_param, model_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * model_param.data + (1.0 - tau) * target_param.data)
