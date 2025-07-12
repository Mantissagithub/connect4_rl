import torch.nn.functional as F

def calculate_loss(policy_pred, value_pred, policy_target, value_target, policy_weight=1.0, value_weight=1.0):
    if value_pred.dim() > 1:
        value_pred = value_pred.squeeze(-1)
    if value_target.dim() > 1:
        value_target = value_target.squeeze(-1)
    
    policy_loss = F.kl_div(
        F.log_softmax(policy_pred, dim=1), 
        policy_target, 
        reduction='batchmean'
    )
    
    value_loss = F.mse_loss(value_pred, value_target)
    
    total_loss = policy_weight * policy_loss + value_weight * value_loss
    
    return total_loss, policy_loss, value_loss
