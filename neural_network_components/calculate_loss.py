# Calculate Loss
# Purpose
# Compute training loss by comparing network predictions with target values from self-play.

# Loss Components
# Policy Loss: Cross-entropy between predicted policy and MCTS-improved policy

# Value Loss: Mean squared error between predicted value and actual game outcome

# Combined Loss: Weighted sum of policy and value losses (typically equal weights)

# Mathematical Formulation
# Policy Loss: -Σ(target_policy × log(predicted_policy))

# Value Loss: (target_value - predicted_value)²

# Total Loss: λ₁ × Policy_Loss + λ₂ × Value_Loss

# Regularization
# L2 Weight Decay: Prevent overfitting to training data

# Gradient Clipping: Stabilize training with large gradients

# this is for my ref, don't mind!!

import torch
import torch.nn as nn

# def calculate_loss(predicted_p, target_p, predicted_v, target_v, policy_weight=1.0, value_weight=1.0):
#     policy_loss = nn.CrossEntropyLoss(predicted_p, target_p)
#     value_loss = nn.MSELoss(predicted_v, target_v)
#     total_loss = (policy_weight*policy_loss) + (value_weight*value_loss)
#     return total_loss

#need to set the weights for lambdas 1 and 2
#in alphago they've used it as 1.0 and 1.0, so going with that

#i fucked up the code, i mean the loss function, so i need to rewrite it

def calculate_loss(predicted_p, target_p, predicted_v, target_v, policy_weight=1.0, value_weight=1.0):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    policy_loss = kl_loss(torch.log_softmax(predicted_p, dim=1), target_p)

    mse_loss = nn.MSELoss()
    value_loss = mse_loss(predicted_v, target_v)

    total_loss = (policy_weight*policy_loss) + (value_weight*value_loss)

    return total_loss