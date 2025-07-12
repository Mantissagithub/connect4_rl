# Update Weights
# Purpose
# Apply computed gradients to network parameters using backpropagation and optimization algorithms.

# Optimization Details
# Optimizer Choice: Adam optimizer with adaptive learning rates

# Learning Rate: Typically 0.001 with potential decay schedules

# Gradient Computation: Automatic differentiation through PyTorch

# Parameter Updates: Apply gradients to all network weights and biases

# Training Stability
# Batch Normalization: Normalize intermediate activations

# Learning Rate Scheduling: Reduce learning rate over time

# Gradient Monitoring: Track gradient magnitudes for debugging

# this id for my ref, don't mind!!

import torch
import torch.optim as optim
import torch.nn as nn

def update_weights(model, loss, optimizer):
    optimizer.zero_grad()  #clearing the previous computed gradients
    loss.backward()  #fresh gradient calculation
    optimizer.step()  #updating weights based on the loss gradients
    return model