import torch
import torch.nn as nn

# Purpose
# Convert shared features into action probabilities for each of the 7 columns in Connect 4.

# Architecture Details
# Input: Feature maps from shared extraction (typically 128 channels, 6x7 spatial)

# Reduction Layer: 1x1 convolution to reduce channels (128 → 2 channels)

# Flattening: Convert spatial features to vector (2 × 6 × 7 = 84 dimensions)

# Dense Layer: Fully connected layer mapping to 7 outputs (one per column)

# Output Activation: Softmax to ensure probabilities sum to 1.0

# What It Outputs
# Probability Distribution: 7 values representing P(action|state) for each column

# Valid Action Masking: Should be combined with valid move checking

# Strategic Preferences: Higher probabilities for stronger positional moves

# Training Target
# MCTS Policy: Improved policy from tree search visit counts

# Exploration Guidance: Helps MCTS focus on promising moves

#the above thing is for my ref, don't mind!!

def policy_head(shared_features):
    #now reduce to 2 channels
    reduction_layer = nn.Conv2d(128, 2, kernel_size=1)

    reduced_x = reduction_layer(shared_features)

    #now flattening to 2x6x7 = 84 dimesnions
    falttend = reduced_x.view(reduced_x.size(0), -1)

    #now the dense layer
    dense_layer = nn.Linear(84, 7)

    output = dense_layer(falttend)

    #now softmax
    softmax = nn.Softmax(dim=1)

    return softmax(output)