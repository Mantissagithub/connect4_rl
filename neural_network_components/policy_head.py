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

class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()
        self.reduction_layer = nn.Conv2d(128, 2, kernel_size=1)
        self.dense_layer = nn.Linear(84, 7)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, shared_features):
        reduced_x = self.reduction_layer(shared_features)
        
        flattened = reduced_x.view(reduced_x.size(0), -1)
        
        output = self.dense_layer(flattened)
        
        return self.softmax(output)

# policy_head = PolicyHead()
