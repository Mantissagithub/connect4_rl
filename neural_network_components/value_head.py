# Value Head
# Purpose
# Estimate the expected outcome of the current position from the perspective of the player to move.

# Architecture Details
# Input: Same shared features as policy head

# Reduction Layer: 1x1 convolution to single channel (128 → 1 channel)

# Flattening: Convert to vector (1 × 6 × 7 = 42 dimensions)

# Dense Layers: Two fully connected layers (42 → 64 → 1)

# Output Activation: Tanh to bound output between [-1, 1]

# What It Outputs
# Position Evaluation: Single scalar value V(s)

# Range: -1 (losing position) to +1 (winning position)

# Perspective: Always from current player's viewpoint

# Training Target
# Game Outcomes: Final result of self-play games (+1 win, -1 loss, 0 draw)

# Temporal Difference: Can also use bootstrapped values from future positions

#the above thing is for my ref, don't mind!!

import torch
import torch.nn as nn

class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.reduction_layer = nn.Conv2d(128, 1, kernel_size=1)
        self.dense_layer1 = nn.Linear(42, 64)
        self.dense_layer2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, shared_features):
        reduced_x = self.reduction_layer(shared_features)
        
        flattened = reduced_x.view(reduced_x.size(0), -1)
        
        x = self.dense_layer1(flattened)
        x = self.relu(x)
        
        x = self.dense_layer2(x)
        x = self.tanh(x)
        
        return x

value_head = ValueHead()
