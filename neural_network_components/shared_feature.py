# Shared Feature Extraction
# Purpose
# Extract spatial patterns and features from the Connect 4 board state that are useful for both policy and value estimation.

# Architecture Details
# Input Processing: Takes 3-channel board representation (empty cells, player 1 pieces, player 2 pieces)

# Convolutional Layers: 3-4 Conv2D layers with increasing filter counts (32 → 64 → 128)

# Kernel Sizes: Typically 3x3 kernels to capture local patterns like potential winning sequences

# Activation Functions: ReLU activations after each convolutional layer

# Spatial Preservation: Maintains 6x7 spatial dimensions through padding

# Feature Maps: Final layer outputs rich feature representations (e.g., 128 channels)

# What It Learns
# Horizontal, vertical, and diagonal threat patterns

# Blocking opportunities and defensive positions

# Center column importance and positional advantages

# Multi-piece combinations and strategic formations

#the above is for my ref, don't mind!!

import torch
import torch.nn as nn

class SharedFeatureExtraction(nn.Module):
    def __init__(self):
        super(SharedFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, board_tensor):
        x = self.conv1(board_tensor)
        x = self.relu(x)  # obvious relu activations, maybe will migrate to gelu if needed, need to explore, as of now
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        return x

shared_feature_extraction = SharedFeatureExtraction()
