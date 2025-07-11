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

def shared_feature_extraction(empty_places, player1_places, player2_places):
    empty_places_tensor = torch.tensor(empty_places, dtype=torch.float32).unsqueeze(0)
    player1_places_tensor = torch.tensor(player1_places, dtype=torch.float32).unsqueeze(0)
    player2_places_tensor = torch.tensor(player2_places, dtype=torch.float32).unsqueeze(0)
    board_tensor = torch.cat((empty_places_tensor, player1_places_tensor, player2_places_tensor), dim=0).unsqueeze(0)

    #conv layers now to extract certain features
    conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    x = conv1(board_tensor)
    x = nn.ReLU()(x) #obvious relu activations, maybe will migrate to gelu if needed, need to explore, as of now
    x = conv2(x)
    x = nn.ReLU()(x)
    x = conv3(x)
    x = nn.ReLU()(x)

    return x
