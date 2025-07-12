import torch
import torch.nn as nn

from shared_feature import SharedFeatureExtraction
from policy_head import PolicyHead
from value_head import ValueHead

# Forward Pass
# Purpose
# Complete inference pipeline that processes board state and returns both policy and value predictions.

# Implementation Flow
# Input Preprocessing: Convert board to 3-channel tensor format

# Feature Extraction: Pass through shared convolutional layers

# Dual Head Processing: Split features to both policy and value heads

# Output Generation: Return policy probabilities and value estimate

# Batch Processing: Handle multiple positions simultaneously for efficiency

# Computational Considerations
# GPU Acceleration: Utilize CUDA for faster matrix operations

# Batch Inference: Process multiple positions together

# Memory Management: Efficient tensor operations to avoid memory leaks

# the above thing is for my ref, don't mind!!

#initializing the diff modules here
shared_feature_extraction = SharedFeatureExtraction()
policy_head = PolicyHead()
value_head = ValueHead()

def forward_pass(board):
    board_tensor = torch.tensor(board, dtype=torch.float32)

    empty_channel = (board_tensor == 0).float()
    player1_channel = (board_tensor == 1).float()
    player2_channel = (board_tensor == 2).float()
    
    input_tensor = torch.stack([empty_channel, player1_channel, player2_channel], dim=0).unsqueeze(0)
    
    shared_features = shared_feature_extraction(input_tensor)
    policy_output = policy_head(shared_features)
    value_output = value_head(shared_features)
    
    return policy_output, value_output
