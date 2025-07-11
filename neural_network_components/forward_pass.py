import torch
import torch.nn as nn

from shared_featue import shared_feature_extraction
from policy_head import policy_head
from value_head import value_head

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

def forward_pass(board):
    empty_places = []
    for r in range(6):
        for c in range(7):
            if board[r][c] == 0:
                empty_places.append([r, c])
    empty_places_tensor = torch.tensor(empty_places, dtype=torch.float32).unsqueeze(0)

    player1_places = []
    for r in range(6):
        for c in range(7):
            if board[r][c] == 1:
                player1_places.append([r, c])
    player1_places_tensor = torch.tensor(player1_places, dtype=torch.float32).unsqueeze(0)

    player2_places = []
    for r in range(6):
        for c in range(7):
            if board[r][c] == 2:
                player2_places.append([r, c])
    player2_places_tensor = torch.tensor(player2_places, dtype=torch.float32).unsqueeze(0)

    shared_feature_extraction_output = shared_feature_extraction(empty_places_tensor, player1_places_tensor, player2_places_tensor)

    policy_output = policy_head(shared_feature_extraction_output)
    value_output = value_head(shared_feature_extraction_output)

    return policy_output, value_output

