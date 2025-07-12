import torch
import torch.nn as nn
import numpy as np

# from shared_feature import SharedFeatureExtraction
# from policy_head import PolicyHead
# from value_head import ValueHead

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
# shared_feature_extraction = SharedFeatureExtraction()
# policy_head = PolicyHead()
# value_head = ValueHead()

# def forward_pass(board):
#     board_tensor = torch.tensor(board, dtype=torch.float32)

#     empty_channel = (board_tensor == 0).float()
#     player1_channel = (board_tensor == 1).float()
#     player2_channel = (board_tensor == 2).float()
    
#     input_tensor = torch.stack([empty_channel, player1_channel, player2_channel], dim=0).unsqueeze(0)
    
#     shared_features = shared_feature_extraction(input_tensor)
#     policy_output = policy_head(shared_features)
#     value_output = value_head(shared_features)
    
    # return policy_output, value_output

#now this is all done in the neural_network.py file, so will tweak the code here accordingly

def forward_pass(model, board):
    model.eval()

    input_tensor = preprocess_board_state(board, model.get_device())

    with torch.no_grad():
        policy_output, value_output = model(input_tensor)
    
    return policy_output, value_output

def preprocess_board_state(board_state, device):
    if isinstance(board_state, (list, np.ndarray)):
        board_tensor = torch.tensor(board_state, dtype=torch.float32)
    elif isinstance(board_state, torch.Tensor):
        board_tensor = board_state.float()
    else:
        raise ValueError(f"not proper board_state type: {type(board_state)}")
    
    if board_tensor.dim() == 2:
        board_tensor = board_tensor.unsqueeze(0)  
    elif board_tensor.dim() == 3:
        if board_tensor.shape[0] == 3 and board_tensor.shape[1] == 6 and board_tensor.shape[2] == 7:
            board_tensor = board_tensor.unsqueeze(0)  
            return board_tensor.to(device)
    elif board_tensor.dim() == 4:
        return board_tensor.to(device)
    
    batch_size = board_tensor.shape[0]
    
    empty_channel = (board_tensor == 0).float()      
    player1_channel = (board_tensor == 1).float()    
    player2_channel = (board_tensor == 2).float()    
    
    input_tensor = torch.stack([empty_channel, player1_channel, player2_channel], dim=1)
    
    return input_tensor.to(device)

def batch_forward_pass(model, board_states):
    model.eval()
    
    if isinstance(board_states, list):
        board_tensor = torch.stack([torch.tensor(board, dtype=torch.float32) for board in board_states])
    else:
        board_tensor = board_states
    
    input_tensor = preprocess_board_state(board_tensor, model.get_device())
    
    with torch.no_grad():
        policy_outputs, value_outputs = model(input_tensor)
    
    return policy_outputs, value_outputs

def get_move_probabilities(model, board_state, valid_moves=None):
    policy_output, _ = forward_pass(model, board_state)
    
    probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
    
    if valid_moves is not None:
        masked_probs = np.zeros(7)
        for move in valid_moves:
            if 0 <= move < 7:
                masked_probs[move] = probs[move]
        
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            for move in valid_moves:
                masked_probs[move] = 1.0 / len(valid_moves)
        
        return masked_probs
    
    return probs

