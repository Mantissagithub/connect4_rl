import torch

def convert_into_tensor(board):
    board_tensor = torch.tensor(board, dtype=torch.float32)
    
    empty_channel = (board_tensor == 0).float()
    player1_channel = (board_tensor == 1).float()
    player2_channel = (board_tensor == 2).float()
    
    input_tensor = torch.stack([empty_channel, player1_channel, player2_channel], dim=0)
    
    return input_tensor