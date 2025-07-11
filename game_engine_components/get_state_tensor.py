import torch

def convert_into_tensor(board):
    tenosr = torch.tensor(board, dtype=torch.float32)
    return tenosr.unsqueeze(0)  #using unqueeze here to send it like with dimension knowledge of the board, not chosing unsqueeze(1) as it give it like (6, 1, 7), what the fuck can i do with this, and then unqueeze(2), gives it like (6, 7, 1), same shit!!