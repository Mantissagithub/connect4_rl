import torch
import torch.nn as nn
from .shared_feature import SharedFeatureExtraction
from .policy_head import PolicyHead
from .value_head import ValueHead
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_channels=3, device=None):
        super(NeuralNetwork, self).__init__()

        self.shared_feature_extraction = SharedFeatureExtraction()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

        self.device = device if device else torch.device("cpu")
        self.to(self.device)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        shared_features = self.shared_feature_extraction(x)

        policy_output = self.policy_head(shared_features)
        value_output = self.value_head(shared_features)

        return policy_output, value_output

    def predict(self, board):
        self.eval()
        with torch.no_grad():
            if isinstance(board, (list, np.ndarray)):
                board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device)
            else:
                # board_tensor = board.to(self.device)
                board_tensor = board.float()
            
            if board_tensor.dim() == 2: #basucally if it only has 2d array like 6x7 values
                empty_channel = (board_tensor == 0).float()
                player1_channel = (board_tensor == 1).float()
                player2_channel = (board_tensor == 2).float()

                input_tensor = torch.stack([empty_channel, player1_channel, player2_channel], dim=0).unsqueeze(0)

            else:
                input_tensor = board_tensor.unsqueeze(0) if board_tensor.dim()==3 else board_tensor

            input_tensor = input_tensor.to(self.device)

            policy_output, value_output = self.forward(input_tensor)

            policy_probs = torch.softmax(policy_output, dim=1).cpu().numpy()[0]
            value = value_output.cpu().item()

            return policy_probs, value
    
    def get_device(self):
        return self.device
    
    def set_device(self, device):
        self.device = device
        self.to(self.device)