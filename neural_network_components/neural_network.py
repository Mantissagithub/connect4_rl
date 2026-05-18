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

    def _encode_board_for_player(self, board_tensor, current_player):
        other_player = 3 - current_player
        current_channel = (board_tensor == current_player).float()
        opponent_channel = (board_tensor == other_player).float()
        empty_channel = (board_tensor == 0).float()
        return torch.stack([current_channel, opponent_channel, empty_channel], dim=0).unsqueeze(0)

    def predict(self, board, current_player=1):
        self.eval()
        with torch.no_grad():
            if isinstance(board, (list, np.ndarray)):
                board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device)
            else:
                board_tensor = board.float().to(self.device)
            
            if board_tensor.dim() == 2:
                input_tensor = self._encode_board_for_player(board_tensor, current_player)
            else:
                input_tensor = board_tensor.unsqueeze(0) if board_tensor.dim() == 3 else board_tensor

            input_tensor = input_tensor.to(self.device)

            policy_logits, value_output = self.forward(input_tensor)

            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value_output.cpu().item()

            return policy_probs, value
    
    def get_device(self):
        return self.device
    
    def set_device(self, device):
        self.device = device
        self.to(self.device)
