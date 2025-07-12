from .neural_network import NeuralNetwork
from .forward_pass import forward_pass
import torch

model = NeuralNetwork()

dummy_board = torch.zeros(6, 7)
policy, value = forward_pass(model, dummy_board)

print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
