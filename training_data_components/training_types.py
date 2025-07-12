# training_types.py - Shared data types for training components

import torch
import numpy as np
from typing import NamedTuple

class TrainExample(NamedTuple):
    state: torch.Tensor
    mcts_policy: np.ndarray
    player: int
    value: float
    # this is how the data will be stored in the training data
