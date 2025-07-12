import torch
import numpy as np
import sys
import os
from typing import NamedTuple, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .training_types import TrainExample

def collect_training_example(board_state, mcts_policy, current_player):
    from game_engine_components.get_state_tensor import convert_into_tensor
    
    state_tensor = convert_into_tensor(board=board_state)
    
    return TrainExample(
        state=state_tensor,
        mcts_policy=mcts_policy,
        player=current_player,
        value=0.0  #will be updated after game ends
    )

def collect_batch_examples(training_data_list: List[List[TrainExample]]) -> List[TrainExample]:
    all_examples = []
    
    for game_data in training_data_list:
        all_examples.extend(game_data)
    
    return all_examples

def validate_training_examples(training_data: List[TrainExample]) -> bool:
    if not training_data:
        print("Warning: No training examples provided")
        return False
    
    for i, example in enumerate(training_data):
        if example.state.shape != (3, 6, 7):
            print(f"Invalid state shape at example {i}: {example.state.shape}")
            return False
        
        if len(example.mcts_policy) != 7:
            print(f"Invalid policy shape at example {i}: {len(example.mcts_policy)}")
            return False
        
        if example.player not in [1, 2]:
            print(f"Invalid player value at example {i}: {example.player}")
            return False
    
    print(f"Validated {len(training_data)} training examples successfully")
    return True

def get_training_statistics(training_data: List[TrainExample]) -> dict:
    if not training_data:
        return {}
    
    stats = {
        'total_examples': len(training_data),
        'player_1_examples': sum(1 for ex in training_data if ex.player == 1),
        'player_2_examples': sum(1 for ex in training_data if ex.player == 2),
        'win_outcomes': sum(1 for ex in training_data if ex.value > 0),
        'loss_outcomes': sum(1 for ex in training_data if ex.value < 0),
        'draw_outcomes': sum(1 for ex in training_data if ex.value == 0)
    }
    
    return stats
