from typing import List
from game_engine_components.check_winner import check_winner
from .training_types import TrainExample
import numpy as np

def get_game_outcome(board):
    winner = check_winner(board)
    
    if winner == 1:
        return 1.0  
    elif winner == 2:
        return -1.0  #player 2 wins from player 1's perspective
    else:
        return 0.0  # draw

def assign_game_outcomes(training_data: List[TrainExample], game_outcome: float) -> List[TrainExample]:
    final_training_data = []
    
    for data_point in training_data:
        player = data_point.player
        
        if player == 1:
            value = game_outcome
        else:
            value = -game_outcome  
        
        final_training_data.append(TrainExample(
            state=data_point.state,
            mcts_policy=data_point.mcts_policy,
            player=player,
            value=value
        ))
    
    return final_training_data

def assign_outcomes_to_batch(game_results: List[List[TrainExample]], outcomes: List[float]) -> List[TrainExample]:
    all_training_data = []
    
    for game_data, outcome in zip(game_results, outcomes):
        game_with_outcomes = assign_game_outcomes(game_data, outcome)
        all_training_data.extend(game_with_outcomes)
    
    return all_training_data

def calculate_outcome_statistics(training_data: List[TrainExample]) -> dict:
    if not training_data:
        return {}
    
    outcomes = [example.value for example in training_data]
    
    stats = {
        'total_positions': len(outcomes),
        'winning_positions': sum(1 for v in outcomes if v > 0),
        'losing_positions': sum(1 for v in outcomes if v < 0),
        'draw_positions': sum(1 for v in outcomes if v == 0),
        'average_outcome': np.mean(outcomes),
        'outcome_distribution': {
            'wins': sum(1 for v in outcomes if v > 0) / len(outcomes),
            'losses': sum(1 for v in outcomes if v < 0) / len(outcomes),
            'draws': sum(1 for v in outcomes if v == 0) / len(outcomes)
        }
    }
    
    return stats

def validate_outcome_assignment(training_data: List[TrainExample]) -> bool:
    for i, example in enumerate(training_data):
        if example.value not in [-1.0, 0.0, 1.0]:
            print(f"Invalid outcome value at position {i}: {example.value}")
            return False
    
    print("All outcomes properly assigned")
    return True
