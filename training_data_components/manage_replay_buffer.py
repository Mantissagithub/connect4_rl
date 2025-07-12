# manage_replay_buffer.py
import pickle
import random
from collections import deque
from typing import List, Optional

from .training_types import TrainExample

class ReplayBuffer:    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0
    
    def add_examples(self, examples: List[TrainExample]):
        self.buffer.extend(examples)
        self.total_added += len(examples)
    
    def get_size(self) -> int:
        return len(self.buffer)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def clear(self):
        self.buffer.clear()
        self.total_added = 0
    
    def get_all_examples(self) -> List[TrainExample]:
        return list(self.buffer)

def manage_replay_buffer(existing_data: List[TrainExample], 
                        new_data: List[TrainExample], 
                        max_size: int = 50000) -> List[TrainExample]:
    combined_data = existing_data + new_data
    
    if len(combined_data) > max_size:
        return combined_data[-max_size:]
    
    return combined_data

def save_replay_buffer(training_data: List[TrainExample], file_path: str):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Saved {len(training_data)} examples to {file_path}")
    except Exception as e:
        print(f"Error saving replay buffer: {e}")

def load_replay_buffer(file_path: str) -> List[TrainExample]:
    try:
        with open(file_path, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} examples from {file_path}")
        return training_data
    except FileNotFoundError:
        print(f"Replay buffer file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading replay buffer: {e}")
        return []

def update_buffer_with_new_games(buffer_file: str, 
                                new_examples: List[TrainExample], 
                                max_size: int = 50000) -> List[TrainExample]:
    existing_data = load_replay_buffer(buffer_file)
    
    updated_buffer = manage_replay_buffer(existing_data, new_examples, max_size)
    
    save_replay_buffer(updated_buffer, buffer_file)
    
    return updated_buffer

def get_buffer_statistics(training_data: List[TrainExample]) -> dict:
    if not training_data:
        return {'size': 0}
    
    stats = {
        'size': len(training_data),
        'player_1_examples': sum(1 for ex in training_data if ex.player == 1),
        'player_2_examples': sum(1 for ex in training_data if ex.player == 2),
        'outcome_distribution': {
            'wins': sum(1 for ex in training_data if ex.value > 0),
            'losses': sum(1 for ex in training_data if ex.value < 0),
            'draws': sum(1 for ex in training_data if ex.value == 0)
        }
    }
    
    return stats
