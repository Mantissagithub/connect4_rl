import random
import numpy as np
import torch
from typing import List, Tuple

from .training_types import TrainExample

def sample_training_batch(training_data: List[TrainExample], batch_size: int) -> List[TrainExample]:
    if len(training_data) <= batch_size:
        return training_data
    
    return random.sample(training_data, batch_size)

def sample_balanced_batch(training_data: List[TrainExample], 
                         batch_size: int, 
                         balance_players: bool = True) -> List[TrainExample]:
    if not balance_players:
        return sample_training_batch(training_data, batch_size)
    
    player1_examples = [ex for ex in training_data if ex.player == 1]
    player2_examples = [ex for ex in training_data if ex.player == 2]
    
    half_batch = batch_size // 2
    
    p1_sample = random.sample(player1_examples, min(half_batch, len(player1_examples)))
    p2_sample = random.sample(player2_examples, min(half_batch, len(player2_examples)))
    
    remaining_slots = batch_size - len(p1_sample) - len(p2_sample)
    if remaining_slots > 0:
        all_remaining = [ex for ex in training_data if ex not in p1_sample + p2_sample]
        additional_sample = random.sample(all_remaining, min(remaining_slots, len(all_remaining)))
        return p1_sample + p2_sample + additional_sample
    
    return p1_sample + p2_sample

def convert_batch_to_tensors(batch: List[TrainExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.stack([example.state for example in batch])
    policies = torch.stack([torch.tensor(example.mcts_policy, dtype=torch.float32) for example in batch])
    values = torch.tensor([example.value for example in batch], dtype=torch.float32)
    
    return states, policies, values

def sample_recent_batch(training_data: List[TrainExample], 
                       batch_size: int, 
                       recent_fraction: float = 0.3) -> List[TrainExample]:
    if recent_fraction <= 0:
        return sample_training_batch(training_data, batch_size)
    
    recent_count = int(batch_size * recent_fraction)
    random_count = batch_size - recent_count

    # ensuring we don't sample more than available
    recent_start = max(0, len(training_data) - int(len(training_data) * 0.3))
    recent_examples = training_data[recent_start:]
    
    recent_sample = random.sample(recent_examples, min(recent_count, len(recent_examples)))

    remaining_examples = [ex for ex in training_data if ex not in recent_sample]
    random_sample = random.sample(remaining_examples, min(random_count, len(remaining_examples)))
    
    return recent_sample + random_sample

def create_training_batches(training_data: List[TrainExample], 
                           batch_size: int, 
                           num_batches: int) -> List[List[TrainExample]]:
    batches = []
    
    for _ in range(num_batches):
        batch = sample_training_batch(training_data, batch_size)
        batches.append(batch)
    
    return batches

def sample_stratified_batch(training_data: List[TrainExample], 
                           batch_size: int) -> List[TrainExample]:
    win_examples = [ex for ex in training_data if ex.value > 0]
    loss_examples = [ex for ex in training_data if ex.value < 0]
    draw_examples = [ex for ex in training_data if ex.value == 0]
    
    total = len(training_data)
    win_ratio = len(win_examples) / total
    loss_ratio = len(loss_examples) / total
    draw_ratio = len(draw_examples) / total
    
    win_count = int(batch_size * win_ratio)
    loss_count = int(batch_size * loss_ratio)
    draw_count = batch_size - win_count - loss_count
    
    sampled_batch = []
    
    if win_examples:
        sampled_batch.extend(random.sample(win_examples, min(win_count, len(win_examples))))
    if loss_examples:
        sampled_batch.extend(random.sample(loss_examples, min(loss_count, len(loss_examples))))
    if draw_examples:
        sampled_batch.extend(random.sample(draw_examples, min(draw_count, len(draw_examples))))
    
    while len(sampled_batch) < batch_size and len(sampled_batch) < len(training_data):
        remaining = [ex for ex in training_data if ex not in sampled_batch]
        if remaining:
            sampled_batch.append(random.choice(remaining))
    
    return sampled_batch

def batch_sampling(batch_size=32):
    """
    Main batch sampling function that returns batches for training.
    This function loads training data and returns batches.
    """
    try:
        # Try to load existing training data
        from .manage_replay_buffer import load_replay_buffer
        training_data = load_replay_buffer("training_data.pkl")
        
        if not training_data or len(training_data) < batch_size:
            # If not enough data, return empty list
            return []
        
        # Create multiple batches from the available data
        batches = create_training_batches(training_data, batch_size, num_batches=min(10, len(training_data) // batch_size))
        return batches
        
    except Exception as e:
        print(f"Error in batch_sampling: {e}")
        return []
