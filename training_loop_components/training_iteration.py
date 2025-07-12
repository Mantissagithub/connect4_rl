import torch
import time
from typing import List, Tuple, Dict

def training_iteration(neural_net, optimizer, nself_play=10, num_simulations=100, batch_size=32, num_epochs=10, verbose=True):

    iteration_start_time = time.time()
    metrics = {}

    if verbose:
        print(f"Starting iteration with {nself_play} self-play games and {num_simulations} simulations per game.")
        print(f"Batch size: {batch_size}, Number of epochs: {num_epochs}")

    from training_data_components.generate_self_play_game import generate_self_play_game
    from training_data_components.manage_replay_buffer import manage_replay_buffer
    from network_training import network_training

    total_examples = 0
    for game_idx in range(nself_play):
        if verbose and (game_idx+1)%5==0:
            print(f"generated {game_idx+1}/{nself_play} self-play games")

        game_data = generate_self_play_game(neural_net=neural_net, num_simulations=num_simulations, verbose=verbose)
        
        manage_replay_buffer(game_data)
        total_examples += len(game_data)
    
    if verbose:
        print(f"Generated {total_examples} training examples from {nself_play} games")
    
    if verbose:
        print("Starting neural network training...")
    
    training_metrics = network_training(
        neural_net=neural_net,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        verbose=verbose
    )
    
    iteration_time = time.time() - iteration_start_time
    metrics.update({
        'iteration_time': iteration_time,
        'total_examples': total_examples,
        'games_played': nself_play,
        **training_metrics
    })
    
    if verbose:
        print(f"Training iteration completed in {iteration_time:.2f} seconds")
        print(f"Final loss: {training_metrics.get('final_loss', 'N/A')}")
    
    return metrics

