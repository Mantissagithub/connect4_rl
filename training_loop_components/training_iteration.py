import torch
import time
from typing import List, Tuple, Dict

def training_iteration(
    neural_net,
    optimizer,
    nself_play=10,
    num_simulations=100,
    batch_size=32,
    num_epochs=10,
    verbose=True,
    replay_buffer_path="training_data.pkl",
    replay_buffer_size=50000,
    status_callback=None,
):

    iteration_start_time = time.time()
    metrics = {}

    if verbose:
        print(f"Starting iteration with {nself_play} self-play games and {num_simulations} simulations per game.")
        print(f"Batch size: {batch_size}, Number of epochs: {num_epochs}")

    from training_data_components.generate_self_play_game import generate_self_play_game
    from training_data_components.manage_replay_buffer import (
        load_replay_buffer,
        manage_replay_buffer,
        save_replay_buffer,
    )
    from training_loop_components.network_training import network_training

    replay_buffer = load_replay_buffer(replay_buffer_path)
    total_examples = 0
    for game_idx in range(nself_play):
        if verbose and (game_idx+1)%5==0:
            print(f"generated {game_idx+1}/{nself_play} self-play games")
        if status_callback is not None:
            status_callback("self_play_game_start", {"game": game_idx + 1, "total_games": nself_play})

        game_data = generate_self_play_game(neural_net=neural_net, num_simulations=num_simulations, verbose=verbose)
        replay_buffer = manage_replay_buffer(replay_buffer, game_data, max_size=replay_buffer_size)
        total_examples += len(game_data)
        if status_callback is not None:
            status_callback(
                "self_play_game_done",
                {
                    "game": game_idx + 1,
                    "total_games": nself_play,
                    "examples": len(game_data),
                    "replay_buffer_size": len(replay_buffer),
                },
            )

    save_replay_buffer(replay_buffer, replay_buffer_path)
    
    if verbose:
        print(f"Generated {total_examples} training examples from {nself_play} games")
    
    if verbose:
        print("Starting neural network training...")
    
    training_metrics = network_training(
        neural_net=neural_net,
        optimizer=optimizer,
        training_data=replay_buffer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        verbose=verbose,
        status_callback=status_callback,
    )
    
    iteration_time = time.time() - iteration_start_time
    metrics.update({
        'iteration_time': iteration_time,
        'total_examples': total_examples,
        'games_played': nself_play,
        'replay_buffer_size': len(replay_buffer),
        **training_metrics
    })
    
    if verbose:
        print(f"Training iteration completed in {iteration_time:.2f} seconds")
        print(f"Final loss: {training_metrics.get('final_loss', 'N/A')}")
    
    return metrics
