#in this file we are gonna make the game self playable

import numpy as np
import random
import torch
from typing import List, Tuple, Dict, NamedTuple #the named tuple is same thing as base class model initialization in fastapi backends, i'll try this one

from .training_types import TrainExample

#getting the game engine components -> initialize board, make_move, check_winner, is_terminal, is_draw, get_state_tensor, get_valid_moves
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_engine_components.intialize_board import initialize_board
from game_engine_components.make_move import make_move
from game_engine_components.check_winner import check_winner
from game_engine_components.is_terminal import is_the_end
from game_engine_components.is_draw import is_draw
from game_engine_components.get_state_tensor import convert_into_tensor
from game_engine_components.get_valid_moves import get_valid_moves
from game_engine_components.copy_game import deep_copy

#neural network components -> forward pass includes everything, calculate loss, update weights, save model, load model
# from neural_network_components.forward_pass import forward_pass
# from neural_network_components.calculate_loss import calculate_loss
# from neural_network_components.update_weights import update_weights
# from neural_network_components.save_model import save_model
# from neural_network_components.load_model import load_model
from neural_network_components.neural_network import NeuralNetwork
from neural_network_components.load_model import load_model_for_inference

#MCTS components -> create_node, expand_node, select_child, backpropagate, run_mcts, get_action_visits
# from mcts_components.create_node import create_root_node
# from mcts_components.expand_node import expand_node
# from mcts_components.select_child import select_child
# from mcts_components.backpropogate_value import backpropagate
from mcts_components.run_simulation import run_simulation
from mcts_components.get_action_visits import get_action_visits

def get_neural_net_uniform_policy(board, current_player, neural_net):
    try:
        # state = convert_into_tensor(board=board)
        policy_probs, _ = neural_net.predict(board)

        valid_moves = get_valid_moves(board)
        valid_cols = [move if isinstance(move, int) else move[1] for move in valid_moves]

        masked_probs = np.zeros(7)
        for col in valid_cols:
            if 0 <= col < 7:
                masked_probs[col] = policy_probs[col]

        if np.sum(masked_probs)>0:
            masked_probs /= np.sum(masked_probs)
        else:
            for col in valid_cols:
                masked_probs[col] = 1.0 / len(valid_cols)

        return masked_probs

    except Exception as e:
        print(f"neural network policy error: {e}")
        valid_moves = get_valid_moves(board)
        valid_cols = [move if isinstance(move, int) else move[1] for move in valid_moves]
        uniform_probs = np.zeros(7)
        for col in valid_cols:
            uniform_probs[col] = 1.0 / len(valid_cols)
        return uniform_probs
    
def sample_action(move_probs, temperature=1.0):
    if temperature <= 0:
        return np.argmax(move_probs)
    
    if temperature != 1.0:
        move_probs = np.power(move_probs+1e-8, 1/temperature)
        move_probs /= np.sum(move_probs)
    
    try:
        return np.random.choice(len(move_probs), p=move_probs)
    except ValueError as e:
        print(f"error sampling action: {e}, move_probs: {move_probs}")
        valid_actions = np.where(move_probs > 0)[0]
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else 3

def get_game_outcome(board):
    winner = check_winner(board)
    if winner == 1:
        return 1  
    elif winner == 2:
        return -1.0
    else:
        return 0.0 

def assign_game_outcomes(training_data: List[TrainExample], game_outcome: int) -> List[TrainExample]:
    final_training_data = []
    for data in training_data:
        player = data.player

        if player == 1:
            value = game_outcome
        else:
            value = -game_outcome

        final_training_data.append(TrainExample(
            state=data.state,
            mcts_policy=data.mcts_policy,
            player=player,
            value=value
        ))

    return final_training_data

def create_fresh_neural_network():
    return NeuralNetwork()

def print_board(board):
    print("\n  0 1 2 3 4 5 6")
    print("  " + "-" * 13)
    for row in board:
        print("| " + " ".join(str(cell) if cell != 0 else "." for cell in row) + " |")
    print("  " + "-" * 13)

def save_training_data(training_data: List[TrainExample], file_path: str):
    import pickle
    with open(file_path, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} training examples to {file_path}")

def load_training_data(file_path: str) -> List[TrainExample]:
    import pickle
    try:
        with open(file_path, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} training examples from {file_path}")
        return training_data
    except FileNotFoundError:
        print(f"Training data file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

def get_move_probs(board, current_player, neural_net, num_simulations=100, verbose=False):
    if verbose:
        print(f"running mcts for player {current_player} with {num_simulations} simulations")

    print(f"Getting move probs for player {current_player}")

    root_node = run_simulation(board=board, current_player=current_player, neural_net=neural_net, num_simulations=num_simulations)

    action_visited = get_action_visits(root_node)

    if verbose:
        print(f"action visits: {action_visited}")

    move_probs = np.zeros(7)
    total_visits = sum(action_visited.values())

    if total_visits > 0:
        for action, visits in action_visited.items():
            move_probs[action] = visits / total_visits

    if total_visits == 0:
        if verbose:
            print("no valid moves found, using uniform policy")
        move_probs = get_neural_net_uniform_policy(board, current_player, neural_net)
    
    valid_moves = get_valid_moves(board)
    valid_cols = [move if isinstance(move, int) else move[1] for move in valid_moves]

    print(f"Final move probabilities: {move_probs}")
    print(f"Sum of probabilities: {sum(move_probs)}")
    
    # return move_probs

    masked_probs = np.zeros(7)
    for col in valid_cols:
        if 0 <= col < 7:
            masked_probs[col] = move_probs[col]

    if np.sum(masked_probs)>0:
        masked_probs /= np.sum(masked_probs)
    else:
        for col in valid_cols:
            masked_probs[col] = 1.0 / len(valid_cols)

    print(f"Masked move probabilities: {masked_probs}")
    print(f"Sum of masked probabilities: {np.sum(masked_probs)}")

    return masked_probs

def generate_self_play_game(neural_net, num_simulations=100, temperature=1.0, verbose=False):
    board = initialize_board()
    training_data = []  
    current_player = 1
    move_count = 0
    
    while not is_the_end(board):
        move_probs = get_move_probs(board, current_player, neural_net, num_simulations, verbose)
        
        state = convert_into_tensor(board=board)
        training_example = TrainExample(
            state=state,
            mcts_policy=move_probs,
            player=current_player,
            value=0.0  
        )
        training_data.append(training_example)
        
        action = sample_action(move_probs, temperature)
        success = make_move(board, action, current_player)
        
        if not success:
            break
            
        current_player = 3 - current_player
        move_count += 1
    
    winner = check_winner(board)
    for i, example in enumerate(training_data):
        if winner == example.player:
            training_data[i] = example._replace(value=1.0)  # Win
        elif winner != 0:
            training_data[i] = example._replace(value=-1.0)  # Loss
        else:
            training_data[i] = example._replace(value=0.0)  # Draw
    
    print(f"Game ended with {len(training_data)} training examples")
    return training_data


def generate_training_batch(neural_net, num_games=10, num_mcts_simulations=100, verbose=False):
    all_training_data = []
    
    for game_num in range(num_games):
        if verbose or game_num % 5 == 0:
            print(f"Generating self-play game {game_num + 1}/{num_games}")
        
        temperature = 1.0 if game_num < num_games * 0.7 else 0.5
        
        game_data = generate_self_play_game(
            neural_net=neural_net,
            num_mcts_simulations=num_mcts_simulations,
            temperature=temperature,
            verbose=verbose and game_num == 0 
        )
        
        all_training_data.extend(game_data)
    
    return all_training_data

if __name__ == "__main__":
    print("connect 4 self play game generator")
    print()

    try:
        neural_net = load_model_for_inference("model.pth")
        print("loaded neural network model for inference from path")
    except:
        neural_net = create_fresh_neural_network()
        print("created a fresh neural network model")

    print("starting self-play game generation..")
    training_data = generate_self_play_game(neural_net=neural_net, num_simulations=100, temperature=0.8, verbose=True)

    print(f"generated {len(training_data)} training examples")

    print("\n generating training batch")
    batch_data = generate_training_batch(neural_net=neural_net, num_games=50, num_mcts_simulations=100, verbose=True)

    save_training_data(batch_data, "training_data.pkl")

    print(f"saved {len(batch_data)} training examples to training_data.pkl")
    print("self-play game generation completed")