import random
from typing import Dict, List, Tuple

import numpy as np

from game_engine_components.connect4_env import Connect4Env
from mcts_components.evaluate_board import evaluate_board_position
from mcts_components.get_action_visits import get_action_visits
from mcts_components.run_simulation import run_simulation

from .training_types import TrainExample


def sample_action(move_probs: np.ndarray, temperature: float = 1.0) -> int:
    if temperature <= 0:
        best_value = np.max(move_probs)
        best_actions = np.flatnonzero(np.isclose(move_probs, best_value))
        return int(random.choice(best_actions.tolist()))

    adjusted = np.power(move_probs + 1e-8, 1.0 / temperature)
    adjusted /= np.sum(adjusted)
    return int(np.random.choice(len(adjusted), p=adjusted))


def get_fallback_policy(board, current_player, neural_net) -> np.ndarray:
    policy_probs, _ = evaluate_board_position(board, current_player, neural_net)
    legal_actions = Connect4Env.from_board(board, current_player=current_player).legal_actions()
    masked = np.zeros(7, dtype=np.float32)

    if policy_probs is None:
        policy_probs = [1.0 / 7] * 7

    for action in legal_actions:
        masked[action] = float(policy_probs[action])

    total = float(masked.sum())
    if total <= 0:
        for action in legal_actions:
            masked[action] = 1.0 / len(legal_actions)
        return masked

    return masked / total


def get_move_probs(board, current_player, neural_net, num_simulations=100, verbose=False):
    root_node = run_simulation(
        board=board,
        current_player=current_player,
        neural_net=neural_net,
        num_simulations=num_simulations,
    )
    action_visits = get_action_visits(root_node)
    move_probs = np.zeros(7, dtype=np.float32)
    total_visits = sum(action_visits.values())

    if total_visits > 0:
        for action, visits in action_visits.items():
            move_probs[action] = visits / total_visits
        return move_probs

    return get_fallback_policy(board, current_player, neural_net)


def _temperature_for_move(move_count: int, exploration_moves: int, early_temperature: float, late_temperature: float) -> float:
    return early_temperature if move_count < exploration_moves else late_temperature


def play_self_play_game(
    neural_net,
    num_simulations: int = 100,
    exploration_moves: int = 10,
    early_temperature: float = 1.0,
    late_temperature: float = 0.0,
    verbose: bool = False,
) -> Tuple[List[TrainExample], Dict[str, int]]:
    env = Connect4Env()
    training_data: List[TrainExample] = []
    move_count = 0

    while not env.is_terminal():
        current_player = env.current_player
        move_probs = get_move_probs(env.board, current_player, neural_net, num_simulations, verbose)
        training_data.append(
            TrainExample(
                state=env.encode_state(current_player),
                mcts_policy=move_probs,
                player=current_player,
                value=0.0,
            )
        )

        temperature = _temperature_for_move(move_count, exploration_moves, early_temperature, late_temperature)
        action = sample_action(move_probs, temperature=temperature)
        _, _, _, legal = env.step(action)
        if not legal:
            raise ValueError(f"Illegal action selected during self-play: {action}")
        move_count += 1

    winner = env.winner()
    final_examples: List[TrainExample] = []
    for example in training_data:
        if winner == 0:
            value = 0.0
        elif winner == example.player:
            value = 1.0
        else:
            value = -1.0
        final_examples.append(example._replace(value=value))

    metadata = {
        "winner": winner,
        "num_moves": move_count,
        "is_draw": int(winner == 0),
    }
    return final_examples, metadata


def generate_self_play_game(neural_net, num_simulations=100, verbose=False):
    examples, _ = play_self_play_game(
        neural_net=neural_net,
        num_simulations=num_simulations,
        verbose=verbose,
    )
    return examples


def generate_training_batch(neural_net, num_games=10, num_simulations=100, verbose=False):
    all_training_data = []
    for game_num in range(num_games):
        if verbose and (game_num + 1) % 5 == 0:
            print(f"Generating self-play game {game_num + 1}/{num_games}")
        all_training_data.extend(
            generate_self_play_game(
                neural_net=neural_net,
                num_simulations=num_simulations,
                verbose=verbose and game_num == 0,
            )
        )
    return all_training_data
