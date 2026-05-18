from game_engine_components.connect4_env import Connect4Env
from .evaluate_board import evaluate_board_position

def create_node(board, parent=None, action=None, current_player=1, neural_net=None):
    env = Connect4Env.from_board(board, current_player=current_player)
    _, initial_value = evaluate_board_position(env.board, current_player, neural_net)
    winner = env.winner()
    is_terminal = env.is_terminal()

    node = {
        'state': env.board,
        'parent': parent,
        'action': action,
        'children': [],
        'visits': 0,
        'value': initial_value,
        'total_value': 0.0,
        'prior': 0.0,
        'current_player': current_player,
        'valid_moves': env.legal_actions(),
        'is_expanded': is_terminal,
        'is_terminal': is_terminal,
        'winner': winner,
    }

    return node


def create_root_node(board, current_player=1, neural_net=None):
    return create_node(board, parent=None, action=None, 
                      current_player=current_player, neural_net=neural_net)
