from game_engine_components.get_valid_moves import get_valid_moves
from evaluate_board import get_initial_node_value


def create_node(board, parent=None, action=None, current_player=1, neural_net=None):
    initial_value = get_initial_node_value(board, current_player, neural_net)
    
    node = {
        'state': board,
        'parent': parent,
        'action': action,
        'children': [],
        'visits': 0,
        'value': initial_value,  
        'total_value': 0,       
        'current_player': current_player,
        'valid_moves': get_valid_moves(board),
        'is_expanded': False,  
        'is_terminal': False     
    }

    return node


def create_root_node(board, current_player=1, neural_net=None):
    return create_node(board, parent=None, action=None, 
                      current_player=current_player, neural_net=neural_net)