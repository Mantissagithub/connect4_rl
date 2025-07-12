# # the node structure is like this: node = {
#         'state': board,
#         'parent': parent,
#         'action': action,
#         'children': [],
#         'visits': 0,
#         'value': initial_value,  
#         'total_value': 0,       
#         'current_player': current_player,
#         'valid_moves': get_valid_moves(board),
#         'is_expanded': False,  
#         'is_terminal': False     
#     }

from game_engine_components.make_move import make_move
from mcts_components.evaluate_board import evaluate_board_position
from mcts_components.create_node import create_node

def expand_node(node, neural_net=None):
    if node['is_expanded']:
        return node
    
    node['is_expanded'] = True
    node['is_terminal'] = False
    
    for move in node['valid_moves']:
        row, col = move
        new_board = [row[:] for row in node['state']]
        
        if make_move(new_board, row, col, node['current_player']):
            child_node = create_node(
                board=new_board,
                parent=node,
                action=move,
                current_player=3 - node['current_player'], 
                neural_net=neural_net
            )
            child_node['value'] = evaluate_board_position(new_board, child_node['current_player'], neural_net)
            node['children'].append(child_node)
    
    return node