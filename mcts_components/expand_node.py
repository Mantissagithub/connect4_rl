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
from .evaluate_board import evaluate_board_position
from .create_node import create_node

def expand_node(node, neural_net=None):
    if node['is_expanded']:
        return node
    
    node['is_expanded'] = True
    
    policy_probs, _ = evaluate_board_position(node['state'], node['current_player'], neural_net)
    
    if policy_probs is None:
        policy_probs = [1.0/7] * 7
    
    for i, move in enumerate(node['valid_moves']):
        col = move if isinstance(move, int) else move[1]
        
        new_board = [row[:] for row in node['state']]
        
        if make_move(new_board, col, node['current_player']):
            child_node = create_node(
                board=new_board,
                parent=node,
                action=col,
                current_player=3 - node['current_player'],
                neural_net=neural_net
            )
            
            if 0 <= col < len(policy_probs):
                child_node['prior'] = policy_probs[col]
            else:
                child_node['prior'] = 0.1
            
            node['children'].append(child_node)
    
    return node
