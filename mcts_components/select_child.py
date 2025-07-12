from .calculate_ucb import calculate_ucb

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

def select_child(node):
    if not node['children']:
        return node
    
    best_child = None
    best_ucb = float('-inf')
    
    for child in node['children']:
        if child['visits'] > 0:
            q_value = child['total_value'] / child['visits']
        else:
            q_value = child['value']  

        # using thee AlphaGo UCB formula
        ucb_value = calculate_ucb(
            q=q_value,
            p=child['prior'],
            n=node['visits'],
            n_a=child['visits'],
            c_puct=1.0
        )
        
        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_child = child
    
    return best_child if best_child is not None else node