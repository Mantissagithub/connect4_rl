from calculate_ucb import calculate_ucb

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

def select_child(Node):
    Node['visits'] += 1  
    if not Node['children']:
        return Node 
    best_child = None
    best_ucb = float('-inf') #similar to INT_MIN in cpp

    for child in Node['children']:
        ucb_value = calculate_ucb(child['value'], child['visits'], Node['visits'])
        
        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_child = child

    return best_child if best_child is not None else Node  