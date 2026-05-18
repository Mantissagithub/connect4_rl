from .calculate_ucb import calculate_ucb
import random

def select_child(node):
    if not node['children']:
        return node
    
    best_children = []
    best_ucb = float('-inf')
    
    for child in node['children']:
        q_value = 0.0 if child['visits'] == 0 else -(child['total_value'] / child['visits'])

        ucb_value = calculate_ucb(
            q=q_value,
            p=child['prior'],
            n=node['visits'],
            n_a=child['visits'],
            c_puct=1.0
        )
        
        if ucb_value > best_ucb:
            best_ucb = ucb_value
            best_children = [child]
        elif ucb_value == best_ucb:
            best_children.append(child)
    
    return random.choice(best_children) if best_children else node
