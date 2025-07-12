# here we are gonna do the whole simulation process

from .select_child import select_child
from .expand_node import expand_node
from .backpropogate_value import backpropagate_value
from .evaluate_board import evaluate_board_position
from .create_node import create_root_node, create_node

def run_simulation(board, current_player=1, neural_net=None, num_simulations=100):
    root_node = create_root_node(board, current_player, neural_net)
    
    for _ in range(num_simulations):
        #step1 -> selection
        node = select_child(root_node)
        
        #step2-> expansion
        if not node['is_terminal'] and not node['is_expanded']:
            node = expand_node(node, neural_net)
            # Select one of the new children for evaluation
            if node['children']:
                node = node['children'][0]
        
        #step3 -> evaluation
        _, value = evaluate_board_position(node['state'], node['current_player'], neural_net)
        
        #step4 -> backpropagation
        backpropagate_value(node, value)
    
    return root_node

def run_single_simulation(board, current_player=1, neural_net=None):
    return run_simulation(board, current_player, neural_net, num_simulations=1)