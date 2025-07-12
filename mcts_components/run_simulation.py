# here we are gonna do the whole simulation process

from select_child import select_child
from expand_node import expand_node
from backpropogate_value import backpropagate_value
from evaluate_board import evaluate_board_position
from create_node import create_root_node, create_node

def run_simulation(board, current_player=1, neural_net=None):
    root_node = create_root_node(board, current_player, neural_net)
    
    #step1 -> selection
    node = select_child(root_node)
    
    # step 2->expansion
    if not node['is_terminal']:
        node = expand_node(node, neural_net)
    
    # step 3 -> evaluation
    value = evaluate_board_position(node['state'], node['current_player'], neural_net)
    
    # step 4 -> backpropogation
    backpropagate_value(node, value)
    
    return root_node