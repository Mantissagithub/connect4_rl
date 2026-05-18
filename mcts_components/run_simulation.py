from .select_child import select_child
from .expand_node import expand_node
from .backpropogate_value import backpropagate_value
from .evaluate_board import evaluate_board_position
from .create_node import create_root_node

def run_simulation(board, current_player=1, neural_net=None, num_simulations=100):
    root_node = create_root_node(board, current_player, neural_net)
    
    for _ in range(num_simulations):
        node = root_node
        path = [node]

        while node['is_expanded'] and node['children'] and not node['is_terminal']:
            node = select_child(node)
            path.append(node)

        if node['is_terminal']:
            _, value = evaluate_board_position(node['state'], node['current_player'], neural_net)
            backpropagate_value(path, value)
            continue

        policy_probs, value = evaluate_board_position(node['state'], node['current_player'], neural_net)
        expand_node(node, neural_net=neural_net, policy_probs=policy_probs)
        backpropagate_value(path, value)
    
    return root_node

def run_single_simulation(board, current_player=1, neural_net=None):
    return run_simulation(board, current_player, neural_net, num_simulations=1)
