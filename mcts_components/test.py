from neural_network_components.neural_network import NeuralNetwork
from mcts_components.run_simulation import run_simulation
from game_engine_components.intialize_board import initialize_board

neural_net = NeuralNetwork()
board = initialize_board()

root_node = run_simulation(board, current_player=1, neural_net=neural_net, num_simulations=50)

from mcts_components.get_action_visits import get_action_visits
action_visits = get_action_visits(root_node)
print(f"Action visits: {action_visits}")
