import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .generate_self_play_game import generate_self_play_game
from .collect_training_example import collect_batch_examples, validate_training_examples
from .manage_replay_buffer import manage_replay_buffer, save_replay_buffer, load_replay_buffer
from .batch_sampling import sample_training_batch, convert_batch_to_tensors
from neural_network_components.neural_network import NeuralNetwork

neural_net = NeuralNetwork()
replay_buffer = load_replay_buffer("training_data.pkl")

def debug_neural_network_integration(neural_net, board):
    print("=== Neural Network Integration Debug ===")
    
    try:
        policy, value = neural_net.predict(board)
        print(f"✓ Neural network predict: policy shape {len(policy)}, value {value}")
    except Exception as e:
        print(f"✗ Neural network predict failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from mcts_components.run_simulation import run_simulation
        from mcts_components.get_action_visits import get_action_visits
        
        root_node = run_simulation(board, 1, neural_net, num_simulations=5)
        action_visits = get_action_visits(root_node)
        print(f"✓ MCTS simulation: action visits {action_visits}")
        
        if not action_visits:
            print("✗ MCTS returned empty action visits!")
            return False
            
    except Exception as e:
        print(f"✗ MCTS simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

from game_engine_components.intialize_board import initialize_board
test_board = initialize_board()
if not debug_neural_network_integration(neural_net, test_board):
    print("Neural network integration failed - fix before continuing")
    exit(1)
new_games = []
for i in range(3): 
    print(f"Generating game {i+1}/3...")
    game_data = generate_self_play_game(neural_net, num_simulations=10, verbose=True)
    new_games.append(game_data)

new_examples = collect_batch_examples(new_games)
replay_buffer = manage_replay_buffer(replay_buffer, new_examples, max_size=50000)

print(f"Total examples in buffer: {len(replay_buffer)}")

if len(replay_buffer) >= 8: 
    print("Sampling training batch...")
    training_batch = sample_training_batch(replay_buffer, batch_size=min(8, len(replay_buffer)))
    states, policies, values = convert_batch_to_tensors(training_batch)
    print(f"Training batch size: {len(training_batch)}")
    print(f"States shape: {states.shape}")
    print(f"Policies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")
else:
    print(f"Not enough examples ({len(replay_buffer)}) to create training batch")

print("Saving replay buffer...")
save_replay_buffer(replay_buffer, "training_data.pkl")
print("Training setup completed!")
