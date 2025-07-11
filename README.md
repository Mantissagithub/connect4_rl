# Connect4 RL

Creating a self playing agent for Connect4 using RL, inspired from AlphaGo.

## Component Architecture

### 1. Game Engine Component
- **initialize_board**: Set up empty 6x7 Connect 4 board - done
- **get_valid_moves**: Return list of playable columns
- **make_move**: Drop piece in specified column
- **check_winner**: Detect win/loss/draw conditions
- **get_state_tensor**: Convert board to neural network input format
- **copy_game**: Create deep copy of current game state
- **reset_game**: Return to initial empty board
- **is_terminal**: Check if game has ended
- **is_draw**: Check if game is a draw

### 2. Neural Network Component
- **shared_feature_extraction**: Convolutional layers for spatial patterns
- **policy_head**: Output action probabilities for each column
- **value_head**: Output single value estimating position strength
- **forward_pass**: Complete network inference
- **calculate_loss**: Compute policy and value losses
- **update_weights**: Apply gradients to network parameters
- **save_model**: Store trained weights to file
- **load_model**: Restore weights from file

### 3. MCTS Component
- **create_node**: Initialize tree node with game state
- **calculate_ucb_score**: Balance exploration vs exploitation
- **select_child**: Choose best child node using UCB
- **expand_node**: Add children for all valid actions
- **evaluate_position**: Use neural network to assess leaf nodes
- **backpropagate_value**: Update statistics up the tree
- **run_simulations**: Execute full MCTS search process
- **get_action_visits**: Return visit counts for root children

### 4. Training Data Component
- **generate_self_play_game**: Play complete game using MCTS
- **collect_training_examples**: Store state-policy-value tuples
- **sample_action**: Choose move from MCTS policy distribution
- **assign_game_outcomes**: Fill in final results for all positions
- **manage_replay_buffer**: Maintain fixed-size training data storage
- **batch_sampling**: Randomly select training examples

### 5. Training Loop Component
- **training_iteration**: One complete cycle of data generation and learning
- **network_training**: Update neural network on collected data
- **save_checkpoint**: Store model at regular intervals
- **evaluate_progress**: Test agent strength against baselines
- **adjust_hyperparameters**: Modify learning rates and exploration

### 6. Agent Interface Component
- **select_best_move**: Choose action for given position
- **set_playing_strength**: Adjust MCTS simulation count
- **load_trained_model**: Initialize with pre-trained weights
- **play_game**: Interface for human or agent opponents