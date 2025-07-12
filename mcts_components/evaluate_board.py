from game_engine_components.check_winner import check_winner
from game_engine_components.is_terminal import is_the_end
from game_engine_components.is_draw import is_draw
import torch

def evaluate_board_position(board, current_player, neural_net=None):
    if is_the_end(board):
        winner = check_winner(board)
        if winner == current_player:
            return None, 1.0  
        elif winner != 0:
            return None, -1.0  
        else:
            return None, 0.0 
    
    if neural_net is not None:
        try:
            policy_probs, value = neural_net.predict(board)
            
            if hasattr(policy_probs, 'tolist'):
                policy_probs = policy_probs.tolist()
            
            if hasattr(value, 'item'):
                value = value.item()
            
            return policy_probs, float(value)
            
        except Exception as e:
            print(f"Neural network evaluation error: {e}")
            pass
    
    uniform_policy = [1.0/7] * 7
    heuristic_value = evaluate_with_heuristic(board, current_player)
    return uniform_policy, heuristic_value

def evaluate_with_neural_network(board, current_player, neural_net):
    try:
        policy_probs, value = neural_net.predict(board)
        return float(value)
    except Exception as e:
        print(f"Neural network evaluation error: {e}")
        return evaluate_with_heuristic(board, current_player)


def evaluate_with_heuristic(board, current_player):
    other_player = 3 - current_player 
    
    current_score = calculate_position_score(board, current_player)
    opponent_score = calculate_position_score(board, other_player)
    
    score_diff = current_score - opponent_score
    max_possible_score = 100  
    
    return max(-1.0, min(1.0, score_diff / max_possible_score))


def calculate_position_score(board, player):
    score = 0
    n_rows = len(board)
    n_cols = len(board[0])
    
    for row in range(n_rows):
        for col in range(n_cols - 3):
            line = [board[row][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    
    for row in range(n_rows - 3):
        for col in range(n_cols):
            line = [board[row + i][col] for i in range(4)]
            score += evaluate_line(line, player)
    
    for row in range(n_rows - 3):
        for col in range(n_cols - 3):
            line = [board[row + i][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    
    for row in range(n_rows - 3):
        for col in range(3, n_cols):
            line = [board[row + i][col - i] for i in range(4)]
            score += evaluate_line(line, player)
    
    return score


def evaluate_line(line, player):
    other_player = 3 - player  
    
    player_count = line.count(player)
    opponent_count = line.count(other_player)
    empty_count = line.count(0)
    
    if opponent_count > 0:
        return 0
    
    if player_count == 4:
        return 1000  
    elif player_count == 3 and empty_count == 1:
        return 50  
    elif player_count == 2 and empty_count == 2:
        return 10   
    elif player_count == 1 and empty_count == 3:
        return 1   
    
    return 0


def get_initial_node_value(board, current_player, neural_net=None):
    return evaluate_board_position(board, current_player, neural_net)

def evaluate_with_neural_network_full(board, current_player, neural_net):
    try:
        from game_engine_components.get_state_tensor import convert_into_tensor
        from neural_network_components.forward_pass import forward_pass
        
        # state_tensor = convert_into_tensor(board)
        policy_logits, value = forward_pass(neural_net, board=board)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        
        if hasattr(policy_probs, 'detach'):
            policy_probs = policy_probs.detach().cpu().numpy()
        if hasattr(value, 'detach'):
            value = value.detach().cpu().item()
        else:
            value = float(value)
        
        return policy_probs.flatten(), value
        
    except Exception as e:
        print(f"error evaluating with neural network: {e}")
        uniform_policy = [1.0/7] * 7  
        heuristic_value = evaluate_with_heuristic(board, current_player)
        return uniform_policy, heuristic_value