import numpy as np
from game_engine_components.get_valid_moves import get_valid_moves

def sample_action(move_probabilities, temperature=1.0):
    if temperature <= 0:
        return np.argmax(move_probabilities)
    
    if temperature != 1.0:
        move_probabilities = np.power(move_probabilities + 1e-8, 1.0 / temperature)
        move_probabilities = move_probabilities / np.sum(move_probabilities)
    
    try:
        return np.random.choice(len(move_probabilities), p=move_probabilities)
    except ValueError as e:
        print(f"Error sampling action: {e}")
        print(f"Move probabilities: {move_probabilities}")
        
        valid_actions = np.where(move_probabilities > 0)[0]
        return np.random.choice(valid_actions) if len(valid_actions) > 0 else 3

def sample_action_with_masking(move_probabilities, board, temperature=1.0):
    valid_moves = get_valid_moves(board)
    valid_cols = [move if isinstance(move, int) else move[1] for move in valid_moves]
    
    masked_probs = np.zeros(7)
    for col in valid_cols:
        if 0 <= col < 7:
            masked_probs[col] = move_probabilities[col]
    
    if np.sum(masked_probs) > 0:
        masked_probs = masked_probs / np.sum(masked_probs)
    else:
        for col in valid_cols:
            masked_probs[col] = 1.0 / len(valid_cols)
    
    return sample_action(masked_probs, temperature)

def get_action_probabilities_with_temperature(action_visits, temperature=1.0):
    if not action_visits:
        return np.ones(7) / 7  
    
    visits = np.zeros(7)
    for action, count in action_visits.items():
        if 0 <= action < 7:
            visits[action] = count
    
    if temperature == 0:
        best_action = np.argmax(visits)
        probs = np.zeros(7)
        probs[best_action] = 1.0
        return probs
    
    if temperature != 1.0:
        visits = np.power(visits + 1e-8, 1.0 / temperature)
    
    total_visits = np.sum(visits)
    if total_visits > 0:
        return visits / total_visits
    else:
        return np.ones(7) / 7

def sample_multiple_actions(move_probabilities, num_samples=1, temperature=1.0):
    actions = []
    for _ in range(num_samples):
        action = sample_action(move_probabilities, temperature)
        actions.append(action)
    return actions
