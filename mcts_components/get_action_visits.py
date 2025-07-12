#getting the action visits from the mcts tree

def get_action_visits(root_node):
    action_visits = {}
    
    for child in root_node['children']:
        action = child['action']
        visits = child['visits']
        action_visits[action] = visits
    
    return action_visits

def get_action_probabilities(root_node, temperature=1.0):
    action_visits = get_action_visits(root_node)
    
    if not action_visits:
        return {}
    
    if temperature == 0:
        best_action = max(action_visits.items(), key=lambda x: x[1])[0]
        return {action: (1.0 if action == best_action else 0.0) for action in action_visits}
    
    import math
    total_visits = sum(action_visits.values())
    action_probs = {}
    
    for action, visits in action_visits.items():
        prob = (visits / total_visits) ** (1.0 / temperature)
        action_probs[action] = prob
    
    total_prob = sum(action_probs.values())
    if total_prob > 0:
        for action in action_probs:
            action_probs[action] /= total_prob
    
    return action_probs
