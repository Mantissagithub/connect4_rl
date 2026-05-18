import random

from game_engine_components.connect4_env import Connect4Env
from .evaluate_board import evaluate_board_position
from .create_node import create_node

def _mask_policy(policy_probs, valid_moves):
    masked = [0.0] * 7
    for move in valid_moves:
        masked[move] = policy_probs[move]

    total = sum(masked)
    if total <= 0:
        uniform = 1.0 / len(valid_moves)
        for move in valid_moves:
            masked[move] = uniform
        return masked

    return [prob / total for prob in masked]

def expand_node(node, neural_net=None, policy_probs=None):
    if node['is_expanded'] or node['is_terminal']:
        return node

    node['is_expanded'] = True
    if policy_probs is None:
        policy_probs, _ = evaluate_board_position(node['state'], node['current_player'], neural_net)

    if policy_probs is None:
        policy_probs = [1.0 / 7] * 7

    masked_policy = _mask_policy(policy_probs, node['valid_moves']) if node['valid_moves'] else [0.0] * 7

    for col in node['valid_moves']:
        child_env = Connect4Env.from_board(node['state'], current_player=node['current_player'])
        _, _, _, legal = child_env.step(col)
        if legal:
            child_node = create_node(
                board=child_env.board,
                parent=node,
                action=col,
                current_player=child_env.current_player,
                neural_net=neural_net
            )
            child_node['prior'] = masked_policy[col]
            node['children'].append(child_node)

    if not node['children'] and node['valid_moves']:
        random.shuffle(node['valid_moves'])

    return node
