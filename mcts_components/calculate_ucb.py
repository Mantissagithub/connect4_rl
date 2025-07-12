# # so the formula for ucb(simply the upper confdence bound, saying the most confident node is the best) from the paper alphago is:
# # UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

# # q(s, a)-> the score of the action a in state s
# # p(s, a)-> the prior probability of action a in state s
# # n(s)-> the number of visits to state s
# # n(s, a)-> the number of visits to action a in state s
# # c_puct-> a constant that balances exploration and exploitation

# from math import sqrt

# def calculate_ucb(q, p, n_s, n_sa, c_puct):
#     return q + c_puct * p * sqrt(n_s) / (1 + n_sa)

# this above thing from the alphago paper, for now will just keep the function more simpler

from math import log, sqrt
    
# def calculate_ucb(node_value, parent_visits, child_visits, c_puct=1.4):
#     return node_value + c_puct * sqrt(log(parent_visits + 1) / (child_visits + 1))

#now implementing the ucb function as per the alphago paper, so the formula is:

def calculate_ucb(q, p, n, n_a, c_puct=1.4):
    if n_a == 0:
        return float('inf')
    
    exploitation = q
    exploration = c_puct * p * sqrt(n) / (1 + n_a)

    return exploitation + exploration

def calculate_ucb_fallback(node_value, parent_visits, child_visits, c_puct=1.4):
    return node_value + c_puct * sqrt(log(parent_visits + 1) / (child_visits + 1))
