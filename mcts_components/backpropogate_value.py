# here we are gonna just reached the terminal so just backpop the value

def backpropagate_value(node, value):
    while node is not None:
        node['total_value'] += value
        node['visits'] += 1
        node = node['parent']