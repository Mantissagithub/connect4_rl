def backpropagate_value(path, value):
    for node in reversed(path):
        node['total_value'] += value
        node['visits'] += 1
        value = -value
