#getting the action visits from the mcts tree

def get_action_visits(node):
    action_visits = {}
    
    def traverse(node):
        if node is None:
            return
        
        for child in node['children']:
            action = child['action']
            action_visits[action] = action_visits.get(action, 0) + child['visits']
            traverse(child)
    
    traverse(node)
    
    return action_visits