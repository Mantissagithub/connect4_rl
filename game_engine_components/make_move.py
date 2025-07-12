from get_valid_moves import get_valid_moves

def make_move(board, col, player):
    if col < 0 or col >= len(board[0]):
        return False
    
    valid_moves = get_valid_moves(board)
    
    for row, valid_col in valid_moves:
        if valid_col == col:
            board[row][col] = player
            return True
    
    return False
