from get_valid_moves import get_valid_moves

def make__move(board, row, col, player):
    valid_moves = get_valid_moves(board)
    if [row, col] not in valid_moves:
        return False
    board[row][col] = player
    return True
