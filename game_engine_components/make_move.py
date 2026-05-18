def make_move(board, col, player):
    if col < 0 or col >= len(board[0]):
        return False

    for row in range(len(board) - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return True
    return False 
