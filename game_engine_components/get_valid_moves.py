def get_valid_moves(board):
    return [col for col in range(len(board[0])) if board[0][col] == 0]
