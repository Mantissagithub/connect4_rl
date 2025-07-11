def get_valid_moves(board):
    played_cols = []
    for col in range(len(board[0])):
        for row in range(len(board)):
            if(board[row][col] == 1 or board[row][col] == 2):
                played_cols.append(col)
                break
    
    valid_moves = []
    for col in range(len(board[0])):
        for row in range(len(board)):
            if col in played_cols and board[row][col] == 0:
                valid_moves.append([row, col])

    return valid_moves