def get_valid_moves(board):
    valid_moves = []
    
    for col in range(len(board[0])):
        for row in range(len(board)):
            if board[row][col] != 0:
                if row == 0:
                    break
                else:
                    valid_moves.append([row-1, col])
                    break
        else:
            valid_moves.append([len(board)-1, col])
    
    return valid_moves
