from .check_winner import check_winner


def is_draw(board):
    return check_winner(board) == 0 and all(board[0][col] != 0 for col in range(len(board[0])))
