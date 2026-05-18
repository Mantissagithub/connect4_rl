from .check_winner import check_winner
from .is_draw import is_draw

def is_the_end(board):
    return check_winner(board) != 0 or is_draw(board)
