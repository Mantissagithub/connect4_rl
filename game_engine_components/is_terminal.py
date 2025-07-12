#to see whether the game has been ended or not
from .check_winner import check_winner
from .is_draw import is_draw

def is_the_end(board):
    if check_winner(board)!=0:
        return True
    else:
        #defining the draw condition
        if is_draw(board):
            return True
    return False
