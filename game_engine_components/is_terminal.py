#to see whether the game has been ended or not
from .check_winner import check_winner
from .is_draw import is_draw

def is_the_end(board):
    winner = check_winner(board)
    is_draw_result = is_draw(board)
    
    # print(f"Debug - Winner: {winner}, Is draw: {is_draw_result}")?
    
    if winner != 0:
        return True
    
    if is_draw_result:
        return True
    
    return False
