def check_winner(board):
    n = len(board)
    m = len(board[0])

    #horizonatal checking like - 
    # 2 2 2 2
    # 1 1 1 1
    # either of these
    for i in range(n):
        for j in range(m - 3):
            if (board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] and 
                board[i][j] != 0):
                return board[i][j]
    #vertical checking like -
    # 2   1
    # 2   1
    # 2   1
    # 2   1
    for i in range(n - 3):
        for j in range(m):
            if (board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] and 
                board[i][j] != 0):
                return board[i][j]
            
    #diagonal cehcking like - 
    #2
    # 2
    #  2
    #   2
    #i think you undestand by now
    for i in range(n - 3):
        for j in range(m - 3):
            if (board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] and 
                board[i][j] != 0):
                return board[i][j]
            
    for i in range(3, n):
        for j in range(m - 3):
            if (board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] and 
                board[i][j] != 0):
                return board[i][j]
            
    return 0 #no winners