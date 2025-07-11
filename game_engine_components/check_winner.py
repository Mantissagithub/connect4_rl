def check_winner(board):
    n = len(board)
    m = len(board[0])

    #horizonatal checking like - 
    # 2 2 2 2
    # 1 1 1 1
    # either of these
    for i in range(n):
        for j in range(m):
            if j + 3 < m and board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3] != 0:
                return board[i][j]
            
    #vertical checking like -
    # 2   1
    # 2   1
    # 2   1
    # 2   1
    for i in range(n):
        for j in range(m):
            if i + 3 < n and board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j] != 0:
                return board[i][j]
            
    #diagonal cehcking like - 
    #2
    # 2
    #  2
    #   2
    #i think you undestand by now
    for i in range(n):
        for j in range(m):
            if i + 3 < n and j + 3 < m and board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3] != 0:
                return board[i][j]
            if i - 3 >= 0 and j + 3 < m and board[i][j] == board[i-1][j+1] == board[i-2][j+2] == board[i-3][j+3] != 0:
                return board[i][j]