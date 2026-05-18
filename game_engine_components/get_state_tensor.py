from .connect4_env import Connect4Env


def convert_into_tensor(board, current_player=None):
    env = Connect4Env.from_board(board, current_player=current_player)
    return env.encode_state(current_player or env.current_player)
