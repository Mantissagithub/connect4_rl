from game_engine_components.connect4_env import Connect4Env


def test_step_places_piece_and_switches_player():
    env = Connect4Env()
    _, done, winner, legal = env.step(3)
    assert legal is True
    assert done is False
    assert winner == 0
    assert env.board[5][3] == 1
    assert env.current_player == 2


def test_horizontal_win_detection():
    env = Connect4Env()
    env.board[5][0:4] = [1, 1, 1, 1]
    assert env.winner() == 1
    assert env.is_terminal() is True


def test_encode_state_uses_player_perspective():
    env = Connect4Env()
    env.board[5][0] = 1
    env.board[5][1] = 2
    encoded = env.encode_state(perspective_player=2)
    assert encoded.shape == (3, 6, 7)
    assert encoded[0][5][1].item() == 1.0
    assert encoded[1][5][0].item() == 1.0
    assert encoded[2][5][2].item() == 1.0
