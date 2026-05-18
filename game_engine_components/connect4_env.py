from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


BOARD_ROWS = 6
BOARD_COLS = 7


def _copy_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


@dataclass
class Connect4Env:
    board: List[List[int]] = field(
        default_factory=lambda: [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    )
    current_player: int = 1

    def reset(self) -> List[List[int]]:
        self.board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.current_player = 1
        return _copy_board(self.board)

    def clone(self) -> "Connect4Env":
        return Connect4Env(board=_copy_board(self.board), current_player=self.current_player)

    @classmethod
    def from_board(cls, board: List[List[int]], current_player: Optional[int] = None) -> "Connect4Env":
        return cls(board=_copy_board(board), current_player=current_player or cls.infer_current_player(board))

    @staticmethod
    def infer_current_player(board: List[List[int]]) -> int:
        player1_count = sum(cell == 1 for row in board for cell in row)
        player2_count = sum(cell == 2 for row in board for cell in row)
        return 1 if player1_count <= player2_count else 2

    def legal_actions(self) -> List[int]:
        return [col for col in range(BOARD_COLS) if self.board[0][col] == 0]

    def step(self, action: int) -> Tuple[List[List[int]], bool, int, bool]:
        if action < 0 or action >= BOARD_COLS:
            return _copy_board(self.board), self.is_terminal(), self.winner(), False

        for row in range(BOARD_ROWS - 1, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                winner = self.winner()
                done = winner != 0 or self.is_draw()
                self.current_player = 3 - self.current_player
                return _copy_board(self.board), done, winner, True

        return _copy_board(self.board), self.is_terminal(), self.winner(), False

    def is_draw(self) -> bool:
        return all(self.board[0][col] != 0 for col in range(BOARD_COLS)) and self.winner() == 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or self.is_draw()

    def winner(self) -> int:
        return self.check_winner(self.board)

    def encode_state(self, perspective_player: Optional[int] = None) -> torch.Tensor:
        import torch

        player = perspective_player or self.current_player
        opponent = 3 - player
        board_tensor = torch.tensor(self.board, dtype=torch.float32)
        current_channel = (board_tensor == player).float()
        opponent_channel = (board_tensor == opponent).float()
        empty_channel = (board_tensor == 0).float()
        return torch.stack([current_channel, opponent_channel, empty_channel], dim=0)

    @staticmethod
    def check_winner(board: List[List[int]]) -> int:
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                token = board[row][col]
                if token != 0 and all(board[row][col + offset] == token for offset in range(4)):
                    return token

        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS):
                token = board[row][col]
                if token != 0 and all(board[row + offset][col] == token for offset in range(4)):
                    return token

        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS - 3):
                token = board[row][col]
                if token != 0 and all(board[row + offset][col + offset] == token for offset in range(4)):
                    return token

        for row in range(3, BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                token = board[row][col]
                if token != 0 and all(board[row - offset][col + offset] == token for offset in range(4)):
                    return token

        return 0
