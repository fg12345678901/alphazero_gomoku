from __future__ import annotations

from typing import List, Tuple

import numpy as np

from games.base import GameSpec
from selfplay.augment import rotate_flip

from .board import Board


class GomokuGame:
    def __init__(
        self,
        size: int = 15,
        n_in_row: int = 5,
        history_steps: int = 3,
    ):
        self.size = size
        self.n_in_row = n_in_row
        self.history_steps = history_steps
        self.input_planes = 2 * history_steps + 1
        self._spec = GameSpec(
            name="gomoku",
            board_size=self.size,
            action_size=self.size * self.size,
            input_planes=self.input_planes,
        )

    def getGameSpec(self) -> GameSpec:
        return self._spec

    def getInitBoard(self) -> Board:
        return Board(self.size, self.n_in_row)

    def getBoardSize(self) -> Tuple[int, int]:
        return (self.size, self.size)

    def getActionSize(self) -> int:
        return self._spec.action_size

    def getCurrentPlayer(self, board: Board) -> int:
        return int(board.current_player)

    def getNextState(self, board: Board, action: int) -> Tuple[Board, int]:
        b = board.copy()
        b.do_move(action)
        return b, b.current_player

    def getValidMoves(self, board: Board) -> np.ndarray:
        moves = np.zeros(self.getActionSize(), dtype=np.int8)
        moves[board.legal_moves()] = 1
        return moves

    def getGameEnded(self, board: Board, player: int) -> float:
        winner = board.get_winner()
        if winner is None:
            return 0.0
        if winner == 0:
            return 1e-4
        return 1.0 if winner == player else -1.0

    def getCanonicalForm(self, board: Board, player: int) -> np.ndarray:
        planes = np.zeros((self.input_planes, self.size, self.size), dtype=np.float32)

        state = board.board.copy()
        history = board.move_history
        for i in range(self.history_steps):
            planes[i] = (state == player).astype(np.float32)
            planes[i + self.history_steps] = (state == -player).astype(np.float32)
            if i < len(history):
                move = history[-1 - i]
                x, y = board.move_to_coord(move)
                state[x, y] = 0
            else:
                state.fill(0)

        planes[-1].fill(float(player))
        return planes

    def getSymmetries(
        self,
        board_planes: np.ndarray,
        pi: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        size = self.size
        pi_board = pi.reshape((size, size))
        syms = []
        for k in range(4):
            for flip in (False, True):
                new_planes = rotate_flip(board_planes, k, flip)
                new_pi = rotate_flip(pi_board, k, flip).ravel()
                syms.append((new_planes, new_pi))
        return syms

    def stringRepresentation(self, board: Board) -> bytes:
        player_byte = bytes([1 if board.current_player == 1 else 0])
        return board.board.tobytes() + player_byte
