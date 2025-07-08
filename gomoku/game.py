# gomoku/game.py
import numpy as np
from typing import Tuple, List
from .board import Board
from selfplay.augment import rotate_flip

class GomokuGame:
    """
    AlphaZero 接口封装:
      • getInitBoard        -> numpy 状态(3×S×S)
      • getNextState, getValidMoves, getGameEnded, getCanonicalForm
      • stringRepresentation(用于哈希)
    """
    def __init__(self, size=15, n_in_row=5):
        self.size = size
        self.n_in_row = n_in_row

    # ---------- 接口 ---------- #
    def getInitBoard(self) -> Board:
        return Board(self.size, self.n_in_row)

    def getBoardSize(self) -> Tuple[int, int]:
        return (self.size, self.size)

    def getActionSize(self) -> int:
        return self.size * self.size

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
            return 0
        elif winner == 0:
            return 1e-4  # 平局略大于0 防梯度消失
        elif winner == player:
            return 1
        else:
            return -1

    def getCanonicalForm(self, board: Board, player: int) -> np.ndarray:
        """返回以 `player` 视角的 3×S×S 特征平面"""
        plane_current = (board.board == player).astype(np.float32)
        plane_opp     = (board.board == -player).astype(np.float32)
        plane_turn    = np.full_like(plane_current, player, dtype=np.float32)  # or 1 channel of 1s
        return np.stack([plane_current, plane_opp, plane_turn], axis=0)

    def getSymmetries(self, board_planes: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """八向对称增强"""
        size = self.size
        pi_board = pi.reshape((size, size))
        syms = []
        for k in range(4):
            for flip in [False, True]:
                new_planes = rotate_flip(board_planes, k, flip)
                new_pi = rotate_flip(pi_board, k, flip).ravel()
                syms.append((new_planes, new_pi))
        return syms

    def stringRepresentation(self, board: Board):
        return board.board.tobytes()
