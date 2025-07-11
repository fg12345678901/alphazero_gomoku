# gomoku/board.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class Board:
    """五子棋棋盘状态"""
    def __init__(self, size: int = 15, n_in_row: int = 5):
        self.size = size
        self.n_in_row = n_in_row
        self.reset()

    # ---------- 基本操作 ---------- #
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # 0 空, 1 黑, -1 白
        self.current_player = 1
        self.move_history: List[int] = []

    def copy(self) -> "Board":
        b = Board(self.size, self.n_in_row)
        b.board = self.board.copy()
        b.current_player = self.current_player
        b.move_history = self.move_history.copy()
        return b

    def move_to_coord(self, m: int) -> Tuple[int, int]:
        return divmod(m, self.size)

    def coord_to_move(self, x: int, y: int) -> int:
        return x * self.size + y

    def legal_moves(self) -> List[int]:
        return list(np.where(self.board.ravel() == 0)[0])

    def do_move(self, move: int):
        x, y = self.move_to_coord(move)
        assert self.board[x, y] == 0, "非法落子"
        self.board[x, y] = self.current_player
        self.move_history.append(move)
        self.current_player *= -1

    def undo_move(self) -> int | None:
        """悔棋：撤销最后一步落子，返回该手。"""
        if not self.move_history:
            return None
        last = self.move_history.pop()
        x, y = self.move_to_coord(last)
        self.board[x, y] = 0
        self.current_player *= -1
        return last

    # ---------- 终局判定 ---------- #
    def _check_dir(self, x, y, dx, dy, player) -> bool:
        """检查方向 (dx,dy) 上是否有 n 连"""
        cnt = 0
        i, j = x, y
        while 0 <= i < self.size and 0 <= j < self.size and self.board[i, j] == player:
            cnt += 1
            i += dx
            j += dy
        i, j = x - dx, y - dy
        while 0 <= i < self.size and 0 <= j < self.size and self.board[i, j] == player:
            cnt += 1
            i -= dx
            j -= dy
        return cnt >= self.n_in_row

    def get_winner(self) -> int | None:
        if not self.move_history:
            return None
        last = self.move_history[-1]
        x, y = self.move_to_coord(last)
        player = -self.current_player  # 上一步落子者
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dx, dy in directions:
            if self._check_dir(x, y, dx, dy, player):
                return player
        if len(self.move_history) == self.size * self.size:
            return 0  # 平局
        return None

    # ---------- 渲染 ---------- #
    def __str__(self):
        stone = {1: '●', -1: '○', 0: '·'}
        rows = []
        for i in range(self.size):
            rows.append(' '.join(stone[int(s)] for s in self.board[i]))
        return '\n'.join(rows)
