# trainer/arena.py
from __future__ import annotations

import numpy as np
from tqdm import trange

from games.base import GameLike
from mcts.mcts import MCTS
from network.model import AlphaZeroNet


class Arena:
    def __init__(
        self,
        game: GameLike,
        net1: AlphaZeroNet,
        net2: AlphaZeroNet,
        games: int,
        show_progress: bool = True,
    ):
        self.game = game
        self.net1 = net1
        self.net2 = net2
        self.games = games
        self.show_progress = show_progress

    def play(self):
        n1_win = n2_win = draws = 0
        for i in trange(self.games, desc="Arena", disable=not self.show_progress):
            if i % 2 == 0:
                w = self._play_single(self.net1, self.net2)
            else:
                w = self._play_single(self.net2, self.net1)
                if w != 0:
                    w = -w
            if w == 1:
                n1_win += 1
            elif w == -1:
                n2_win += 1
            else:
                draws += 1
        return n1_win, n2_win, draws

    def _play_single(self, net_black, net_white):
        board = self.game.getInitBoard()
        mcts_b = MCTS(self.game, net_black)
        mcts_w = MCTS(self.game, net_white)
        while True:
            mcts = mcts_b if board.current_player == 1 else mcts_w
            # AlphaZero evaluation is greedy with respect to root visit counts.
            pi = mcts.get_action_probs(board, temp=0, add_noise=False)
            move = int(np.argmax(pi))
            board, _ = self.game.getNextState(board, move)

            # Result normalized to black's perspective.
            result_for_black = self.game.getGameEnded(board, 1)
            if result_for_black != 0:
                if abs(result_for_black) < 1e-3:
                    return 0
                return 1 if result_for_black > 0 else -1
