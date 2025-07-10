# trainer/arena.py
from __future__ import annotations
import numpy as np
from tqdm import trange
from gomoku.game import GomokuGame
from network.model import AlphaZeroNet
from mcts.mcts import MCTS
from config import BOARD_SIZE, N_IN_ROW, DEVICE, MCTS_SIMS, N_TEMP_MOVES

class Arena:
    def __init__(self, net1: AlphaZeroNet, net2: AlphaZeroNet, games:int):
        self.game  = GomokuGame(BOARD_SIZE, N_IN_ROW)
        self.net1  = net1
        self.net2  = net2
        self.games = games

    def play(self):
        n1_win = n2_win = draws = 0
        for i in trange(self.games, desc="Arena"):
            if i % 2 == 0:
                w = self._play_single(self.net1, self.net2)
            else:
                w = self._play_single(self.net2, self.net1)
                if w != 0:
                    w = -w
            if w == 1:   n1_win += 1
            elif w == -1: n2_win += 1
            else:        draws += 1
        return n1_win, n2_win, draws

    def _play_single(self, net_black, net_white):
        board = self.game.getInitBoard()
        mcts_b = MCTS(self.game, net_black)
        mcts_w = MCTS(self.game, net_white)
        step = 0
        while True:
            mcts = mcts_b if board.current_player == 1 else mcts_w
            # ---------- 加随机性 (首步加入噪声，前 N_TEMP_MOVES 的 temp 为 1) ----------
            add_noise = (step == 0)                 # 仅首手注入 Dirichlet
            temp      = 1 if step < N_TEMP_MOVES else 0        # 前 N_TEMP_MOVES 手用采样，可自行调
            pi = mcts.get_action_probs(board, temp=temp, add_noise=add_noise)
            move = np.random.choice(len(pi), p=pi)  # 按概率采样
            board, _ = self.game.getNextState(board, move)
            # from gomoku.display import display
            # display(board)

            # result = self.game.getGameEnded(board, board.current_player)
            # if result != 0:
            #     return result
            winner = board.get_winner()  # 1=黑胜, -1=白胜, 0=未分，None=继续
            # print(winner,step)
            if winner is not None:
                if winner == 0:
                    return 0
                return 1 if winner == 1 else -1
            step += 1
