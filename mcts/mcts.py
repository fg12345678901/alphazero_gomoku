# mcts/mcts.py
from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict
from config import (MCTS_SIMS, CPUCT, DEVICE,
                    DIRICHLET_ALPHA, DIRICHLET_EPS)
from network.model import AlphaZeroNet

class TreeNode:
    __slots__ = ("P","N","W","Q","children")
    def __init__(self, P):
        self.P = P        # 先验概率
        self.N = 0        # 访问次数
        self.W = 0.0      # 累积价值
        self.Q = 0.0      # 平均价值
        self.children: Dict[int, TreeNode] = {}

class MCTS:
    def __init__(self, game, net: AlphaZeroNet):
        self.game = game
        self.net  = net
        self.Qsa   = {}  # (s,a) -> Q
        self.Nsa   = {}  # (s,a) -> N
        self.Ns    = {}  # s -> N
        self.Ps    = {}  # s -> policy vector
        self.Es    = {}  # s -> gameEnded
        self.Vs    = {}  # s -> valid moves mask
        self.W     = {}  # (s,a) -> 累积价值

    # ----------------------- 公共接口 ----------------------- #
    def get_action_probs(self, board, temp=1.0, add_noise: bool = False):
        """
        返回当前棋盘的概率分布 π
        add_noise=True 时（自我对弈）在根节点混 Dirichlet 噪声
        """

        s_root = self.game.stringRepresentation(board)

        # ------------- 如需探索噪声，先保证根节点已扩展 -------------
        if add_noise:
            if s_root not in self.Ps:
                # 先扩展一次，得到 Ps 与 Vs
                self.search(board)
            self._add_dirichlet_noise(s_root)

        # ------------ 正常的蒙特卡洛树搜索 -------------
        for _ in range(MCTS_SIMS):
            self.search(board)

        s = self.game.stringRepresentation(board)
        counts = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for a in range(self.game.getActionSize()):
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]

        if temp == 0:
            best_as = np.argwhere(counts == np.max(counts)).flatten()
            probs = np.zeros_like(counts)
            probs[np.random.choice(best_as)] = 1.0
            return probs

        counts = counts ** (1. / temp)
        probs = counts / np.sum(counts)
        return probs

    # ----------------------- 树搜索 ----------------------- #
    def search(self, board):
        s = self.game.stringRepresentation(board)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, board.current_player)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            # 神经网络扩展
            canonical = self.game.getCanonicalForm(board, board.current_player)
            canonical = torch.tensor(canonical, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                policy, value = self.net(canonical)
            policy = torch.softmax(policy, dim=1).cpu().numpy()[0]

            valids = self.game.getValidMoves(board)
            policy = policy * valids  # mask 非法
            sum_p = np.sum(policy)
            if sum_p > 0:
                policy /= sum_p
            else:  # 所有合法位被网络评为0, 均匀分布
                policy = policy + valids
                policy /= np.sum(policy)

            self.Ps[s] = policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -value.item()

        valids = self.Vs[s]
        best_ucb, best_a = -float("inf"), -1
        # UCB 选子
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            if (s, a) in self.Qsa:
                ucb = self.Qsa[(s, a)] + CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                ucb = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a

        a = best_a
        next_board, _ = self.game.getNextState(board, a)
        v = self.search(next_board)

        # 反向回传
        if (s, a) in self.Qsa:
            self.W[(s, a)] += v
            self.Nsa[(s, a)] += 1
            self.Qsa[(s, a)] = self.W[(s, a)] / self.Nsa[(s, a)]
        else:
            self.W[(s, a)] = v
            self.Nsa[(s, a)] = 1
            self.Qsa[(s, a)] = v
        self.Ns[s] += 1
        return -v


    # ---------- 私有：对根节点注入 Dirichlet 噪声 ----------
    def _add_dirichlet_noise(self, s):
        """将噪声混入 self.Ps[s]（只对合法着法混合）"""
        valids = self.Vs[s]
        legal_idx = np.flatnonzero(valids)
        if len(legal_idx) == 0:
            return  # 理论不会发生
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_idx))
        # 按照 AlphaZero 公式线性插值
        self.Ps[s][legal_idx] = (
            (1 - DIRICHLET_EPS) * self.Ps[s][legal_idx] +
            DIRICHLET_EPS * noise
        )

        # print("dir_noise added, sum(Ps)=", self.Ps[s].sum())

