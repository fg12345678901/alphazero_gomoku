from __future__ import annotations

import math

import numpy as np
import torch

from config import CPUCT, DEVICE, DIRICHLET_ALPHA, DIRICHLET_EPS, MCTS_SIMS
from network.model import AlphaZeroNet


class MCTS:
    def __init__(
        self,
        game,
        net: AlphaZeroNet,
        sims: int = MCTS_SIMS,
        cpuct: float = CPUCT,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_eps: float = DIRICHLET_EPS,
    ):
        self.game = game
        self.net = net
        self.sims = sims
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.Qsa = {}  # (s, a) -> Q
        self.Nsa = {}  # (s, a) -> N
        self.Ns = {}  # s -> N
        self.Ps = {}  # s -> policy
        self.Es = {}  # s -> gameEnded
        self.Vs = {}  # s -> valid moves
        self.W = {}  # (s, a) -> total value

    def get_action_probs(self, board, temp: float = 1.0, add_noise: bool = False) -> np.ndarray:
        s_root = self.game.stringRepresentation(board)

        if add_noise:
            if s_root not in self.Ps:
                self.search(board)
            self._add_dirichlet_noise(s_root)

        for _ in range(self.sims):
            self.search(board)

        s = self.game.stringRepresentation(board)
        action_size = self.game.getActionSize()
        counts = np.zeros(action_size, dtype=np.float32)
        for a in range(action_size):
            if (s, a) in self.Nsa:
                counts[a] = self.Nsa[(s, a)]

        valids = self.Vs.get(s)
        if valids is None:
            valids = self.game.getValidMoves(board).astype(np.float32)

        if temp == 0:
            max_count = np.max(counts)
            if max_count <= 0:
                legal = np.flatnonzero(valids)
                probs = np.zeros_like(counts)
                probs[np.random.choice(legal)] = 1.0
                return probs

            best_as = np.flatnonzero(counts == max_count)
            probs = np.zeros_like(counts)
            probs[np.random.choice(best_as)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        total = float(np.sum(counts))
        if total <= 0:
            legal_sum = float(np.sum(valids))
            if legal_sum <= 0:
                return np.full(action_size, 1.0 / action_size, dtype=np.float32)
            return (valids / legal_sum).astype(np.float32)

        return (counts / total).astype(np.float32)

    def search(self, board):
        s = self.game.stringRepresentation(board)
        player = self.game.getCurrentPlayer(board)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, player)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            canonical = self.game.getCanonicalForm(board, player)
            x = torch.tensor(canonical, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.net(x)

            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            valids = self.game.getValidMoves(board).astype(np.float32)
            policy = policy * valids
            sum_p = float(np.sum(policy))
            if sum_p > 0:
                policy /= sum_p
            else:
                legal_sum = float(np.sum(valids))
                if legal_sum > 0:
                    policy = valids / legal_sum
                else:
                    policy = np.full(self.game.getActionSize(), 1.0 / self.game.getActionSize(), dtype=np.float32)

            self.Ps[s] = policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -float(value.item())

        valids = self.Vs[s]
        best_ucb = -float("inf")
        best_a = -1
        action_size = self.game.getActionSize()

        for a in range(action_size):
            if valids[a] == 0:
                continue

            if (s, a) in self.Qsa:
                ucb = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                ucb = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a

        if best_a < 0:
            return 0.0

        next_board, _ = self.game.getNextState(board, best_a)
        v = self.search(next_board)

        if (s, best_a) in self.Qsa:
            self.W[(s, best_a)] += v
            self.Nsa[(s, best_a)] += 1
            self.Qsa[(s, best_a)] = self.W[(s, best_a)] / self.Nsa[(s, best_a)]
        else:
            self.W[(s, best_a)] = v
            self.Nsa[(s, best_a)] = 1
            self.Qsa[(s, best_a)] = v

        self.Ns[s] += 1
        return -v

    def _add_dirichlet_noise(self, s) -> None:
        valids = self.Vs[s]
        legal_idx = np.flatnonzero(valids)
        if len(legal_idx) == 0:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_idx))
        self.Ps[s][legal_idx] = (
            (1.0 - self.dirichlet_eps) * self.Ps[s][legal_idx]
            + self.dirichlet_eps * noise
        )
