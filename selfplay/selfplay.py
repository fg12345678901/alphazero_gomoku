# selfplay/selfplay.py
from __future__ import annotations

import logging
import os
import pickle
import time
from typing import List

import numpy as np
import torch
from tqdm import trange

from config import DATA_DIR, DEVICE, GAME_NAME, N_TEMP_MOVES, SELFPLAY_TEMPERATURE
from games.base import GameLike
from games.registry import create_game, normalize_game_name
from mcts.mcts import MCTS
from network.model import AlphaZeroNet

logger = logging.getLogger(__name__)


class SelfPlayWorker:
    def __init__(
        self,
        net_path: str | None,
        out_dir: str,
        num_games: int,
        game: GameLike | None = None,
        game_name: str = GAME_NAME,
    ):
        self.game_name = normalize_game_name(game_name)
        self.game = game if game is not None else create_game(self.game_name)
        self.out_dir = out_dir or DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        self.net = AlphaZeroNet().to(DEVICE)
        if net_path and os.path.exists(net_path):
            self.net.load_state_dict(torch.load(net_path, map_location=DEVICE))
            logger.info("Loaded model %s", net_path)
        self.net.eval()
        self.mcts = MCTS(self.game, self.net)
        self.num_games = num_games
        self.examples: List = []

    def run(self):
        for _ in trange(self.num_games, desc=f"Self-play[{self.game_name}]"):
            self.play_single()

        timestamp = f"{int(time.time() * 1000)}_{os.getpid()}"
        fname = os.path.join(self.out_dir, f"selfplay_{timestamp}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(self.examples, f)
        logger.info("Saved %d examples to %s", len(self.examples), fname)

    def play_single(self):
        board = self.game.getInitBoard()
        # Use a fresh tree for each game to avoid cross-game search contamination.
        self.mcts = MCTS(self.game, self.net)

        examples = []
        step = 0
        while True:
            temp = SELFPLAY_TEMPERATURE if step < N_TEMP_MOVES else 0
            # AlphaZero self-play uses root Dirichlet noise to encourage exploration.
            pi = self.mcts.get_action_probs(board, temp=temp, add_noise=True)
            canonical = self.game.getCanonicalForm(board, board.current_player)
            examples.append((canonical, pi, board.current_player))

            move = np.random.choice(self.game.getActionSize(), p=pi)
            board, _ = self.game.getNextState(board, move)

            result = self.game.getGameEnded(board, board.current_player)
            if result != 0:
                final_examples = []
                for planes, pi, player in examples:
                    z = result if player == board.current_player else -result

                    for sym_planes, sym_pi in self.game.getSymmetries(planes, pi):
                        final_examples.append((sym_planes, sym_pi, z))

                        # Keep the historical behavior of duplicating turn-plane inversion.
                        planes_turn_flipped = sym_planes.copy()
                        planes_turn_flipped[-1] *= -1
                        final_examples.append((planes_turn_flipped, sym_pi, z))

                self.examples.extend(final_examples)
                return
            step += 1
