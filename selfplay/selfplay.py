from __future__ import annotations

import logging
import os
import pickle
import time
from typing import List

import numpy as np
from tqdm import trange

from config import DATA_DIR, DEVICE, GAME_NAME, get_search_config
from games.base import GameLike
from games.registry import create_game, normalize_game_name
from mcts.mcts import MCTS
from network.checkpoint import (
    build_model_for_game,
    load_checkpoint,
    validate_checkpoint_meta,
)

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
        self.search_cfg = get_search_config(self.game_name)

        self.out_dir = out_dir or DATA_DIR
        os.makedirs(self.out_dir, exist_ok=True)

        self.net = build_model_for_game(self.game, device=DEVICE)
        if net_path and os.path.exists(net_path):
            state_dict, meta = load_checkpoint(net_path, map_location=DEVICE)
            validate_checkpoint_meta(meta, self.game.getGameSpec())
            self.net.load_state_dict(state_dict)
            logger.info("Loaded model %s", net_path)

        self.net.eval()
        self.mcts = self._create_mcts()
        self.num_games = num_games
        self.examples: List = []

    def _create_mcts(self) -> MCTS:
        return MCTS(
            self.game,
            self.net,
            sims=self.search_cfg.mcts_sims,
            cpuct=self.search_cfg.cpuct,
            dirichlet_alpha=self.search_cfg.dirichlet_alpha,
            dirichlet_eps=self.search_cfg.dirichlet_eps,
        )

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
        self.mcts = self._create_mcts()

        examples = []
        step = 0
        while True:
            temp = (
                self.search_cfg.selfplay_temperature
                if step < self.search_cfg.n_temp_moves
                else 0
            )

            pi = self.mcts.get_action_probs(board, temp=temp, add_noise=True)
            current_player = self.game.getCurrentPlayer(board)
            canonical = self.game.getCanonicalForm(board, current_player)
            examples.append((canonical, pi, current_player))

            move = int(np.random.choice(self.game.getActionSize(), p=pi))
            board, _ = self.game.getNextState(board, move)

            next_player = self.game.getCurrentPlayer(board)
            result = self.game.getGameEnded(board, next_player)
            if result != 0:
                final_examples = []
                for planes, pi_item, player in examples:
                    z = result if player == next_player else -result

                    for sym_planes, sym_pi in self.game.getSymmetries(planes, pi_item):
                        final_examples.append((sym_planes, sym_pi, z))

                        # Keep historical behavior: duplicate with sign-inverted turn plane.
                        planes_turn_flipped = sym_planes.copy()
                        planes_turn_flipped[-1] *= -1
                        final_examples.append((planes_turn_flipped, sym_pi, z))

                self.examples.extend(final_examples)
                return
            step += 1
