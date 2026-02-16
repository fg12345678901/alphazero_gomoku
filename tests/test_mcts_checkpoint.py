from __future__ import annotations

import tempfile
import unittest
import os

import numpy as np
import torch

from gomoku.game import GomokuGame
from mcts.mcts import MCTS
from network.checkpoint import checkpoint_meta, load_checkpoint, save_checkpoint
from network.model import AlphaZeroNet


class MctsAndCheckpointTests(unittest.TestCase):
    def test_mcts_returns_valid_distribution(self):
        game = GomokuGame(size=5, n_in_row=3, history_steps=2)
        spec = game.getGameSpec()
        net = AlphaZeroNet(
            board_size=spec.board_size,
            action_size=spec.action_size,
            input_planes=spec.input_planes,
            channels=16,
            blocks=1,
        )
        net.eval()

        board = game.getInitBoard()
        mcts = MCTS(game, net, sims=4, cpuct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)
        probs = mcts.get_action_probs(board, temp=1.0, add_noise=True)

        self.assertEqual(probs.shape[0], spec.action_size)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0))

    def test_checkpoint_roundtrip(self):
        game = GomokuGame(size=5, n_in_row=3, history_steps=2)
        spec = game.getGameSpec()
        net = AlphaZeroNet(
            board_size=spec.board_size,
            action_size=spec.action_size,
            input_planes=spec.input_planes,
            channels=16,
            blocks=1,
        )

        meta = checkpoint_meta(spec, game_name=spec.name, channels=16, blocks=1)
        state = net.state_dict()

        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
            save_checkpoint(ckpt_path, state, meta)
            loaded_state, loaded_meta = load_checkpoint(ckpt_path, map_location="cpu")

        self.assertEqual(loaded_meta["game_name"], "gomoku")
        self.assertEqual(int(loaded_meta["board_size"]), 5)
        self.assertEqual(int(loaded_meta["action_size"]), 25)

        for key in state.keys():
            self.assertTrue(torch.equal(state[key], loaded_state[key]))


if __name__ == "__main__":
    unittest.main()
