from __future__ import annotations

import unittest

from config import get_rule_config
from games.registry import available_games, create_game
from gomoku.game import GomokuGame


class GomokuSpecTests(unittest.TestCase):
    def test_gomoku_spec_matches_rule_config(self):
        rules = get_rule_config("gomoku")
        game = GomokuGame(
            size=rules.board_size,
            n_in_row=rules.n_in_row or 5,
            history_steps=rules.history_steps,
        )

        spec = game.getGameSpec()
        self.assertEqual(spec.name, "gomoku")
        self.assertEqual(spec.board_size, rules.board_size)
        self.assertEqual(spec.action_size, rules.board_size * rules.board_size)
        self.assertEqual(spec.input_planes, rules.input_planes)

    def test_registry_and_current_player(self):
        self.assertIn("gomoku", available_games())
        self.assertIn("go", available_games())

        game = create_game("gomoku")
        board = game.getInitBoard()
        self.assertEqual(game.getCurrentPlayer(board), 1)

        board, _ = game.getNextState(board, 0)
        self.assertEqual(game.getCurrentPlayer(board), -1)


if __name__ == "__main__":
    unittest.main()
