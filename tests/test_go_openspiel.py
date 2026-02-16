from __future__ import annotations

import numpy as np
import unittest


try:
    from games import go_openspiel
except Exception:  # pragma: no cover
    go_openspiel = None


GoOpenSpielGame = getattr(go_openspiel, "GoOpenSpielGame", None)
HAS_PYSPIEL = bool(go_openspiel is not None and getattr(go_openspiel, "pyspiel", None) is not None)


@unittest.skipUnless(HAS_PYSPIEL, "OpenSpiel not available")
class GoOpenSpielAdapterTests(unittest.TestCase):
    def test_go_spec_and_canonical_shape(self):
        game = GoOpenSpielGame(board_size=9, komi=7.5, history_steps=8)
        spec = game.getGameSpec()

        self.assertEqual(spec.name, "go")
        self.assertEqual(spec.board_size, 9)
        self.assertEqual(spec.action_size, 9 * 9 + 1)

        board = game.getInitBoard()
        player = game.getCurrentPlayer(board)
        planes = game.getCanonicalForm(board, player)
        self.assertEqual(planes.shape[0], spec.input_planes)
        self.assertEqual(planes.shape[1], 9)
        self.assertEqual(planes.shape[2], 9)

    def test_pass_action_and_terminal(self):
        game = GoOpenSpielGame(board_size=9, komi=7.5, history_steps=8)
        board = game.getInitBoard()
        valids = game.getValidMoves(board)

        self.assertEqual(valids.shape[0], game.getActionSize())
        self.assertEqual(int(valids[game.pass_action]), 1)

        board, _ = game.getNextState(board, game.pass_action)
        board, _ = game.getNextState(board, game.pass_action)
        result = game.getGameEnded(board, 1)
        self.assertNotEqual(result, 0.0)

    def test_symmetry_keeps_pass_probability(self):
        game = GoOpenSpielGame(board_size=9, komi=7.5, history_steps=8)
        board = game.getInitBoard()
        player = game.getCurrentPlayer(board)
        planes = game.getCanonicalForm(board, player)

        pi = np.zeros(game.getActionSize(), dtype=np.float32)
        pi[0] = 0.6
        pi[10] = 0.3
        pi[game.pass_action] = 0.1

        for _, sym_pi in game.getSymmetries(planes, pi):
            self.assertAlmostEqual(float(sym_pi[game.pass_action]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
