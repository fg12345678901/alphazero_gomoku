from __future__ import annotations

from typing import List, Tuple

import numpy as np

from games.base import GameSpec
from selfplay.augment import rotate_flip

try:
    import pyspiel
except ImportError as exc:  # pragma: no cover - import guarded by runtime usage
    pyspiel = None
    _PYSPIEL_IMPORT_ERROR = exc
else:
    _PYSPIEL_IMPORT_ERROR = None


class GoOpenSpielGame:
    """
    Go adapter backed by OpenSpiel's official `go` environment.

    Notes:
    - Action space includes PASS as the final action index.
    - Canonical features are taken from OpenSpiel observation tensors.
    """

    def __init__(
        self,
        board_size: int = 9,
        komi: float = 7.5,
        history_steps: int = 8,
    ):
        if pyspiel is None:
            raise ImportError(
                "OpenSpiel is required for --game go. Install with `pip install open-spiel` "
                "(Windows may also need CMake + a C++17 compiler; Linux is recommended)."
            ) from _PYSPIEL_IMPORT_ERROR

        self.board_size = board_size
        self.komi = komi
        self.history_steps = history_steps  # kept for config symmetry with Gomoku

        game_def = f"go(board_size={board_size},komi={komi})"
        self._engine = pyspiel.load_game(game_def)
        if self._engine.num_players() != 2:
            raise ValueError("GoOpenSpielGame only supports 2-player games")

        self._obs_shape = tuple(self._engine.observation_tensor_shape())
        self.input_planes = 2 * self.history_steps + 1

        action_size = int(self._engine.num_distinct_actions())
        self.pass_action = action_size - 1
        self._spec = GameSpec(
            name="go",
            board_size=self.board_size,
            action_size=action_size,
            input_planes=self.input_planes,
        )

    def getGameSpec(self) -> GameSpec:
        return self._spec

    def getInitBoard(self):
        return self._engine.new_initial_state()

    def getBoardSize(self) -> Tuple[int, int]:
        return (self.board_size, self.board_size)

    def getActionSize(self) -> int:
        return self._spec.action_size

    def getCurrentPlayer(self, board) -> int:
        current = int(board.current_player())
        if current == 0:
            return 1
        if current == 1:
            return -1

        # OpenSpiel uses TERMINAL sentinel when game ended.
        if board.is_terminal():
            return 1 if (len(board.history()) % 2 == 0) else -1

        raise RuntimeError(f"Unexpected current_player from OpenSpiel state: {current}")

    def getNextState(self, board, action: int):
        next_state = board.clone()
        next_state.apply_action(int(action))
        return next_state, self.getCurrentPlayer(next_state)

    def getValidMoves(self, board) -> np.ndarray:
        moves = np.zeros(self.getActionSize(), dtype=np.int8)
        legal = board.legal_actions()
        moves[np.asarray(legal, dtype=np.int32)] = 1
        return moves

    def getGameEnded(self, board, player: int) -> float:
        if not board.is_terminal():
            return 0.0

        returns = board.returns()
        black_return = float(returns[0])
        white_return = float(returns[1])

        if abs(black_return - white_return) < 1e-8:
            result_for_black = 1e-4
        else:
            result_for_black = 1.0 if black_return > white_return else -1.0

        return result_for_black if player == 1 else -result_for_black

    def getCanonicalForm(self, board, player: int) -> np.ndarray:
        return self._canonical_planes(board, player=player)

    def getSymmetries(
        self,
        board_planes: np.ndarray,
        pi: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        size = self.board_size
        board_policy = pi[: size * size].reshape(size, size)
        pass_prob = float(pi[self.pass_action])

        syms: List[Tuple[np.ndarray, np.ndarray]] = []
        for k in range(4):
            for flip in (False, True):
                new_planes = rotate_flip(board_planes, k, flip)
                new_pi_board = rotate_flip(board_policy, k, flip).ravel()
                new_pi = np.concatenate(
                    [new_pi_board, np.asarray([pass_prob], dtype=np.float32)],
                    axis=0,
                )
                syms.append((new_planes, new_pi))
        return syms

    def stringRepresentation(self, board) -> bytes:
        if hasattr(board, "serialize"):
            serialized = board.serialize()
            if isinstance(serialized, bytes):
                return serialized
            return str(serialized).encode("utf-8")

        history = np.asarray(board.history(), dtype=np.int32).tobytes()
        player = int(board.current_player()).to_bytes(2, byteorder="little", signed=True)
        return history + player

    def _canonical_planes(self, board, player: int) -> np.ndarray:
        size = self.board_size
        h = self.history_steps
        planes = np.zeros((2 * h + 1, size, size), dtype=np.float32)

        work = board.clone()
        move_count = len(board.history())
        exhausted = False

        for i in range(h):
            if not exhausted:
                black, white = self._extract_color_planes(work)
                if player == 1:
                    own, opp = black, white
                elif player == -1:
                    own, opp = white, black
                else:
                    raise ValueError(f"GoOpenSpielGame expects player in {{1, -1}}, got {player}")

                planes[i] = own
                planes[i + h] = opp

            if exhausted:
                continue

            # Keep behavior consistent with Gomoku adapter:
            # include the initial board once, then pad older history with zeros.
            if i < move_count:
                # PySpiel bindings on Windows require two positional placeholders.
                work.undo_action(0, 0)
            else:
                exhausted = True

        planes[-1].fill(float(player))
        return planes

    def _extract_color_planes(self, board) -> Tuple[np.ndarray, np.ndarray]:
        # OpenSpiel Go observation tensor encodes absolute board colors:
        # [black, white, empty, white_to_play].
        obs_flat = np.asarray(board.observation_tensor(0), dtype=np.float32)
        obs = self._reshape_observation(obs_flat)
        if obs.shape[0] < 2:
            raise ValueError(f"Unexpected Go observation channels: {obs.shape}")
        black = obs[0].astype(np.float32, copy=False)
        white = obs[1].astype(np.float32, copy=False)
        return black, white

    def _reshape_observation(self, obs_flat: np.ndarray) -> np.ndarray:
        obs = obs_flat.reshape(self._obs_shape)

        if obs.ndim == 2:
            return obs[None, :, :]

        if obs.ndim != 3:
            raise ValueError(f"Unsupported observation tensor rank: {obs.ndim}")

        if obs.shape[0] == self.board_size and obs.shape[1] == self.board_size:
            # HWC -> CHW
            return np.transpose(obs, (2, 0, 1)).astype(np.float32, copy=False)

        if obs.shape[1] == self.board_size and obs.shape[2] == self.board_size:
            # Already CHW
            return obs.astype(np.float32, copy=False)

        raise ValueError(f"Unexpected observation shape for Go: {obs.shape}")
