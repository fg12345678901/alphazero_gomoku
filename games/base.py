from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class GameSpec:
    name: str
    board_size: int
    action_size: int
    input_planes: int


class GameLike(Protocol):
    """
    Game interface required by the AlphaZero pipeline.

    The core loop must only use this interface, and must not rely on
    game-specific board internals (for example `board.current_player`).
    """

    def getGameSpec(self) -> GameSpec:
        ...

    def getInitBoard(self) -> Any:
        ...

    def getBoardSize(self) -> Tuple[int, int]:
        ...

    def getActionSize(self) -> int:
        ...

    def getCurrentPlayer(self, board: Any) -> int:
        ...

    def getNextState(self, board: Any, action: int):
        ...

    def getValidMoves(self, board: Any) -> np.ndarray:
        ...

    def getGameEnded(self, board: Any, player: int) -> float:
        ...

    def getCanonicalForm(self, board: Any, player: int) -> np.ndarray:
        ...

    def getSymmetries(self, board_planes: np.ndarray, pi: np.ndarray):
        ...

    def stringRepresentation(self, board: Any) -> bytes:
        ...
