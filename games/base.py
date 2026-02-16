from __future__ import annotations

from typing import Any, Protocol, Tuple

import numpy as np


class GameLike(Protocol):
    """
    Minimal game interface required by the AlphaZero training pipeline.
    """

    def getInitBoard(self) -> Any:
        ...

    def getBoardSize(self) -> Tuple[int, int]:
        ...

    def getActionSize(self) -> int:
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
