from __future__ import annotations

from typing import Callable

from config import BOARD_SIZE, GAME_NAME, N_IN_ROW
from gomoku.game import GomokuGame

from .base import GameLike

GameFactory = Callable[[], GameLike]


def _make_gomoku() -> GomokuGame:
    return GomokuGame(BOARD_SIZE, N_IN_ROW)


GAME_FACTORIES: dict[str, GameFactory] = {
    "gomoku": _make_gomoku,
}


def available_games() -> list[str]:
    return sorted(GAME_FACTORIES.keys())


def normalize_game_name(game_name: str | None) -> str:
    name = (game_name or GAME_NAME).strip().lower()
    if not name:
        return GAME_NAME
    return name


def create_game(game_name: str | None = None) -> GameLike:
    name = normalize_game_name(game_name)
    factory = GAME_FACTORIES.get(name)
    if factory is None:
        options = ", ".join(available_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return factory()
