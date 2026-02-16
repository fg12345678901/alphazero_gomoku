from __future__ import annotations

from typing import Callable

from config import GAME_NAME, get_rule_config
from gomoku.game import GomokuGame

from .base import GameLike
from .go_openspiel import GoOpenSpielGame

GameFactory = Callable[[], GameLike]


def _make_gomoku() -> GomokuGame:
    rules = get_rule_config("gomoku")
    return GomokuGame(
        size=rules.board_size,
        n_in_row=rules.n_in_row or 5,
        history_steps=rules.history_steps,
    )


def _make_go() -> GoOpenSpielGame:
    rules = get_rule_config("go")
    return GoOpenSpielGame(
        board_size=rules.board_size,
        komi=rules.komi if rules.komi is not None else 7.5,
        history_steps=rules.history_steps,
    )


GAME_FACTORIES: dict[str, GameFactory] = {
    "gomoku": _make_gomoku,
    "go": _make_go,
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
