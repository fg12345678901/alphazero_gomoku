"""Game abstraction entrypoints.

Keep this module lightweight: importing ``games.base`` from game
implementations should not eagerly import the game registry, otherwise
``gomoku.game -> games.base`` can recurse back into ``gomoku.game``.
"""

from .base import GameLike, GameSpec


def available_games() -> list[str]:
    from .registry import available_games as _available_games

    return _available_games()


def create_game(game_name: str | None = None):
    from .registry import create_game as _create_game

    return _create_game(game_name)


def normalize_game_name(game_name: str | None = None) -> str:
    from .registry import normalize_game_name as _normalize_game_name

    return _normalize_game_name(game_name)


__all__ = ["GameLike", "GameSpec", "available_games", "create_game", "normalize_game_name"]
