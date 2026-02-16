"""Game abstraction entrypoints."""

from .base import GameLike, GameSpec
from .registry import available_games, create_game, normalize_game_name

__all__ = [
    "GameLike",
    "GameSpec",
    "available_games",
    "create_game",
    "normalize_game_name",
]
