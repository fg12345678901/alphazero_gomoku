"""Game abstraction entrypoints."""

from .base import GameLike
from .registry import available_games, create_game, normalize_game_name

__all__ = [
    "GameLike",
    "available_games",
    "create_game",
    "normalize_game_name",
]
