from __future__ import annotations

from dataclasses import dataclass
import os

from config import DATA_DIR, GAME_NAME, LOG_DIR, MODEL_DIR, TB_DIR
from games.registry import normalize_game_name


@dataclass(frozen=True)
class RuntimePaths:
    model_dir: str
    data_dir: str
    log_dir: str
    tb_dir: str


def _scoped_dir(base_dir: str, game_name: str) -> str:
    # Keep legacy paths for Gomoku; create per-game namespaces for new games.
    if game_name == "gomoku":
        return base_dir
    return os.path.join(base_dir, game_name)


def resolve_runtime_paths(game_name: str | None = None) -> RuntimePaths:
    normalized = normalize_game_name(game_name or GAME_NAME)
    return RuntimePaths(
        model_dir=_scoped_dir(MODEL_DIR, normalized),
        data_dir=_scoped_dir(DATA_DIR, normalized),
        log_dir=_scoped_dir(LOG_DIR, normalized),
        tb_dir=_scoped_dir(TB_DIR, normalized),
    )
