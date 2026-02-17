from __future__ import annotations

from dataclasses import dataclass
import os

from config import GAME_NAME, get_runtime_config
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
    runtime_cfg = get_runtime_config(normalized)
    return RuntimePaths(
        model_dir=_scoped_dir(runtime_cfg.model_dir, normalized),
        data_dir=_scoped_dir(runtime_cfg.data_dir, normalized),
        log_dir=_scoped_dir(runtime_cfg.log_dir, normalized),
        tb_dir=_scoped_dir(runtime_cfg.tb_dir, normalized),
    )
