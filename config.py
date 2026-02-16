from __future__ import annotations

from dataclasses import dataclass
import os

import torch


def _normalize_game_name(name: str | None) -> str:
    normalized = (name or "").strip().lower()
    return normalized or "gomoku"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class RuleConfig:
    board_size: int
    history_steps: int
    n_in_row: int | None = None
    komi: float | None = None

    @property
    def input_planes(self) -> int:
        return 2 * self.history_steps + 1


@dataclass(frozen=True)
class SearchConfig:
    mcts_sims: int
    cpuct: float
    dirichlet_alpha: float
    dirichlet_eps: float
    selfplay_temperature: float
    n_temp_moves: int
    eval_games: int


GAME_NAME = _normalize_game_name(os.getenv("GAME_NAME", "gomoku"))


_RULE_CONFIGS: dict[str, RuleConfig] = {
    "gomoku": RuleConfig(
        board_size=_env_int("GOMOKU_BOARD_SIZE", 15),
        n_in_row=_env_int("GOMOKU_N_IN_ROW", 5),
        history_steps=_env_int("GOMOKU_HISTORY_STEPS", 3),
    ),
    "go": RuleConfig(
        board_size=_env_int("GO_BOARD_SIZE", 9),
        history_steps=_env_int("GO_HISTORY_STEPS", 8),
        komi=_env_float("GO_KOMI", 7.5),
    ),
}

_SEARCH_CONFIGS: dict[str, SearchConfig] = {
    "gomoku": SearchConfig(
        mcts_sims=_env_int("GOMOKU_MCTS_SIMS", 1700),
        cpuct=_env_float("GOMOKU_CPUCT", 2.5),
        dirichlet_alpha=_env_float("GOMOKU_DIRICHLET_ALPHA", 0.06),
        dirichlet_eps=_env_float("GOMOKU_DIRICHLET_EPS", 0.25),
        selfplay_temperature=_env_float("GOMOKU_SELFPLAY_TEMPERATURE", 1.0),
        n_temp_moves=_env_int("GOMOKU_N_TEMP_MOVES", 10),
        eval_games=_env_int("GOMOKU_EVAL_GAMES", 100),
    ),
    "go": SearchConfig(
        mcts_sims=_env_int("GO_MCTS_SIMS", 800),
        cpuct=_env_float("GO_CPUCT", 1.5),
        dirichlet_alpha=_env_float("GO_DIRICHLET_ALPHA", 0.03),
        dirichlet_eps=_env_float("GO_DIRICHLET_EPS", 0.25),
        selfplay_temperature=_env_float("GO_SELFPLAY_TEMPERATURE", 1.0),
        n_temp_moves=_env_int("GO_N_TEMP_MOVES", 30),
        eval_games=_env_int("GO_EVAL_GAMES", 100),
    ),
}


def supported_games() -> list[str]:
    return sorted(_RULE_CONFIGS.keys())


def get_rule_config(game_name: str | None = None) -> RuleConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _RULE_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _RULE_CONFIGS[normalized]


def get_search_config(game_name: str | None = None) -> SearchConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _SEARCH_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _SEARCH_CONFIGS[normalized]


CHANNELS = _env_int("AZ_CHANNELS", 256)
NUM_RES = _env_int("AZ_NUM_RES", 15)

BUFFER_SIZE = _env_int("AZ_BUFFER_SIZE", 800_000)
BATCH_SIZE = _env_int("AZ_BATCH_SIZE", 512)
TRAIN_UPDATES = _env_int("AZ_TRAIN_UPDATES", 8000)
LEARNING_RATE = _env_float("AZ_LEARNING_RATE", 8e-4)
WEIGHT_DECAY = _env_float("AZ_WEIGHT_DECAY", 1e-4)

EVAL_THRESHOLD = _env_float("AZ_EVAL_THRESHOLD", 0.55)  # kept for compatibility

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
DATA_DIR = "data"

LOG_DIR = "logs"
LOG_LEVEL = os.getenv("AZ_LOG_LEVEL", "INFO")
LOG_NAME = "alphazero"
TB_DIR = "tb"


# Legacy aliases for scripts that still import fixed Gomoku constants.
_GOMOKU_RULE = get_rule_config("gomoku")
BOARD_SIZE = _GOMOKU_RULE.board_size
N_IN_ROW = _GOMOKU_RULE.n_in_row or 5
HISTORY_STEPS = _GOMOKU_RULE.history_steps
INPUT_PLANES = _GOMOKU_RULE.input_planes

_GOMOKU_SEARCH = get_search_config("gomoku")
MCTS_SIMS = _GOMOKU_SEARCH.mcts_sims
CPUCT = _GOMOKU_SEARCH.cpuct
DIRICHLET_ALPHA = _GOMOKU_SEARCH.dirichlet_alpha
DIRICHLET_EPS = _GOMOKU_SEARCH.dirichlet_eps
SELFPLAY_TEMPERATURE = _GOMOKU_SEARCH.selfplay_temperature
N_TEMP_MOVES = _GOMOKU_SEARCH.n_temp_moves
EVAL_GAMES = _GOMOKU_SEARCH.eval_games
