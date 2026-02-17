from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
import yaml


def _normalize_game_name(name: str | None) -> str:
    normalized = (name or "").strip().lower()
    return normalized


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


def _require_dict(parent: dict[str, Any], key: str, file_path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{file_path} missing mapping section '{key}'")
    return value


def _require_int(section: dict[str, Any], key: str, file_path: Path) -> int:
    if key not in section:
        raise ValueError(f"{file_path} missing key '{key}'")
    try:
        return int(section[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{file_path} key '{key}' must be int") from exc


def _require_float(section: dict[str, Any], key: str, file_path: Path) -> float:
    if key not in section:
        raise ValueError(f"{file_path} missing key '{key}'")
    try:
        return float(section[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{file_path} key '{key}' must be float") from exc


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


@dataclass(frozen=True)
class GameConfig:
    rule: RuleConfig
    search: SearchConfig


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _load_game_yaml(game_name: str, file_path: Path) -> GameConfig:
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found for game '{game_name}': {file_path}")

    with file_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a mapping")

    rules = _require_dict(payload, "rules", file_path)
    search = _require_dict(payload, "search", file_path)

    n_in_row = rules.get("n_in_row")
    komi = rules.get("komi")
    if n_in_row is not None:
        n_in_row = int(n_in_row)
    if komi is not None:
        komi = float(komi)

    rule_cfg = RuleConfig(
        board_size=_require_int(rules, "board_size", file_path),
        history_steps=_require_int(rules, "history_steps", file_path),
        n_in_row=n_in_row,
        komi=komi,
    )
    search_cfg = SearchConfig(
        mcts_sims=_require_int(search, "mcts_sims", file_path),
        cpuct=_require_float(search, "cpuct", file_path),
        dirichlet_alpha=_require_float(search, "dirichlet_alpha", file_path),
        dirichlet_eps=_require_float(search, "dirichlet_eps", file_path),
        selfplay_temperature=_require_float(search, "selfplay_temperature", file_path),
        n_temp_moves=_require_int(search, "n_temp_moves", file_path),
        eval_games=_require_int(search, "eval_games", file_path),
    )

    return GameConfig(rule=rule_cfg, search=search_cfg)


def _load_all_game_configs() -> dict[str, GameConfig]:
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"Config directory does not exist: {CONFIG_DIR}")

    configs: dict[str, GameConfig] = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        game_name = path.stem.strip().lower()
        if not game_name:
            continue
        configs[game_name] = _load_game_yaml(game_name, path)

    if not configs:
        raise RuntimeError(f"No game YAML config found under {CONFIG_DIR}")
    return configs


def _apply_rule_env_overrides(game_name: str, rule: RuleConfig) -> RuleConfig:
    if game_name == "gomoku":
        return RuleConfig(
            board_size=_env_int("GOMOKU_BOARD_SIZE", rule.board_size),
            history_steps=_env_int("GOMOKU_HISTORY_STEPS", rule.history_steps),
            n_in_row=_env_int("GOMOKU_N_IN_ROW", rule.n_in_row or 5),
            komi=rule.komi,
        )

    if game_name == "go":
        return RuleConfig(
            board_size=_env_int("GO_BOARD_SIZE", rule.board_size),
            history_steps=_env_int("GO_HISTORY_STEPS", rule.history_steps),
            n_in_row=rule.n_in_row,
            komi=_env_float("GO_KOMI", rule.komi if rule.komi is not None else 7.5),
        )

    return rule


def _apply_search_env_overrides(game_name: str, search: SearchConfig) -> SearchConfig:
    if game_name == "gomoku":
        return SearchConfig(
            mcts_sims=_env_int("GOMOKU_MCTS_SIMS", search.mcts_sims),
            cpuct=_env_float("GOMOKU_CPUCT", search.cpuct),
            dirichlet_alpha=_env_float("GOMOKU_DIRICHLET_ALPHA", search.dirichlet_alpha),
            dirichlet_eps=_env_float("GOMOKU_DIRICHLET_EPS", search.dirichlet_eps),
            selfplay_temperature=_env_float("GOMOKU_SELFPLAY_TEMPERATURE", search.selfplay_temperature),
            n_temp_moves=_env_int("GOMOKU_N_TEMP_MOVES", search.n_temp_moves),
            eval_games=_env_int("GOMOKU_EVAL_GAMES", search.eval_games),
        )

    if game_name == "go":
        return SearchConfig(
            mcts_sims=_env_int("GO_MCTS_SIMS", search.mcts_sims),
            cpuct=_env_float("GO_CPUCT", search.cpuct),
            dirichlet_alpha=_env_float("GO_DIRICHLET_ALPHA", search.dirichlet_alpha),
            dirichlet_eps=_env_float("GO_DIRICHLET_EPS", search.dirichlet_eps),
            selfplay_temperature=_env_float("GO_SELFPLAY_TEMPERATURE", search.selfplay_temperature),
            n_temp_moves=_env_int("GO_N_TEMP_MOVES", search.n_temp_moves),
            eval_games=_env_int("GO_EVAL_GAMES", search.eval_games),
        )

    return search


_RAW_GAME_CONFIGS = _load_all_game_configs()
_RULE_CONFIGS: dict[str, RuleConfig] = {
    game: _apply_rule_env_overrides(game, cfg.rule)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_SEARCH_CONFIGS: dict[str, SearchConfig] = {
    game: _apply_search_env_overrides(game, cfg.search)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}


def supported_games() -> list[str]:
    return sorted(_RULE_CONFIGS.keys())


_DEFAULT_GAME_NAME = "gomoku" if "gomoku" in _RULE_CONFIGS else supported_games()[0]
GAME_NAME = _normalize_game_name(os.getenv("GAME_NAME", _DEFAULT_GAME_NAME))
if GAME_NAME not in _RULE_CONFIGS:
    options = ", ".join(supported_games())
    raise ValueError(f"Unsupported GAME_NAME='{GAME_NAME}'. Available games: {options}")


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
