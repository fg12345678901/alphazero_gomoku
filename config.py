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


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


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


def _require_str(section: dict[str, Any], key: str, file_path: Path) -> str:
    if key not in section:
        raise ValueError(f"{file_path} missing key '{key}'")
    value = section[key]
    if value is None:
        raise ValueError(f"{file_path} key '{key}' must be non-empty string")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{file_path} key '{key}' must be non-empty string")
    return text


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
class ModelConfig:
    channels: int
    num_res: int


@dataclass(frozen=True)
class TrainConfig:
    buffer_size: int
    batch_size: int
    train_updates: int
    learning_rate: float
    weight_decay: float
    eval_threshold: float
    selfplay_games: int


@dataclass(frozen=True)
class RuntimeConfig:
    model_dir: str
    data_dir: str
    log_dir: str
    tb_dir: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    name: str


@dataclass(frozen=True)
class SystemConfig:
    default_game: str
    device: str


@dataclass(frozen=True)
class GameConfig:
    rule: RuleConfig
    search: SearchConfig
    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeConfig
    logging: LoggingConfig


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
SYSTEM_CONFIG_PATH = CONFIG_DIR / "system.yaml"
_NON_GAME_CONFIG_NAMES = {"system"}


def _load_game_yaml(game_name: str, file_path: Path) -> GameConfig:
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found for game '{game_name}': {file_path}")

    with file_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a mapping")

    rules = _require_dict(payload, "rules", file_path)
    search = _require_dict(payload, "search", file_path)
    model = _require_dict(payload, "model", file_path)
    train = _require_dict(payload, "train", file_path)
    runtime = _require_dict(payload, "runtime", file_path)
    logging_cfg = _require_dict(payload, "logging", file_path)

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
    model_cfg = ModelConfig(
        channels=_require_int(model, "channels", file_path),
        num_res=_require_int(model, "num_res", file_path),
    )
    train_cfg = TrainConfig(
        buffer_size=_require_int(train, "buffer_size", file_path),
        batch_size=_require_int(train, "batch_size", file_path),
        train_updates=_require_int(train, "train_updates", file_path),
        learning_rate=_require_float(train, "learning_rate", file_path),
        weight_decay=_require_float(train, "weight_decay", file_path),
        eval_threshold=_require_float(train, "eval_threshold", file_path),
        selfplay_games=_require_int(train, "selfplay_games", file_path),
    )
    runtime_cfg = RuntimeConfig(
        model_dir=_require_str(runtime, "model_dir", file_path),
        data_dir=_require_str(runtime, "data_dir", file_path),
        log_dir=_require_str(runtime, "log_dir", file_path),
        tb_dir=_require_str(runtime, "tb_dir", file_path),
    )
    logging_resolved = LoggingConfig(
        level=_require_str(logging_cfg, "level", file_path),
        name=_require_str(logging_cfg, "name", file_path),
    )

    return GameConfig(
        rule=rule_cfg,
        search=search_cfg,
        model=model_cfg,
        train=train_cfg,
        runtime=runtime_cfg,
        logging=logging_resolved,
    )


def _load_system_yaml(file_path: Path) -> SystemConfig:
    if not file_path.exists():
        raise FileNotFoundError(f"System config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a mapping")

    system = _require_dict(payload, "system", file_path)
    return SystemConfig(
        default_game=_normalize_game_name(_require_str(system, "default_game", file_path)),
        device=_require_str(system, "device", file_path),
    )


def _load_all_game_configs() -> dict[str, GameConfig]:
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"Config directory does not exist: {CONFIG_DIR}")

    configs: dict[str, GameConfig] = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        game_name = path.stem.strip().lower()
        if not game_name or game_name in _NON_GAME_CONFIG_NAMES:
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


def _apply_model_env_overrides(game_name: str, model: ModelConfig) -> ModelConfig:
    prefix = game_name.upper()
    return ModelConfig(
        channels=_env_int(f"{prefix}_CHANNELS", _env_int("AZ_CHANNELS", model.channels)),
        num_res=_env_int(f"{prefix}_NUM_RES", _env_int("AZ_NUM_RES", model.num_res)),
    )


def _apply_train_env_overrides(game_name: str, train: TrainConfig) -> TrainConfig:
    prefix = game_name.upper()
    return TrainConfig(
        buffer_size=_env_int(f"{prefix}_BUFFER_SIZE", _env_int("AZ_BUFFER_SIZE", train.buffer_size)),
        batch_size=_env_int(f"{prefix}_BATCH_SIZE", _env_int("AZ_BATCH_SIZE", train.batch_size)),
        train_updates=_env_int(f"{prefix}_TRAIN_UPDATES", _env_int("AZ_TRAIN_UPDATES", train.train_updates)),
        learning_rate=_env_float(f"{prefix}_LEARNING_RATE", _env_float("AZ_LEARNING_RATE", train.learning_rate)),
        weight_decay=_env_float(f"{prefix}_WEIGHT_DECAY", _env_float("AZ_WEIGHT_DECAY", train.weight_decay)),
        eval_threshold=_env_float(
            f"{prefix}_EVAL_THRESHOLD",
            _env_float("AZ_EVAL_THRESHOLD", train.eval_threshold),
        ),
        selfplay_games=_env_int(
            f"{prefix}_SELFPLAY_GAMES",
            _env_int("AZ_SELFPLAY_GAMES", train.selfplay_games),
        ),
    )


def _apply_runtime_env_overrides(game_name: str, runtime: RuntimeConfig) -> RuntimeConfig:
    prefix = game_name.upper()
    return RuntimeConfig(
        model_dir=_env_str(f"{prefix}_MODEL_DIR", _env_str("AZ_MODEL_DIR", runtime.model_dir)),
        data_dir=_env_str(f"{prefix}_DATA_DIR", _env_str("AZ_DATA_DIR", runtime.data_dir)),
        log_dir=_env_str(f"{prefix}_LOG_DIR", _env_str("AZ_LOG_DIR", runtime.log_dir)),
        tb_dir=_env_str(f"{prefix}_TB_DIR", _env_str("AZ_TB_DIR", runtime.tb_dir)),
    )


def _apply_logging_env_overrides(game_name: str, logging_cfg: LoggingConfig) -> LoggingConfig:
    prefix = game_name.upper()
    return LoggingConfig(
        level=_env_str(
            f"{prefix}_LOG_LEVEL",
            _env_str("AZ_LOG_LEVEL", logging_cfg.level),
        ),
        name=_env_str(
            f"{prefix}_LOG_NAME",
            _env_str("AZ_LOG_NAME", logging_cfg.name),
        ),
    )


def _apply_system_env_overrides(system_cfg: SystemConfig) -> SystemConfig:
    return SystemConfig(
        default_game=_normalize_game_name(
            _env_str("GAME_NAME", _env_str("AZ_DEFAULT_GAME", system_cfg.default_game))
        ),
        device=_env_str("AZ_DEVICE", system_cfg.device),
    )


def _resolve_device(device_cfg: str) -> str:
    normalized = (device_cfg or "").strip().lower()
    if not normalized:
        raise ValueError("Device config must be non-empty")

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "mps":
        mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if not mps_available:
            raise ValueError("Configured device='mps' but torch MPS backend is unavailable")
        return "mps"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Configured device='cuda' but CUDA is unavailable")
        return "cuda"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Configured device='{normalized}' but CUDA is unavailable")
        index_text = normalized.split(":", 1)[1]
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device index in '{normalized}'") from exc
        if index < 0:
            raise ValueError(f"Invalid CUDA device index in '{normalized}'")
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Configured device='{normalized}' but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        return normalized

    raise ValueError("Unsupported device config. Use auto/cpu/cuda/cuda:<index>/mps")


_RAW_SYSTEM_CONFIG = _load_system_yaml(SYSTEM_CONFIG_PATH)
_SYSTEM_CONFIG = _apply_system_env_overrides(_RAW_SYSTEM_CONFIG)
_RAW_GAME_CONFIGS = _load_all_game_configs()
_RULE_CONFIGS: dict[str, RuleConfig] = {
    game: _apply_rule_env_overrides(game, cfg.rule)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_SEARCH_CONFIGS: dict[str, SearchConfig] = {
    game: _apply_search_env_overrides(game, cfg.search)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_MODEL_CONFIGS: dict[str, ModelConfig] = {
    game: _apply_model_env_overrides(game, cfg.model)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_TRAIN_CONFIGS: dict[str, TrainConfig] = {
    game: _apply_train_env_overrides(game, cfg.train)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_RUNTIME_CONFIGS: dict[str, RuntimeConfig] = {
    game: _apply_runtime_env_overrides(game, cfg.runtime)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}
_LOGGING_CONFIGS: dict[str, LoggingConfig] = {
    game: _apply_logging_env_overrides(game, cfg.logging)
    for game, cfg in _RAW_GAME_CONFIGS.items()
}


def supported_games() -> list[str]:
    return sorted(_RULE_CONFIGS.keys())


_DEFAULT_GAME_NAME = _normalize_game_name(_SYSTEM_CONFIG.default_game)
if not _DEFAULT_GAME_NAME:
    _DEFAULT_GAME_NAME = "gomoku" if "gomoku" in _RULE_CONFIGS else supported_games()[0]
GAME_NAME = _DEFAULT_GAME_NAME
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


def get_model_config(game_name: str | None = None) -> ModelConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _MODEL_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _MODEL_CONFIGS[normalized]


def get_train_config(game_name: str | None = None) -> TrainConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _TRAIN_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _TRAIN_CONFIGS[normalized]


def get_runtime_config(game_name: str | None = None) -> RuntimeConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _RUNTIME_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _RUNTIME_CONFIGS[normalized]


def get_logging_config(game_name: str | None = None) -> LoggingConfig:
    normalized = _normalize_game_name(game_name or GAME_NAME)
    if normalized not in _LOGGING_CONFIGS:
        options = ", ".join(supported_games())
        raise ValueError(f"Unsupported game '{game_name}'. Available games: {options}")
    return _LOGGING_CONFIGS[normalized]


def get_system_config() -> SystemConfig:
    return _SYSTEM_CONFIG


_ACTIVE_RULE = get_rule_config(GAME_NAME)
_ACTIVE_SEARCH = get_search_config(GAME_NAME)
_ACTIVE_MODEL = get_model_config(GAME_NAME)
_ACTIVE_TRAIN = get_train_config(GAME_NAME)
_ACTIVE_RUNTIME = get_runtime_config(GAME_NAME)
_ACTIVE_LOGGING = get_logging_config(GAME_NAME)
_ACTIVE_SYSTEM = get_system_config()

CHANNELS = _ACTIVE_MODEL.channels
NUM_RES = _ACTIVE_MODEL.num_res

BUFFER_SIZE = _ACTIVE_TRAIN.buffer_size
BATCH_SIZE = _ACTIVE_TRAIN.batch_size
TRAIN_UPDATES = _ACTIVE_TRAIN.train_updates
LEARNING_RATE = _ACTIVE_TRAIN.learning_rate
WEIGHT_DECAY = _ACTIVE_TRAIN.weight_decay
EVAL_THRESHOLD = _ACTIVE_TRAIN.eval_threshold  # kept for compatibility
SELFPLAY_GAMES = _ACTIVE_TRAIN.selfplay_games

DEVICE = _resolve_device(_ACTIVE_SYSTEM.device)
MODEL_DIR = _ACTIVE_RUNTIME.model_dir
DATA_DIR = _ACTIVE_RUNTIME.data_dir

LOG_DIR = _ACTIVE_RUNTIME.log_dir
TB_DIR = _ACTIVE_RUNTIME.tb_dir
LOG_LEVEL = _ACTIVE_LOGGING.level
LOG_NAME = _ACTIVE_LOGGING.name


# Legacy aliases for scripts importing scalar constants.
BOARD_SIZE = _ACTIVE_RULE.board_size
N_IN_ROW = _ACTIVE_RULE.n_in_row or 5
HISTORY_STEPS = _ACTIVE_RULE.history_steps
INPUT_PLANES = _ACTIVE_RULE.input_planes

MCTS_SIMS = _ACTIVE_SEARCH.mcts_sims
CPUCT = _ACTIVE_SEARCH.cpuct
DIRICHLET_ALPHA = _ACTIVE_SEARCH.dirichlet_alpha
DIRICHLET_EPS = _ACTIVE_SEARCH.dirichlet_eps
SELFPLAY_TEMPERATURE = _ACTIVE_SEARCH.selfplay_temperature
N_TEMP_MOVES = _ACTIVE_SEARCH.n_temp_moves
EVAL_GAMES = _ACTIVE_SEARCH.eval_games
