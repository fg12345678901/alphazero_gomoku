from __future__ import annotations

import glob
import os
import re
import threading
from copy import deepcopy

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

from config import DEVICE, GAME_NAME, get_search_config
from games.registry import available_games, create_game, normalize_game_name
from mcts.mcts import MCTS
from network.checkpoint import build_model_for_game, load_checkpoint, validate_checkpoint_meta
from runtime_paths import RuntimePaths, resolve_runtime_paths


app = Flask(__name__)
MODEL_LOCK = threading.Lock()

ACTIVE_GAME_NAME = normalize_game_name(GAME_NAME)
GAME = None
NET = None
PATHS: RuntimePaths | None = None

CURRENT_MODEL_PATH: str | None = None
_LOADED_MODEL_PATH: str | None = None

BOARD = None
STATE_HISTORY = []
ACTION_HISTORY: list[int] = []
MOVE_LABELS: list[str] = []
VALUE_CURVE: list[float] = []
POLICY_BOARD: list[list[float]] = []
PASS_PROB = 0.0

MODE = "human_ai"  # human_ai, human_human, ai_ai
HUMAN_PLAYER = 1
MCTS_OBJ: MCTS | None = None


def _ensure_paths(paths: RuntimePaths) -> None:
    os.makedirs(paths.model_dir, exist_ok=True)
    os.makedirs(paths.data_dir, exist_ok=True)
    os.makedirs(paths.log_dir, exist_ok=True)
    os.makedirs(paths.tb_dir, exist_ok=True)


def _latest_model_for_game(game_name: str) -> str | None:
    paths = resolve_runtime_paths(game_name)
    files = sorted(glob.glob(os.path.join(paths.model_dir, "net_*.pt")))
    return files[-1] if files else None


def _list_models_for_game(game_name: str) -> list[str]:
    paths = resolve_runtime_paths(game_name)
    files = sorted(glob.glob(os.path.join(paths.model_dir, "net_*.pt")))
    return [os.path.basename(f) for f in files]


def _resolve_model_path(game_name: str, path: str | None) -> str | None:
    if not path:
        return None
    if os.path.basename(path) == path:
        return os.path.join(resolve_runtime_paths(game_name).model_dir, path)
    return path


def _is_go() -> bool:
    return ACTIVE_GAME_NAME == "go"


def _clone_state(state):
    if hasattr(state, "copy"):
        return state.copy()
    if hasattr(state, "clone"):
        return state.clone()
    return deepcopy(state)


def _go_columns_from_state(state, size: int) -> list[str]:
    cols: list[str] = []
    for action in range(size):
        try:
            coord = str(state.action_to_string(state.current_player(), int(action))).split()[-1]
        except Exception:
            cols = []
            break
        if not coord:
            cols = []
            break
        cols.append(coord[0].upper())

    if len(cols) == size and len(set(cols)) == size:
        return cols

    # Fallback: Go coordinates conventionally skip I.
    fallback = [c for c in "ABCDEFGHJKLMNOPQRSTUVWXYZ"]
    return fallback[:size]


def _columns_for_state(state) -> list[str]:
    size = GAME.getGameSpec().board_size
    if _is_go():
        return _go_columns_from_state(state, size)
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:size])


def _go_action_to_xy(state, action: int, size: int) -> tuple[int, int] | None:
    try:
        coord = str(state.action_to_string(state.current_player(), int(action))).split()[-1].upper()
    except Exception:
        return None

    if coord == "PASS":
        return None

    col = coord[0]
    row = int(coord[1:])
    cols = _go_columns_from_state(state, size)
    if col not in cols:
        return None

    x = size - row
    y = cols.index(col)
    if 0 <= x < size and 0 <= y < size:
        return (x, y)
    return None


def _go_xy_to_action(state, x: int, y: int, size: int) -> int:
    cols = _go_columns_from_state(state, size)
    coord = f"{cols[y].lower()}{size - x}"
    return int(state.string_to_action(coord))


def _board_matrix(state) -> np.ndarray:
    size = GAME.getGameSpec().board_size
    if not _is_go():
        return np.asarray(state.board, dtype=np.int8)

    rows: list[list[int]] = []
    line_re = re.compile(r"^\s*\d+\s+([+XOxo.]+)\s*$")
    char_map = {"X": 1, "x": 1, "O": -1, "o": -1, "+": 0, ".": 0}

    for line in str(state).splitlines():
        matched = line_re.match(line)
        if not matched:
            continue
        row = [char_map.get(ch, 0) for ch in matched.group(1)]
        rows.append(row)

    if len(rows) != size or any(len(r) != size for r in rows):
        return np.zeros((size, size), dtype=np.int8)

    return np.asarray(rows, dtype=np.int8)


def _policy_to_payload(policy_vec: np.ndarray, state) -> tuple[list[list[float]], float]:
    size = GAME.getGameSpec().board_size
    board_probs = np.zeros((size, size), dtype=np.float32)
    pass_prob = 0.0

    if not _is_go():
        board_probs = np.asarray(policy_vec[: size * size], dtype=np.float32).reshape(size, size)
        return board_probs.tolist(), pass_prob

    for action in range(min(size * size, policy_vec.shape[0])):
        pos = _go_action_to_xy(state, action, size)
        if pos is None:
            continue
        x, y = pos
        board_probs[x, y] = float(policy_vec[action])

    if policy_vec.shape[0] > size * size:
        pass_prob = float(policy_vec[size * size])

    return board_probs.tolist(), pass_prob


def _action_label(state, action: int) -> str:
    size = GAME.getGameSpec().board_size
    if _is_go():
        try:
            coord = str(state.action_to_string(state.current_player(), int(action))).split()[-1]
            return coord.upper()
        except Exception:
            if action == size * size:
                return "PASS"
            return str(action)

    row, col = divmod(int(action), size)
    cols = _columns_for_state(state)
    return f"{cols[col]}{row + 1}"


def _action_from_xy(state, x: int, y: int) -> int:
    size = GAME.getGameSpec().board_size
    if _is_go():
        return _go_xy_to_action(state, x, y, size)
    return x * size + y


def _winner_from_board(state) -> int | None:
    result_for_black = float(GAME.getGameEnded(state, 1))
    if result_for_black == 0:
        return None
    if abs(result_for_black) < 1e-3:
        return 0
    return 1 if result_for_black > 0 else -1


def _evaluate_state(state) -> tuple[np.ndarray, float]:
    current_player = GAME.getCurrentPlayer(state)
    planes = GAME.getCanonicalForm(state, current_player)
    x = torch.tensor(planes, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = NET(x)

    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    valids = GAME.getValidMoves(state).astype(np.float32)
    policy = policy * valids
    s = float(np.sum(policy))
    if s > 0:
        policy /= s
    else:
        legal_sum = float(np.sum(valids))
        if legal_sum > 0:
            policy = valids / legal_sum
        else:
            policy = np.full(GAME.getActionSize(), 1.0 / GAME.getActionSize(), dtype=np.float32)

    return policy.astype(np.float32), float(value.item())


def _create_mcts(sims: int | None = None) -> MCTS:
    cfg = get_search_config(ACTIVE_GAME_NAME)
    use_sims = int(sims) if sims is not None and int(sims) > 0 else cfg.mcts_sims
    return MCTS(
        GAME,
        NET,
        sims=use_sims,
        cpuct=cfg.cpuct,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_eps=cfg.dirichlet_eps,
    )


def _reset_random_network() -> None:
    global NET, CURRENT_MODEL_PATH, _LOADED_MODEL_PATH
    NET = build_model_for_game(GAME, device=DEVICE)
    NET.eval()
    CURRENT_MODEL_PATH = None
    _LOADED_MODEL_PATH = None


def _load_model(path: str | None) -> bool:
    global CURRENT_MODEL_PATH, _LOADED_MODEL_PATH

    resolved = _resolve_model_path(ACTIVE_GAME_NAME, path)
    if resolved is None or not os.path.isfile(resolved):
        return False

    with MODEL_LOCK:
        if _LOADED_MODEL_PATH == resolved:
            CURRENT_MODEL_PATH = resolved
            return True

        state_dict, meta = load_checkpoint(resolved, map_location=DEVICE)
        validate_checkpoint_meta(meta, GAME.getGameSpec())
        NET.load_state_dict(state_dict)
        NET.eval()

        _LOADED_MODEL_PATH = resolved
        CURRENT_MODEL_PATH = resolved
    return True


def _activate_game(game_name: str) -> None:
    global ACTIVE_GAME_NAME, GAME, NET, PATHS, CURRENT_MODEL_PATH, _LOADED_MODEL_PATH

    normalized = normalize_game_name(game_name)
    if GAME is not None and ACTIVE_GAME_NAME == normalized:
        return

    GAME = create_game(normalized)
    NET = build_model_for_game(GAME, device=DEVICE)
    NET.eval()

    ACTIVE_GAME_NAME = normalized
    PATHS = resolve_runtime_paths(ACTIVE_GAME_NAME)
    _ensure_paths(PATHS)

    CURRENT_MODEL_PATH = _latest_model_for_game(ACTIVE_GAME_NAME)
    _LOADED_MODEL_PATH = None


def _recompute_policy(append_curve: bool) -> None:
    global POLICY_BOARD, PASS_PROB, VALUE_CURVE

    policy_vec, value = _evaluate_state(BOARD)
    POLICY_BOARD, PASS_PROB = _policy_to_payload(policy_vec, BOARD)

    signed_value = value * GAME.getCurrentPlayer(BOARD)
    if append_curve:
        VALUE_CURVE.append(float(signed_value))
    else:
        if VALUE_CURVE:
            VALUE_CURVE[-1] = float(signed_value)
        else:
            VALUE_CURVE.append(float(signed_value))


def _apply_action(action: int) -> None:
    global BOARD

    before = BOARD
    label = _action_label(before, action)
    next_board, _ = GAME.getNextState(before, int(action))

    BOARD = next_board
    STATE_HISTORY.append(next_board)
    ACTION_HISTORY.append(int(action))
    MOVE_LABELS.append(label)


def _undo_once() -> bool:
    global BOARD

    if not ACTION_HISTORY or len(STATE_HISTORY) <= 1:
        return False

    ACTION_HISTORY.pop()
    MOVE_LABELS.pop()
    STATE_HISTORY.pop()
    if VALUE_CURVE:
        VALUE_CURVE.pop()

    BOARD = STATE_HISTORY[-1]
    return True


def _ai_move() -> None:
    if _winner_from_board(BOARD) is not None:
        return

    pi = MCTS_OBJ.get_action_probs(BOARD, temp=0, add_noise=False)
    action = int(np.argmax(pi))

    valids = GAME.getValidMoves(BOARD)
    if action < 0 or action >= GAME.getActionSize() or int(valids[action]) == 0:
        legal = np.flatnonzero(valids)
        if len(legal) == 0:
            return
        action = int(np.random.choice(legal))

    _apply_action(action)
    _recompute_policy(append_curve=True)


def _serialize_state() -> dict:
    winner = _winner_from_board(BOARD)
    matrix = _board_matrix(BOARD)
    columns = _columns_for_state(BOARD)

    model_name = os.path.basename(CURRENT_MODEL_PATH) if CURRENT_MODEL_PATH else "random(init)"
    return {
        "game": ACTIVE_GAME_NAME,
        "size": GAME.getGameSpec().board_size,
        "columns": columns,
        "pass_supported": bool(_is_go()),
        "board": matrix.tolist(),
        "current_player": int(GAME.getCurrentPlayer(BOARD)),
        "history": MOVE_LABELS,
        "history_actions": ACTION_HISTORY,
        "policy": POLICY_BOARD,
        "pass_prob": PASS_PROB,
        "value_curve": VALUE_CURVE,
        "winner": winner,
        "model": model_name,
        "mode": MODE,
        "human_player": HUMAN_PLAYER,
    }


def _reset_position(mode: str, human_player: int, sims: int | None) -> None:
    global BOARD, STATE_HISTORY, ACTION_HISTORY, MOVE_LABELS, VALUE_CURVE, MODE, HUMAN_PLAYER, MCTS_OBJ

    MODE = mode
    HUMAN_PLAYER = 1 if int(human_player) >= 0 else -1

    BOARD = GAME.getInitBoard()
    STATE_HISTORY = [BOARD]
    ACTION_HISTORY = []
    MOVE_LABELS = []
    VALUE_CURVE = []

    MCTS_OBJ = _create_mcts(sims)
    _recompute_policy(append_curve=True)

    if MODE != "human_human" and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        _ai_move()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/games")
def games_api():
    return jsonify({"games": available_games(), "current": ACTIVE_GAME_NAME})


@app.route("/models")
def models_api():
    game = normalize_game_name(request.args.get("game", ACTIVE_GAME_NAME))
    models = _list_models_for_game(game)

    current_name = None
    if game == ACTIVE_GAME_NAME and CURRENT_MODEL_PATH:
        current_name = os.path.basename(CURRENT_MODEL_PATH)

    return jsonify({"models": models, "current": current_name})


@app.route("/start", methods=["POST"])
def start_game():
    data = request.get_json(force=True)

    game_name = normalize_game_name(data.get("game", ACTIVE_GAME_NAME))
    mode = data.get("mode", "human_ai")
    human_player = int(data.get("human_player", 1))
    sims = data.get("mcts_sims")
    sims = int(sims) if sims is not None and str(sims).strip() else None

    try:
        _activate_game(game_name)
    except ImportError as exc:
        return jsonify({"error": str(exc)}), 400

    requested_model = data.get("model")
    if requested_model in (None, "", "latest"):
        requested_model = _latest_model_for_game(ACTIVE_GAME_NAME)

    if requested_model == "random":
        _reset_random_network()
    elif requested_model:
        if not _load_model(requested_model):
            return jsonify({"error": f"model not found: {requested_model}"}), 400

    _reset_position(mode=mode, human_player=human_player, sims=sims)
    return jsonify({"success": True, **_serialize_state()})


@app.route("/state")
def state_api():
    return jsonify(_serialize_state())


@app.route("/prior", methods=["POST"])
def prior_api():
    data = request.get_json(force=True)
    size = GAME.getGameSpec().board_size

    if data.get("pass"):
        action = GAME.getActionSize() - 1
    elif "action" in data:
        action = int(data["action"])
    else:
        x = int(data["x"])
        y = int(data["y"])
        if not (0 <= x < size and 0 <= y < size):
            return jsonify({"error": "out of range"}), 400
        action = _action_from_xy(BOARD, x, y)

    valids = GAME.getValidMoves(BOARD)
    if action < 0 or action >= GAME.getActionSize() or int(valids[action]) == 0:
        return jsonify({"error": "invalid"}), 400

    tmp_board, _ = GAME.getNextState(BOARD, action)
    policy, _ = _evaluate_state(tmp_board)
    board_policy, pass_prob = _policy_to_payload(policy, tmp_board)
    return jsonify({"policy": board_policy, "pass_prob": pass_prob})


@app.route("/move", methods=["POST"])
def move_api():
    data = request.get_json(force=True)
    size = GAME.getGameSpec().board_size

    if _winner_from_board(BOARD) is not None:
        return jsonify({"error": "game over"}), 400

    if MODE != "human_human" and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        return jsonify({"error": "not your turn"}), 400

    if data.get("pass"):
        action = GAME.getActionSize() - 1
    elif "action" in data:
        action = int(data["action"])
    else:
        x = int(data["x"])
        y = int(data["y"])
        if not (0 <= x < size and 0 <= y < size):
            return jsonify({"error": "out of range"}), 400
        action = _action_from_xy(BOARD, x, y)

    valids = GAME.getValidMoves(BOARD)
    if action < 0 or action >= GAME.getActionSize() or int(valids[action]) == 0:
        return jsonify({"error": "invalid"}), 400

    _apply_action(action)
    _recompute_policy(append_curve=True)

    if _winner_from_board(BOARD) is None and MODE != "human_human" and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        _ai_move()

    return jsonify(_serialize_state())


@app.route("/ai_step", methods=["POST"])
def ai_step_api():
    if MODE != "ai_ai":
        return jsonify({"error": "invalid mode"}), 400

    if _winner_from_board(BOARD) is not None:
        return jsonify({"error": "game over"}), 400

    _ai_move()
    return jsonify(_serialize_state())


@app.route("/undo", methods=["POST"])
def undo_api():
    global MCTS_OBJ

    if not ACTION_HISTORY:
        return jsonify({"error": "no moves"}), 400

    _undo_once()
    if MODE != "human_human" and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        _undo_once()

    sims = getattr(MCTS_OBJ, "sims", get_search_config(ACTIVE_GAME_NAME).mcts_sims)
    MCTS_OBJ = _create_mcts(sims)
    _recompute_policy(append_curve=False)

    return jsonify(_serialize_state())


@app.route("/analyze", methods=["POST"])
def analyze_api():
    data = request.get_json(force=True)
    sims = int(data.get("sims", get_search_config(ACTIVE_GAME_NAME).mcts_sims))

    mcts = _create_mcts(sims)
    pi = mcts.get_action_probs(BOARD, temp=1, add_noise=False)

    s_root = GAME.stringRepresentation(BOARD)
    value = 0.0
    for action in range(GAME.getActionSize()):
        if (s_root, action) in mcts.Qsa:
            value += float(pi[action]) * float(mcts.Qsa[(s_root, action)])

    policy, pass_prob = _policy_to_payload(np.asarray(pi, dtype=np.float32), BOARD)
    return jsonify({
        "policy": policy,
        "pass_prob": pass_prob,
        "value": float(value * GAME.getCurrentPlayer(BOARD)),
    })


def _bootstrap() -> None:
    _activate_game(ACTIVE_GAME_NAME)

    default_model = _latest_model_for_game(ACTIVE_GAME_NAME)
    if default_model:
        _load_model(default_model)

    _reset_position(mode="human_ai", human_player=1, sims=None)


_bootstrap()


if __name__ == "__main__":
    app.run(debug=True)
