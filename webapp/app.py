import os
import sys
import threading

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import glob
from copy import deepcopy

# Ensure local modules take precedence over any installed packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gomoku.game import GomokuGame
from mcts.mcts import MCTS
from config import DEVICE, MCTS_SIMS, get_rule_config
from network.checkpoint import build_model_for_game, load_checkpoint, validate_checkpoint_meta

app = Flask(__name__)

# ---------------- 宸ュ叿鍑芥暟 ----------------
def latest_model():
    files = glob.glob(os.path.join('models', 'net_*.pt'))
    if not files:
        return None
    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[1]))
    return files[-1]

def list_models():
    files = glob.glob(os.path.join('models', 'net_*.pt'))
    files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[1]))
    return [os.path.basename(f) for f in files]


def _resolve_model_path(path):
    if not path:
        return None
    if os.path.basename(path) == path:
        return os.path.join('models', path)
    return path


def load_model(path):
    global CURRENT_MODEL_PATH, _LOADED_MODEL_PATH
    resolved = _resolve_model_path(path)
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

def evaluate(net, game, board):
    current_player = game.getCurrentPlayer(board)
    planes = game.getCanonicalForm(board, current_player)
    x = torch.tensor(planes, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        policy, value = net(x)
    policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
    valids = game.getValidMoves(board)
    policy = policy * valids
    s = policy.sum()
    if s > 0:
        policy /= s
    else:
        policy = valids / valids.sum()
    return policy.tolist(), float(value.item())

# ---------------- 鍏ㄥ眬瀵瑰眬鐘舵€?----------------

_RULES = get_rule_config('gomoku')
GAME = GomokuGame(size=_RULES.board_size, n_in_row=_RULES.n_in_row or 5, history_steps=_RULES.history_steps)
NET = build_model_for_game(GAME, device=DEVICE)
MODEL_LOCK = threading.Lock()
CURRENT_MODEL_PATH = latest_model()
_LOADED_MODEL_PATH = None
NET.eval()

BOARD = GAME.getInitBoard()
MCTS_OBJ = MCTS(GAME, NET)
HISTORY = []
VALUE_CURVE = []
MODE = 'human_ai'  # human_ai, human_human, ai_ai
HUMAN_PLAYER = 1   # 1 榛? -1 鐧?
POLICY = []        # 褰撳墠灞€闈㈢殑缃戠粶钀藉瓙姒傜巼

# ---------------- 璺敱 ----------------
@app.route('/')
def index():
    return render_template('index.html', size=GAME.size)

@app.route('/models')
def models_list():
    current_name = os.path.basename(CURRENT_MODEL_PATH) if CURRENT_MODEL_PATH else None
    return jsonify(models=list_models(), current=current_name)

@app.route('/start', methods=['POST'])
def start_game():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, MODE, HUMAN_PLAYER, POLICY
    data = request.get_json(force=True)
    MODE = data.get('mode', 'human_ai')
    HUMAN_PLAYER = int(data.get('human_player', 1))
    sims = int(data.get('mcts_sims', MCTS_SIMS))
    requested_model = data.get('model')
    if requested_model in (None, '', 'latest'):
        requested_model = latest_model()
    if requested_model:
        load_model(requested_model)
    BOARD = GAME.getInitBoard()
    MCTS_OBJ = MCTS(GAME, NET, sims)
    HISTORY = []
    VALUE_CURVE = []
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * GAME.getCurrentPlayer(BOARD))
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()
    if MODE != 'human_human' and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        ai_move()
    return jsonify(success=True,
                   board=BOARD.board.tolist(),
                   current_player=int(GAME.getCurrentPlayer(BOARD)),
                   history=HISTORY,
                   policy=POLICY,
                   value_curve=VALUE_CURVE,
                   model=os.path.basename(CURRENT_MODEL_PATH) if CURRENT_MODEL_PATH else None)

@app.route('/state')
def get_state():
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(GAME.getCurrentPlayer(BOARD)),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   model=os.path.basename(CURRENT_MODEL_PATH) if CURRENT_MODEL_PATH else None,
                   winner=winner)

# 瀵瑰綋鍓嶆鐩樿繘琛岀綉缁滃墠鍚戯紝杩斿洖鍏堥獙姒傜巼
@app.route('/prior', methods=['POST'])
def calc_prior():
    data = request.get_json(force=True)
    x = int(data['x'])
    y = int(data['y'])
    if BOARD.board[x, y] != 0:
        return jsonify(error='invalid'), 400
    move = BOARD.coord_to_move(x, y)
    tmp_board, _ = GAME.getNextState(BOARD, move)
    policy, _ = evaluate(NET, GAME, tmp_board)
    return jsonify(policy=np.array(policy).reshape(GAME.size, GAME.size).tolist())

# 杈呭姪鍑芥暟锛欰I 钀藉瓙
def ai_move():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, POLICY
    pi = MCTS_OBJ.get_action_probs(BOARD, temp=0)
    move = int(np.argmax(pi))
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * GAME.getCurrentPlayer(BOARD))
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

@app.route('/ai_step', methods=['POST'])
def ai_step():
    """鍦?AI 瀵规垬妯″紡涓嬫墽琛屼竴姝?AI 琛屾"""
    global BOARD, HISTORY, VALUE_CURVE, POLICY
    if MODE != 'ai_ai':
        return jsonify(error='invalid mode'), 400
    if BOARD.get_winner() is not None:
        return jsonify(error='game over'), 400
    ai_move()
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(GAME.getCurrentPlayer(BOARD)),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

@app.route('/move', methods=['POST'])
def make_move():
    global BOARD, HISTORY, VALUE_CURVE, POLICY
    data = request.get_json(force=True)
    x = int(data['x'])
    y = int(data['y'])
    if BOARD.board[x, y] != 0:
        return jsonify(error='invalid'), 400
    if MODE != 'human_human' and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        return jsonify(error='not your turn'), 400
    move = BOARD.coord_to_move(x, y)
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * GAME.getCurrentPlayer(BOARD))
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

    winner = BOARD.get_winner()
    # 濡傛灉杞埌AI
    if winner is None and MODE != 'human_human' and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER:
        ai_move()
        winner = BOARD.get_winner()

    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(GAME.getCurrentPlayer(BOARD)),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

@app.route('/undo', methods=['POST'])
def undo():
    global BOARD, HISTORY, VALUE_CURVE, POLICY, MCTS_OBJ
    if not HISTORY:
        return jsonify(error='no moves'), 400

    # 鎾ら攢涓€姝?
    HISTORY.pop()
    BOARD.undo_move()
    if VALUE_CURVE:
        VALUE_CURVE.pop()

    # 濡傛灉鏄汉鏈烘ā寮忓苟涓旇疆鍒?AI锛屼笅閫€涓€姝ヤ互鍥炲埌鐜╁鎵嬪姩鍐崇瓥鍓嶇殑灞€闈?
    if MODE != 'human_human' and GAME.getCurrentPlayer(BOARD) != HUMAN_PLAYER and HISTORY:
        HISTORY.pop()
        BOARD.undo_move()
        if VALUE_CURVE:
            VALUE_CURVE.pop()

    # 閲嶆柊璇勪及褰撳墠灞€闈?
    MCTS_OBJ = MCTS(GAME, NET, getattr(MCTS_OBJ, 'sims', MCTS_SIMS))
    policy, value = evaluate(NET, GAME, BOARD)
    value = value * GAME.getCurrentPlayer(BOARD)
    if VALUE_CURVE:
        VALUE_CURVE[-1] = value
    else:
        VALUE_CURVE.append(value)

    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(GAME.getCurrentPlayer(BOARD)),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

# 浣跨敤 MCTS 瀵瑰綋鍓嶅眬闈㈣繘琛屾繁鍏ユ悳绱紝杩斿洖鎼滅储姒傜巼鍜屼及鍊?
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    sims = int(data.get('sims', MCTS_SIMS))
    mcts = MCTS(GAME, NET, sims)
    pi = mcts.get_action_probs(BOARD, temp=1)
    s_root = GAME.stringRepresentation(BOARD)
    value = 0.0
    for a in range(GAME.getActionSize()):
        if (s_root, a) in mcts.Qsa:
            value += pi[a] * mcts.Qsa[(s_root, a)]
    policy = np.array(pi).reshape(GAME.size, GAME.size).tolist()
    return jsonify(policy=policy, value=float(value * GAME.getCurrentPlayer(BOARD)))

if __name__ == '__main__':
    app.run(debug=True)

