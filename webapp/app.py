import os
import sys
import glob
from copy import deepcopy

from flask import Flask, render_template, request, jsonify
import numpy as np
import torch

# 让本地模块优先
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gomoku.game import GomokuGame
from network.model import AlphaZeroNet
from mcts.mcts import MCTS
from config import DEVICE, MCTS_SIMS

# 新增：OM 适配器（基于 ais-bench 的 InferSession）
try:
    from network.ascend_om_net_ais import AscendOMNetAIS
except Exception:
    AscendOMNetAIS = None  # 没装也不影响 .pt 路径

app = Flask(__name__)

# ---------------- 工具函数 ----------------
def is_om(path: str) -> bool:
    return path is not None and path.lower().endswith(".om")

def list_models():
    """列出 models/ 目录下的 .pt 和 .om，均按数值或 mtime 排序"""
    pt_files = glob.glob(os.path.join('models', 'net_*.pt'))
    om_files = glob.glob(os.path.join('models', '*.om'))

    def _key_num_or_mtime(f):
        base = os.path.splitext(os.path.basename(f))[0]
        # 尝试从 net_123 提取 123 作为排序键；失败则用修改时间
        try:
            if "_" in base:
                return int(base.split("_")[-1])
        except Exception:
            pass
        return int(os.path.getmtime(f))

    files = pt_files + om_files
    files.sort(key=_key_num_or_mtime)
    return [os.path.basename(f) for f in files]

def latest_model():
    """返回最新的 .om 或 .pt（优先 .om，没有再用 .pt）"""
    names = list_models()
    if not names:
        return None
    # 优先找 .om
    for name in reversed(names):
        if name.lower().endswith(".om"):
            return os.path.join('models', name)
    # 退而求其次 .pt
    return os.path.join('models', names[-1])

def load_net_from_path(path: str):
    """
    根据路径加载模型：
     - *.om -> AscendOMNetAIS
     - *.pt -> AlphaZeroNet + load_state_dict
    返回: (net_obj, model_path_str)
    """
    if path is None:
        # 没有模型时，默认创建空的 PyTorch 网络（兼容旧流程）
        net = AlphaZeroNet().to(DEVICE).eval()
        return net, None

    # 补全相对路径
    if os.path.basename(path) == path:
        path = os.path.join('models', path)

    if is_om(path):
        if AscendOMNetAIS is None:
            raise RuntimeError("需要 AscendOMNetAIS（ais-bench）适配器，但未找到。请确保已添加 network/ascend_om_net_ais.py 并安装依赖。")
        # 设备号可从环境变量覆盖（默认为 0）
        device_id = int(os.environ.get("DEVICE_ID", "0"))
        net = AscendOMNetAIS(path, device_id=device_id)  # 你已经小测试验证过
        return net, path
    else:
        # PyTorch .pt
        net = AlphaZeroNet().to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        net.load_state_dict(state)
        net.eval()
        return net, path

def evaluate(net, game, board):
    """
    对当前局面跑一次前向，返回 (policy 概率数组, value 浮点数)
    - 对 OM 适配器：输入要求 CPU float32 [1,7,15,15]，返回 torch.Tensor
    - 对 .pt 网络：保持原有逻辑
    """
    planes = game.getCanonicalForm(board, board.current_player)  # [7,15,15]
    # 统一：先在 CPU 构造 float32，再按需放到 DEVICE（仅对 .pt 生效）
    x = torch.tensor(planes, dtype=torch.float32).unsqueeze(0)   # [1,7,15,15]
    # Ascend 适配器是 CPU 前向；PyTorch 模型才需要 to(DEVICE)
    if hasattr(net, "to"):  # 粗略判定是 torch.nn.Module
        x = x.to(DEVICE)

    with torch.no_grad():
        policy_logits, value = net(x)

    # -> 概率并应用合法位掩码
    policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]   # [225]
    valids = game.getValidMoves(board)                              # [225] in {0,1}
    policy = policy * valids
    s = policy.sum()
    if s > 0:
        policy /= s
    else:
        policy = valids / max(valids.sum(), 1)
    return policy.tolist(), float(value.detach().cpu().numpy().reshape(-1)[0])

# ---------------- 全局对局状态 ----------------

GAME = GomokuGame()
CURRENT_MODEL = latest_model()

NET, CURRENT_MODEL = load_net_from_path(CURRENT_MODEL)

# Ascend OM 推理只支持创建它的线程里执行，
# 因此在开发服务器上要关闭 Flask 的多线程以避免 ctx is NULL。
USE_THREADED = True
if AscendOMNetAIS is not None and isinstance(NET, AscendOMNetAIS):
    USE_THREADED = False

BOARD = GAME.getInitBoard()
MCTS_OBJ = MCTS(GAME, NET)
HISTORY = []
VALUE_CURVE = []
MODE = 'human_ai'  # human_ai, human_human, ai_ai
HUMAN_PLAYER = 1   # 1 黑, -1 白
POLICY = []        # 当前局面的网络落子概率（可视化用）

# ---------------- 路由 ----------------
@app.route('/')
def index():
    return render_template('index.html', size=GAME.size)

@app.route('/models')
def models_list():
    return jsonify(models=list_models(),
                   current=os.path.basename(CURRENT_MODEL) if CURRENT_MODEL else None)

@app.route('/start', methods=['POST'])
def start_game():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, MODE, HUMAN_PLAYER, POLICY, CURRENT_MODEL, NET
    data = request.get_json(force=True)
    MODE = data.get('mode', 'human_ai')
    HUMAN_PLAYER = int(data.get('human_player', 1))
    sims = int(data.get('mcts_sims', MCTS_SIMS))
    requested_model = data.get('model')

    # 选择模型：None/''/'latest' -> 自动挑；否则按文件名从 models/ 取
    if requested_model in (None, '', 'latest'):
        path = latest_model()
    else:
        path = requested_model
        if os.path.basename(path) == path:
            path = os.path.join('models', path)

    # 如果和当前不同，则切换网络（兼容 .om / .pt）
    if path and path != CURRENT_MODEL and os.path.isfile(path):
        NET, CURRENT_MODEL = load_net_from_path(path)

    # 重置对局
    BOARD = GAME.getInitBoard()
    MCTS_OBJ = MCTS(GAME, NET, sims)
    HISTORY = []
    VALUE_CURVE = []

    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * BOARD.current_player)
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

    # 如果是人机/AI先手，自动下一步
    if MODE != 'human_human' and BOARD.current_player != HUMAN_PLAYER:
        ai_move()

    return jsonify(success=True,
                   board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   history=HISTORY,
                   policy=POLICY,
                   value_curve=VALUE_CURVE,
                   model=os.path.basename(CURRENT_MODEL) if CURRENT_MODEL else None)

@app.route('/state')
def get_state():
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   model=os.path.basename(CURRENT_MODEL) if CURRENT_MODEL else None,
                   winner=winner)

# 对当前棋盘进行网络前向，返回先验概率
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

# 辅助函数：AI 落子
def ai_move():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, POLICY
    pi = MCTS_OBJ.get_action_probs(BOARD, temp=0)
    move = int(np.argmax(pi))
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * BOARD.current_player)
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

@app.route('/ai_step', methods=['POST'])
def ai_step():
    """在 AI 对战模式下执行一步 AI 行棋"""
    global BOARD, HISTORY, VALUE_CURVE, POLICY
    if MODE != 'ai_ai':
        return jsonify(error='invalid mode'), 400
    if BOARD.get_winner() is not None:
        return jsonify(error='game over'), 400
    ai_move()
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
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
    if MODE != 'human_human' and BOARD.current_player != HUMAN_PLAYER:
        return jsonify(error='not your turn'), 400
    move = BOARD.coord_to_move(x, y)
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value * BOARD.current_player)
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

    winner = BOARD.get_winner()
    # 如果轮到AI
    if winner is None and MODE != 'human_human' and BOARD.current_player != HUMAN_PLAYER:
        ai_move()
        winner = BOARD.get_winner()

    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

@app.route('/undo', methods=['POST'])
def undo():
    global BOARD, HISTORY, VALUE_CURVE, POLICY, MCTS_OBJ
    if not HISTORY:
        return jsonify(error='no moves'), 400

    # 撤销一步
    HISTORY.pop()
    BOARD.undo_move()
    if VALUE_CURVE:
        VALUE_CURVE.pop()

    # 如果是人机模式并且轮到 AI，下退一步以回到玩家手动决策前的局面
    if MODE != 'human_human' and BOARD.current_player != HUMAN_PLAYER and HISTORY:
        HISTORY.pop()
        BOARD.undo_move()
        if VALUE_CURVE:
            VALUE_CURVE.pop()

    # 重新评估当前局面
    MCTS_OBJ = MCTS(GAME, NET, getattr(MCTS_OBJ, 'sims', MCTS_SIMS))
    policy, value = evaluate(NET, GAME, BOARD)
    value = value * BOARD.current_player
    if VALUE_CURVE:
        VALUE_CURVE[-1] = value
    else:
        VALUE_CURVE.append(value)

    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

# 使用 MCTS 对当前局面进行深入搜索，返回搜索概率和估值
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
    return jsonify(policy=policy, value=float(value * BOARD.current_player))

if __name__ == '__main__':
    # Ascend OM 模型需要单线程环境，其他模型仍可用多线程
    # 生产建议用：gunicorn -w 1 --threads 1 -b 0.0.0.0:8080 app:app（Ascend）
    app.run(host="0.0.0.0", port=8080, threaded=USE_THREADED)
