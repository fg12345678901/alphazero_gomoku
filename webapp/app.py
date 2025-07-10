from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import glob, os

from gomoku.game import GomokuGame
from network.model import AlphaZeroNet
from mcts.mcts import MCTS
from config import DEVICE

app = Flask(__name__)

# ---------------- 工具函数 ----------------
def latest_model():
    files = sorted(glob.glob(os.path.join('models', 'net_*.pt')))
    return files[-1] if files else None

def evaluate(net, game, board):
    planes = game.getCanonicalForm(board, board.current_player)
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

# ---------------- 全局对局状态 ----------------

GAME = GomokuGame()
NET = AlphaZeroNet().to(DEVICE)
model_path = latest_model()
if model_path:
    NET.load_state_dict(torch.load(model_path, map_location=DEVICE))
NET.eval()

BOARD = GAME.getInitBoard()
MCTS_OBJ = MCTS(GAME, NET)
HISTORY = []
VALUE_CURVE = []
MODE = 'human_ai'  # human_ai, human_human, ai_ai
HUMAN_PLAYER = 1   # 1 黑, -1 白
POLICY = []        # 当前局面的网络落子概率

# ---------------- 路由 ----------------
@app.route('/')
def index():
    return render_template('index.html', size=GAME.size)

@app.route('/start', methods=['POST'])
def start_game():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, MODE, HUMAN_PLAYER, POLICY
    data = request.get_json(force=True)
    MODE = data.get('mode', 'human_ai')
    HUMAN_PLAYER = int(data.get('human_player', 1))
    BOARD = GAME.getInitBoard()
    MCTS_OBJ = MCTS(GAME, NET)
    HISTORY = []
    VALUE_CURVE = []
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value)
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()
    return jsonify(success=True,
                   board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   policy=POLICY,
                   value_curve=VALUE_CURVE)

@app.route('/state')
def get_state():
    winner = BOARD.get_winner()
    return jsonify(board=BOARD.board.tolist(),
                   current_player=int(BOARD.current_player),
                   history=HISTORY,
                   value_curve=VALUE_CURVE,
                   policy=POLICY,
                   winner=winner)

# 辅助函数：AI 落子
def ai_move():
    global BOARD, MCTS_OBJ, HISTORY, VALUE_CURVE, POLICY
    pi = MCTS_OBJ.get_action_probs(BOARD, temp=0)
    move = int(np.argmax(pi))
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value)
    POLICY = np.array(policy).reshape(GAME.size, GAME.size).tolist()

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
    move = GAME.coord_to_move(x, y)
    BOARD, _ = GAME.getNextState(BOARD, move)
    HISTORY.append(move)
    policy, value = evaluate(NET, GAME, BOARD)
    VALUE_CURVE.append(value)
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
    BOARD.undo_move()
    HISTORY.pop()
    if VALUE_CURVE:
        VALUE_CURVE.pop()
    # 重新评估当前局面
    MCTS_OBJ = MCTS(GAME, NET)
    policy, value = evaluate(NET, GAME, BOARD)
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

if __name__ == '__main__':
    app.run(debug=True)
