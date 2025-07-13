# config.py
import torch


BOARD_SIZE = 15            # 五子棋 15×15
N_IN_ROW   = 5             # 连五即可胜
CHANNELS   = 256           # 卷积通道
NUM_RES    = 15             # 残差块数量
MCTS_SIMS  = 400          # 每步搜索次数
CPUCT      = 2.5           # MCTS 探索系数

BUFFER_SIZE      = 300_000   # 经验缓存上限
BATCH_SIZE       = 512
TRAIN_UPDATES    = 8000
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 2e-4

# ==== Dirichlet 噪声（根节点探索用） ====
DIRICHLET_ALPHA  = 0.06 # 0.30  # α
DIRICHLET_EPS    = 0.25         # ε

SELFPLAY_TEMPERATURE = 1.0   # 前 N_TEMP_MOVES 步使用高温度
N_TEMP_MOVES         = 25 #10

EVAL_GAMES     = 100
EVAL_THRESHOLD = 0.518        # ≥51.8% 胜率则接受新模型

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'    # 'cpu' or 'cuda'
MODEL_DIR = 'models'                                       # 保存 ckpt
DATA_DIR  = 'data'                                         # 保存 self‑play 样本

# 日志相关
LOG_DIR   = "logs"
LOG_LEVEL = "INFO"          # 开发阶段可改成 "DEBUG"
LOG_NAME  = "alphazero"     # 便于 grep/分析
