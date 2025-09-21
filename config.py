# config.py
import torch
import numpy as np

BOARD_SIZE = 15            # 浜斿瓙妫?15脳15
N_IN_ROW   = 5             # 杩炰簲鍗冲彲鑳?

# 缃戠粶杈撳叆鍘嗗彶姝ユ暟锛圓lphaZero 椋庢牸锛?
HISTORY_STEPS = 3          # 褰撳墠姝?鍓?N-1 姝?
INPUT_PLANES  = 2 * HISTORY_STEPS + 1

CHANNELS   = 256           # 鍗风Н閫氶亾
NUM_RES    = 15             # 娈嬪樊鍧楁暟閲?
MCTS_SIMS  = 1700          # 姣忔鎼滅储娆℃暟
CPUCT      = 2.5           # MCTS 鎺㈢储绯绘暟

BUFFER_SIZE      = 800_000   # 缁忛獙缂撳瓨涓婇檺
BATCH_SIZE       = 512
TRAIN_UPDATES    = 8000
LEARNING_RATE    = 8*1e-4
WEIGHT_DECAY     = 1e-4

# ==== Dirichlet 鍣０锛堟牴鑺傜偣鎺㈢储鐢級 ====
DIRICHLET_ALPHA  = 0.06 # 0.30  # 伪
DIRICHLET_EPS    = 0.25         # 蔚

SELFPLAY_TEMPERATURE = 1.0   # 鍓?N_TEMP_MOVES 姝ヤ娇鐢ㄩ珮娓╁害
N_TEMP_MOVES         = 10 #10

EVAL_GAMES     = 100
EVAL_THRESHOLD = 0.55        # 鈮?5% 鑳滅巼鍒欐帴鍙楁柊妯″瀷

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'    # 'cpu' or 'cuda'
MODEL_DIR = 'models'                                       # 淇濆瓨 ckpt
DATA_DIR  = 'data'                                         # 淇濆瓨 self鈥憄lay 鏍锋湰

# 鏃ュ織鐩稿叧
LOG_DIR   = "logs"
LOG_LEVEL = "DEBUG"          # enable verbose logging during dev
LOG_NAME  = "alphazero"     # 渚夸簬 grep/鍒嗘瀽
TB_DIR    = "tb"            # TensorBoard 鏃ュ織鐩綍

