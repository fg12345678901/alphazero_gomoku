# utils/arena_reduce.py
"""
用法：python utils/arena_reduce.py <tmp_dir>
  - 统计 tmp_dir 下若干 result_*.json
  - 胜率 < EVAL_THRESHOLD 删掉最新模型
  - 记录 Elo 到 logs/elo_history.csv 并写入 TensorBoard
"""
import sys, pathlib, os, json, glob, shutil, logging, csv, math, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_THRESHOLD, MODEL_DIR, LOG_DIR, TB_DIR
from torch.utils.tensorboard import SummaryWriter

from logging_setup import setup_logging
setup_logging()
logger = logging.getLogger(__name__) # 该脚本不经过main.py被调用，所以需要setup_logging()

tmp = sys.argv[1]
wins = losses = draws = 0
for f in glob.glob(os.path.join(tmp, "*.json")):
    r = json.load(open(f))
    wins   += r["wins"]
    losses += r["losses"]
    draws  += r["draws"]

total = wins + losses + draws
win_rate = (wins + 0.5 * draws) / total if total else 0.0
logger.info(f"[Arena] {total} games — win {wins} / loss {losses} / draw {draws}  →  {win_rate:.2%}")

models = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
latest = models[-1] if models else None
accepted = win_rate >= EVAL_THRESHOLD
if latest:
    if accepted:
        logger.info("[Arena] keep latest model")
    else:
        logger.info(f"[Arena] <{EVAL_THRESHOLD:.0%} → reject {os.path.basename(latest)}")

    # ---- Elo rating calculation & logging ----
    os.makedirs(LOG_DIR, exist_ok=True)
    hist_path = os.path.join(LOG_DIR, "elo_history.csv")
    prev_elo = 1000.0
    eval_step = 0
    if os.path.exists(hist_path):
        with open(hist_path, "r", newline="") as fp:
            rows = list(csv.reader(fp))
            if len(rows) > 1:
                last = rows[-1]
                prev_elo = float(last[6])
                eval_step = len(rows) - 1

    if 0 < win_rate < 1:
        elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
    else:
        elo_diff = 0.0

    new_elo = prev_elo + elo_diff if accepted else prev_elo

    with open(hist_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if fp.tell() == 0:
            writer.writerow([
                "timestamp",
                "model",
                "wins",
                "losses",
                "draws",
                "win_rate",
                "elo",
                "accepted",
            ])
        writer.writerow([
            int(time.time()),
            os.path.basename(latest),
            wins,
            losses,
            draws,
            f"{win_rate:.4f}",
            f"{new_elo:.2f}",
            int(accepted),
        ])

    eval_tb_dir = os.path.join(TB_DIR, "eval")
    os.makedirs(eval_tb_dir, exist_ok=True)
    tb = SummaryWriter(eval_tb_dir)
    now = int(time.time())
    tb.add_scalar("elo_by_step", new_elo, eval_step)
    tb.add_scalar("win_rate_by_step", win_rate, eval_step)
    tb.add_scalar("elo_by_time", new_elo, now)
    tb.add_scalar("win_rate_by_time", win_rate, now)
    tb.flush()
    tb.close()

    if not accepted:
        os.remove(latest)


shutil.rmtree(tmp, ignore_errors=True) # 彻底删除 arena_tmp 防止与下次结果混淆（还有另一重保险在eval_parallel.sh中）
