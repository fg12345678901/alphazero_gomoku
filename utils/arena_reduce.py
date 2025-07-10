# utils/arena_reduce.py
"""
用法：python utils/arena_reduce.py <tmp_dir>  <model_dir>
  - 统计 tmp_dir 下若干 result_*.json
  - 胜率 < EVAL_THRESHOLD 删掉 model_dir 最新 ckpt
"""
import sys, pathlib, os, json, glob, shutil, logging
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EVAL_THRESHOLD, MODEL_DIR

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
rate = wins / total if total else 0.0
logger.info(f"[Arena] {total} games — win {wins} / loss {losses} / draw {draws}"
      f"  →  {rate:.2%}")

models = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
latest = models[-1] if models else None
if latest and rate < EVAL_THRESHOLD:
    logger.info(f"[Arena] <{EVAL_THRESHOLD:.0%} → reject {os.path.basename(latest)}")
    os.remove(latest)
else:
    logger.info("[Arena] keep latest model")


shutil.rmtree(tmp, ignore_errors=True) # 彻底删除 arena_tmp 防止与下次结果混淆（还有另一重保险在eval_parallel.sh中）
