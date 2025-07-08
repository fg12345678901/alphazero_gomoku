# main.py
import argparse, glob, os
from selfplay.selfplay import SelfPlayWorker
from trainer.trainer import Trainer
from config import *
from logging_setup import setup_logging
logger = setup_logging()


def latest_model():
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
    return files[-1] if files else None

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sp = sub.add_parser("selfplay")
    p_sp.add_argument("--num-games", type=int, default=100)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--updates", type=int, default=TRAIN_UPDATES)

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--num-games", type=int, default=EVAL_GAMES)

    args = parser.parse_args()

    if args.cmd == "selfplay":
        sp = SelfPlayWorker(latest_model(), DATA_DIR, args.num_games)
        sp.run()

    elif args.cmd == "train":
        tr = Trainer()
        tr.train(args.updates)

    elif args.cmd == "evaluate":
        tr = Trainer()
        tr.evaluate_and_update(num_games=args.num_games)

if __name__ == "__main__":
    logger.info(f"using device: {DEVICE}")
    main()


"""
# 安装依赖
pip install -r requirements.txt

# 生成 100 局自我对弈数据
python main.py selfplay --num-games 100

# 用最近的缓存训练 2000 次梯度更新
python main.py train --updates 2000

# 新旧模型评测（400 局；胜率≥55% 则更新）
python main.py evaluate --num-games 400
"""