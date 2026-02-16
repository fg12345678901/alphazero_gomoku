# utils/arena_reduce.py
"""
Usage:
  python utils/arena_reduce.py <tmp_dir> [--game gomoku]

Aggregate parallel arena JSON outputs and append metrics to Elo history.
In AlphaZero mode, this script never deletes model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pathlib
import shutil
import sys
import time
import logging

from torch.utils.tensorboard import SummaryWriter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import GAME_NAME
from logging_setup import setup_logging
from runtime_paths import resolve_runtime_paths

setup_logging()
logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_dir", help="folder containing result_*.json files")
    parser.add_argument("--game", default=GAME_NAME, help=f"game type (default: {GAME_NAME})")
    return parser.parse_args()


def main():
    args = _parse_args()
    paths = resolve_runtime_paths(args.game)

    wins = losses = draws = 0
    for path in glob.glob(os.path.join(args.tmp_dir, "*.json")):
        with open(path, "r") as fp:
            record = json.load(fp)
        wins += record["wins"]
        losses += record["losses"]
        draws += record["draws"]

    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total if total else 0.0
    logger.info(
        "[Arena] %d games | win %d / loss %d / draw %d -> %.2f%%",
        total,
        wins,
        losses,
        draws,
        win_rate * 100,
    )

    models = sorted(glob.glob(os.path.join(paths.model_dir, "net_*.pt")))
    if total == 0 or len(models) < 2:
        logger.info("[Arena] no previous model or no games played, skip ELO update")
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
        return

    latest = models[-1]
    os.makedirs(paths.log_dir, exist_ok=True)
    hist_path = os.path.join(paths.log_dir, "elo_history.csv")
    prev_elo = 1000.0
    eval_step = 0
    if os.path.exists(hist_path):
        with open(hist_path, "r", newline="") as fp:
            rows = list(csv.reader(fp))
            if len(rows) > 1:
                prev_elo = float(rows[-1][6])
                eval_step = len(rows) - 1

    if 0 < win_rate < 1:
        clamped = min(max(win_rate, 1e-6), 1 - 1e-6)
        elo_diff = 400 * math.log10(clamped / (1 - clamped))
    else:
        elo_diff = 0.0
    new_elo = prev_elo + elo_diff

    with open(hist_path, "a", newline="") as fp:
        writer = csv.writer(fp)
        if fp.tell() == 0:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "wins",
                    "losses",
                    "draws",
                    "win_rate",
                    "elo",
                    "accepted",
                ]
            )
        writer.writerow(
            [
                int(time.time()),
                os.path.basename(latest),
                wins,
                losses,
                draws,
                f"{win_rate:.4f}",
                f"{new_elo:.2f}",
                1,  # kept for schema compatibility; AZ mode has no gating
            ]
        )

    eval_tb_dir = os.path.join(paths.tb_dir, "eval")
    os.makedirs(eval_tb_dir, exist_ok=True)
    tb = SummaryWriter(eval_tb_dir)
    now = int(time.time())
    tb.add_scalar("elo_by_step", new_elo, eval_step)
    tb.add_scalar("win_rate_by_step", win_rate, eval_step)
    tb.add_scalar("elo_by_time", new_elo, now)
    tb.add_scalar("win_rate_by_time", win_rate, now)
    tb.flush()
    tb.close()

    shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
