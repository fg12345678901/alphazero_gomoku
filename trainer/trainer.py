# trainer/trainer.py
from __future__ import annotations
import os, glob, time, pickle, json
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange

from config import *
from network.model import AlphaZeroNet
from trainer.dataset import ReplayBuffer
from trainer.arena import Arena
import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.net = AlphaZeroNet().to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.buffer = ReplayBuffer()
        self.step = 0

        self._load_latest_model()

    # ----------------- 主循环 ----------------- #
    def train(self, updates=TRAIN_UPDATES):
        loader = self.buffer.loader()
        it = iter(loader)
        for _ in trange(updates, desc="Training"):
            try:
                boards, target_pi, target_v = next(it)
            except StopIteration:
                it = iter(loader)
                boards, target_pi, target_v = next(it)

            boards, target_pi, target_v = boards.to(DEVICE), target_pi.to(DEVICE), target_v.to(DEVICE)
            self.optimizer.zero_grad()
            out_pi, out_v = self.net(boards)
            l_pi = -torch.mean(torch.sum(target_pi * torch.log_softmax(out_pi, dim=1), dim=1))
            l_v  = F.mse_loss(out_v, target_v)
            loss = l_pi + l_v
            loss.backward()
            self.optimizer.step()
            self.step += 1

        self._save_model()

    # ----------------- 评估与更替 ----------------- #
    def evaluate_and_update(self, num_games,
                            out: str | None = None,
                            no_update: bool = False):
        latest_path = self._latest_model_path()
        prev_path   = self._previous_model_path()
        if not prev_path:
            logger.info("No previous model, skipping arena.")
            return
        net_new = AlphaZeroNet().to(DEVICE)
        logger.info(f"Load NEW model: {latest_path}")
        net_new.load_state_dict(torch.load(latest_path, map_location=DEVICE))
        net_old = AlphaZeroNet().to(DEVICE)
        logger.info(f"Load OLD model: {prev_path}")
        net_old.load_state_dict(torch.load(prev_path, map_location=DEVICE))

        arena = Arena(net_new, net_old, num_games)
        n1, n2, d = arena.play()


        # ---- 可选保存单卡结果 ----
        if out:
            with open(out, "w") as fp:
                json.dump({"wins": n1, "losses": n2, "draws": d}, fp)
        if no_update:          # 并行模式下直接返回，聚合脚本再决定删不删
            return


        win_rate = n1 / (n1 + n2 + d)
        logger.info(f"Arena result new/old/draw = {n1}/{n2}/{d}, win_rate={win_rate:.2%}")
        if win_rate < EVAL_THRESHOLD:
            logger.info("New model rejected.")
            os.remove(latest_path)
        else:
            logger.info("New model accepted!")

    # ----------------- 模型管理 ----------------- #
    def _save_model(self):
        fname = os.path.join(MODEL_DIR, f"net_{int(time.time())}.pt")
        torch.save(self.net.state_dict(), fname)
        logger.info(f"Model saved to {fname}")

    def _latest_model_path(self):
        files = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
        return files[-1] if files else None

    def _previous_model_path(self):
        files = sorted(glob.glob(os.path.join(MODEL_DIR, "net_*.pt")))
        return files[-2] if len(files) >= 2 else None

    def _load_latest_model(self):
        path = self._latest_model_path()
        if path:
            self.net.load_state_dict(torch.load(path, map_location=DEVICE))
            logger.info(f"Loaded model {path}")
