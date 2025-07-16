# trainer/trainer.py
from __future__ import annotations
import os, glob, time, pickle, json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import trange

from config import *
from network.model import AlphaZeroNet
from trainer.dataset import ReplayBuffer
from trainer.arena import Arena
import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, distributed: bool = False, local_rank: int | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.distributed = distributed

        if distributed:
            if local_rank is None:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(DEVICE)

        self.device = device
        self.net = AlphaZeroNet().to(self.device)

        if distributed:
            self.net = nn.parallel.DistributedDataParallel(
                self.net, device_ids=[local_rank], output_device=local_rank
            )
            logger.info(
                f"DDP enabled · rank {dist.get_rank()}/{dist.get_world_size()}"
            )
        elif device.type == "cuda" and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            logger.info(f"DataParallel enabled · {torch.cuda.device_count()} GPUs")

        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.buffer = ReplayBuffer()
        self.step = 0

        self._load_latest_model()

    # ----------------- 主循环 ----------------- #
    def train(self, updates=TRAIN_UPDATES):
        batch_size = BATCH_SIZE
        if self.distributed:
            world_size = dist.get_world_size()
            batch_size = max(1, BATCH_SIZE // world_size)
            if BATCH_SIZE % world_size != 0 and dist.get_rank() == 0:
                logger.warning(
                    "BATCH_SIZE %d not divisible by world_size %d; using per-rank batch size %d",
                    BATCH_SIZE,
                    world_size,
                    batch_size,
                )
        loader, sampler = self.buffer.loader(distributed=self.distributed, batch_size=batch_size)
        it = iter(loader)
        progress = trange(
            updates,
            desc="Training",
            disable=self.distributed and dist.get_rank() != 0,
        )
        for step in progress:
            if sampler:
                sampler.set_epoch(step)
            try:
                boards, target_pi, target_v = next(it)
            except StopIteration:
                it = iter(loader)
                boards, target_pi, target_v = next(it)

            boards, target_pi, target_v = (
                boards.to(self.device),
                target_pi.to(self.device),
                target_v.to(self.device),
            )
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
        net_new.eval()
        net_old = AlphaZeroNet().to(DEVICE)
        logger.info(f"Load OLD model: {prev_path}")
        net_old.load_state_dict(torch.load(prev_path, map_location=DEVICE))
        net_old.eval()

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
        if self.distributed and dist.get_rank() != 0:
            return
        fname = os.path.join(MODEL_DIR, f"net_{int(time.time())}.pt")
        state = (
            self.net.module.state_dict() if hasattr(self.net, "module") else self.net.state_dict()
        )
        torch.save(state, fname)

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
            # self.net.load_state_dict(torch.load(path, map_location=DEVICE))
            target = self.net.module if hasattr(self.net, "module") else self.net
            target.load_state_dict(torch.load(path, map_location=self.device))
            
            logger.info(f"Loaded model {path}")
