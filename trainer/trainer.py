# trainer/trainer.py
from __future__ import annotations
import os, glob, time, pickle, json, csv, math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        self.writer = None

        self._load_latest_model()

    # ----------------- 主循环 ----------------- #
    def train(self, updates=TRAIN_UPDATES):
        ts = int(time.time())
        if (not self.distributed) or dist.get_rank() == 0:
            os.makedirs(TB_DIR, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(TB_DIR, f"net_{ts}"))

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
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=updates,
            eta_min=LEARNING_RATE / 100,
        )
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
            scheduler.step()
            self.step += 1
            if self.writer:
                self.writer.add_scalar("loss/total", loss.item(), self.step)
                self.writer.add_scalar("loss/value", l_v.item(), self.step)
                self.writer.add_scalar("loss/policy", l_pi.item(), self.step)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.step)

        if self.writer:
            self.writer.flush()
        self._save_model(ts)
        if self.writer:
            self.writer.close()

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


        win_rate = (n1 + 0.5 * d) / (n1 + n2 + d)
        logger.info(
            f"Arena result new/old/draw = {n1}/{n2}/{d}, win_rate={win_rate:.2%}"
        )

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

        new_elo = prev_elo + elo_diff if win_rate >= EVAL_THRESHOLD else prev_elo
        accepted = win_rate >= EVAL_THRESHOLD

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
                    os.path.basename(latest_path),
                    n1,
                    n2,
                    d,
                    f"{win_rate:.4f}",
                    f"{new_elo:.2f}",
                    int(accepted),
                ]
            )

        eval_tb_dir = os.path.join(TB_DIR, "eval")
        os.makedirs(eval_tb_dir, exist_ok=True)
        writer = SummaryWriter(eval_tb_dir)

        now = int(time.time())
        # evaluation count vs. Elo
        writer.add_scalar("elo_by_step", new_elo, eval_step)
        writer.add_scalar("win_rate_by_step", win_rate, eval_step)
        # walltime vs. Elo allows viewing strength growth over real time
        writer.add_scalar("elo_by_time", new_elo, now)
        writer.add_scalar("win_rate_by_time", win_rate, now)

        writer.flush()
        writer.close()

        if not accepted:
            logger.info("New model rejected.")
            os.remove(latest_path)
        else:
            logger.info("New model accepted!")

    # ----------------- 模型管理 ----------------- #
    def _save_model(self, timestamp: int | None = None):
        if self.distributed and dist.get_rank() != 0:
            return
        if timestamp is None:
            timestamp = int(time.time())
        fname = os.path.join(MODEL_DIR, f"net_{timestamp}.pt")
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
