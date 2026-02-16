from __future__ import annotations

import csv
import glob
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    CHANNELS,
    DEVICE,
    GAME_NAME,
    LEARNING_RATE,
    NUM_RES,
    TRAIN_UPDATES,
    WEIGHT_DECAY,
    get_search_config,
)
from games.base import GameLike
from games.registry import create_game, normalize_game_name
from network.checkpoint import (
    build_model_for_game,
    checkpoint_meta,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_meta,
)
from runtime_paths import RuntimePaths, resolve_runtime_paths
from trainer.arena import Arena
from trainer.dataset import ReplayBuffer

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        distributed: bool = False,
        local_rank: int | None = None,
        game: GameLike | None = None,
        game_name: str = GAME_NAME,
        paths: RuntimePaths | None = None,
    ):
        self.distributed = distributed
        self.game_name = normalize_game_name(game_name)
        self.game = game if game is not None else create_game(self.game_name)
        self.game_spec = self.game.getGameSpec()
        self.search_cfg = get_search_config(self.game_name)
        self.paths = paths if paths is not None else resolve_runtime_paths(self.game_name)

        os.makedirs(self.paths.model_dir, exist_ok=True)
        os.makedirs(self.paths.data_dir, exist_ok=True)

        if distributed:
            if local_rank is None:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(DEVICE)

        self.net = build_model_for_game(self.game, device=self.device)
        if distributed:
            self.net = nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[local_rank],
                output_device=local_rank,
            )
            logger.info(
                "DDP enabled | rank %s/%s",
                dist.get_rank(),
                dist.get_world_size(),
            )
        elif self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            logger.info("DataParallel enabled | %s GPUs", torch.cuda.device_count())

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.buffer = ReplayBuffer(
            data_dir=self.paths.data_dir,
            buffer_size=BUFFER_SIZE,
            default_batch_size=BATCH_SIZE,
        )
        self.step = 0
        self.writer: SummaryWriter | None = None

        self._load_latest_model()

    def train(self, updates: int = TRAIN_UPDATES):
        if len(self.buffer) == 0:
            logger.warning(
                "Replay buffer is empty under %s, skip training.",
                self.paths.data_dir,
            )
            return

        ts = int(time.time())
        if (not self.distributed) or dist.get_rank() == 0:
            os.makedirs(self.paths.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.paths.tb_dir, f"net_{ts}"))

        batch_size = BATCH_SIZE
        if self.distributed:
            world_size = dist.get_world_size()
            batch_size = max(1, BATCH_SIZE // world_size)
            if BATCH_SIZE % world_size != 0 and dist.get_rank() == 0:
                logger.warning(
                    "BATCH_SIZE=%d not divisible by world_size=%d; per-rank batch_size=%d",
                    BATCH_SIZE,
                    world_size,
                    batch_size,
                )

        loader, sampler = self.buffer.loader(
            distributed=self.distributed,
            batch_size=batch_size,
        )
        it = iter(loader)
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=updates,
            eta_min=LEARNING_RATE / 100,
        )
        progress = trange(
            updates,
            desc=f"Training[{self.game_name}]",
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

            boards = boards.to(self.device)
            target_pi = target_pi.to(self.device)
            target_v = target_v.to(self.device)

            self.optimizer.zero_grad()
            out_pi, out_v = self.net(boards)
            l_pi = -torch.mean(
                torch.sum(target_pi * torch.log_softmax(out_pi, dim=1), dim=1)
            )
            l_v = F.mse_loss(out_v, target_v)
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

    def evaluate_and_update(
        self,
        num_games: int | None = None,
        out: str | None = None,
        no_update: bool = False,
    ):
        if num_games is None:
            num_games = self.search_cfg.eval_games

        latest_path = self._latest_model_path()
        prev_path = self._previous_model_path()
        if not latest_path or not prev_path:
            logger.info("No previous model under %s, skipping arena.", self.paths.model_dir)
            return

        net_new = build_model_for_game(self.game, device=DEVICE)
        net_old = build_model_for_game(self.game, device=DEVICE)
        logger.info("Load NEW model: %s", latest_path)
        logger.info("Load OLD model: %s", prev_path)

        sd_new, meta_new = load_checkpoint(latest_path, map_location=DEVICE)
        sd_old, meta_old = load_checkpoint(prev_path, map_location=DEVICE)
        validate_checkpoint_meta(meta_new, self.game_spec)
        validate_checkpoint_meta(meta_old, self.game_spec)

        net_new.load_state_dict(sd_new)
        net_old.load_state_dict(sd_old)
        net_new.eval()
        net_old.eval()

        arena = Arena(
            game=self.game,
            net1=net_new,
            net2=net_old,
            games=num_games,
            show_progress=not no_update,
            game_name=self.game_name,
            search_cfg=self.search_cfg,
        )
        n1, n2, d = arena.play()

        if out:
            with open(out, "w") as fp:
                json.dump({"wins": n1, "losses": n2, "draws": d}, fp)
        if no_update:
            return

        total = n1 + n2 + d
        win_rate = (n1 + 0.5 * d) / total if total else 0.0
        logger.info(
            "Arena result new/old/draw = %d/%d/%d, win_rate=%.2f%%",
            n1,
            n2,
            d,
            win_rate * 100,
        )

        os.makedirs(self.paths.log_dir, exist_ok=True)
        hist_path = os.path.join(self.paths.log_dir, "elo_history.csv")
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
                    os.path.basename(latest_path),
                    n1,
                    n2,
                    d,
                    f"{win_rate:.4f}",
                    f"{new_elo:.2f}",
                    1,
                ]
            )

        eval_tb_dir = os.path.join(self.paths.tb_dir, "eval")
        os.makedirs(eval_tb_dir, exist_ok=True)
        writer = SummaryWriter(eval_tb_dir)
        now = int(time.time())
        writer.add_scalar("elo_by_step", new_elo, eval_step)
        writer.add_scalar("win_rate_by_step", win_rate, eval_step)
        writer.add_scalar("elo_by_time", new_elo, now)
        writer.add_scalar("win_rate_by_time", win_rate, now)
        writer.flush()
        writer.close()

    def _save_model(self, timestamp: int | None = None):
        if self.distributed and dist.get_rank() != 0:
            return
        if timestamp is None:
            timestamp = int(time.time())

        fname = os.path.join(self.paths.model_dir, f"net_{timestamp}.pt")
        target = self.net.module if hasattr(self.net, "module") else self.net
        state = target.state_dict()

        meta = checkpoint_meta(
            game_spec=self.game_spec,
            game_name=self.game_name,
            channels=CHANNELS,
            blocks=NUM_RES,
        )
        save_checkpoint(fname, state, meta)
        logger.info("Model saved to %s", fname)

    def _latest_model_path(self):
        files = sorted(glob.glob(os.path.join(self.paths.model_dir, "net_*.pt")))
        return files[-1] if files else None

    def _previous_model_path(self):
        files = sorted(glob.glob(os.path.join(self.paths.model_dir, "net_*.pt")))
        return files[-2] if len(files) >= 2 else None

    def _load_latest_model(self):
        path = self._latest_model_path()
        if path:
            target = self.net.module if hasattr(self.net, "module") else self.net
            state_dict, meta = load_checkpoint(path, map_location=self.device)
            validate_checkpoint_meta(meta, self.game_spec)
            target.load_state_dict(state_dict)
            logger.info("Loaded model %s", path)
