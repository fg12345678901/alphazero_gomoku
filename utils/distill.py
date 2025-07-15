# utils/distill.py
"""Model distillation script.
Distills a larger teacher model to a smaller student model.
Student defaults to config CHANNELS and NUM_RES, but can be overridden.
Outputs the student checkpoint to distillation_model/net_<timestamp>_<channels>x<blocks>.pt.
"""
from __future__ import annotations
import argparse, os, glob, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

# ----- project imports -----
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from config import (BOARD_SIZE, CHANNELS, NUM_RES, DEVICE, BATCH_SIZE, TRAIN_UPDATES, LEARNING_RATE, WEIGHT_DECAY)
from network.model import AlphaZeroNet
from trainer.dataset import ReplayBuffer
from logging_setup import setup_logging


logger = setup_logging()


def latest_ckpt(model_dir: str = "models") -> str | None:
    files = sorted(glob.glob(os.path.join(model_dir, "net_*.pt")))
    return files[-1] if files else None


class Distiller:
    def __init__(self, teacher_path: str, channels: int, blocks: int,
                 lr: float, weight_decay: float, batch_size: int,
                 updates: int, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        logger.info(f"Loading teacher model from {teacher_path}")
        self.teacher = AlphaZeroNet().to(DEVICE)
        self.teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        self.teacher.eval()

        logger.info(f"Initializing student model: channels={channels}, blocks={blocks}")
        self.student = AlphaZeroNet(board_size=BOARD_SIZE, channels=channels,
                                   blocks=blocks).to(DEVICE)
        self.optimizer = optim.Adam(self.student.parameters(), lr=lr,
                                    weight_decay=weight_decay)

        self.buffer = ReplayBuffer()
        self.loader = DataLoader(self.buffer, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.iterator = iter(self.loader)
        self.updates = updates

    def _next_batch(self):
        try:
            boards, _, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            boards, _, _ = next(self.iterator)
        return boards.to(DEVICE)

    def run(self):
        for _ in trange(self.updates, desc="Distillation"):
            boards = self._next_batch()
            with torch.no_grad():
                t_pi, t_v = self.teacher(boards)
                target_pi = torch.softmax(t_pi, dim=1)
                target_v = t_v.detach()

            self.optimizer.zero_grad()
            s_pi, s_v = self.student(boards)
            loss_pi = torch.mean(
                torch.sum(-target_pi * torch.log_softmax(s_pi, dim=1), dim=1))
            loss_v = F.mse_loss(s_v, target_v)
            loss = loss_pi + loss_v
            loss.backward()
            self.optimizer.step()

        ts = int(time.time())
        fname = os.path.join(
            self.out_dir, f"net_{ts}_{self.student.conv.out_channels}x{len(self.student.res_layers)}.pt")
        torch.save(self.student.state_dict(), fname)
        logger.info(f"Distilled model saved to {fname}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a trained AlphaZero model")
    parser.add_argument("--teacher", type=str, default=None,
                        help="path to teacher checkpoint (default: latest in models dir)")
    parser.add_argument("--channels", type=int, default=CHANNELS,
                        help="student conv channels")
    parser.add_argument("--blocks", type=int, default=NUM_RES,
                        help="student residual blocks")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--updates", type=int, default=TRAIN_UPDATES)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--out-dir", type=str, default="distillation_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    teacher = args.teacher or latest_ckpt()
    if not teacher:
        raise FileNotFoundError("Teacher checkpoint not found")

    distiller = Distiller(
        teacher_path=teacher,
        channels=args.channels,
        blocks=args.blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        updates=args.updates,
        out_dir=args.out_dir,
    )
    distiller.run()
