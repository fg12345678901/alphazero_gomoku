# trainer/dataset.py
from __future__ import annotations

import glob
import os
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config import BATCH_SIZE, BUFFER_SIZE, DATA_DIR


class ReplayBuffer(Dataset):
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        buffer_size: int = BUFFER_SIZE,
        default_batch_size: int = BATCH_SIZE,
    ):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.default_batch_size = default_batch_size
        self.data = []  # (planes, pi, z)
        self.load_existing()

    def load_existing(self):
        """Load recent self-play data until buffer_size is satisfied."""
        files = sorted(
            glob.glob(os.path.join(self.data_dir, "selfplay_*.pkl")),
            key=os.path.getmtime,
        )
        chunks = []
        total = 0
        for path in reversed(files):
            with open(path, "rb") as fp:
                chunk = pickle.load(fp)
            chunks.append(chunk)
            total += len(chunk)
            if total >= self.buffer_size:
                break

        self.data = [item for ch in reversed(chunks) for item in ch]
        self._trim()

    def append_from_file(self, file_path: str):
        with open(file_path, "rb") as fp:
            self.data.extend(pickle.load(fp))
        self._trim()

    def _trim(self):
        if len(self.data) > self.buffer_size:
            self.data = self.data[-self.buffer_size :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        planes, pi, z = self.data[idx]
        return (
            torch.tensor(planes, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32),
        )

    def loader(self, shuffle=True, distributed=False, batch_size: int | None = None):
        """Return DataLoader with optional DistributedSampler and custom batch size."""
        if batch_size is None:
            batch_size = self.default_batch_size

        if distributed:
            sampler = DistributedSampler(self, shuffle=shuffle)
            return (
                DataLoader(
                    self,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=0,
                    pin_memory=True,
                ),
                sampler,
            )
        return (
            DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True,
            ),
            None,
        )
