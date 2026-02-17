# trainer/dataset.py
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
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
        expected_input_planes: int | None = None,
        expected_action_size: int | None = None,
    ):
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.default_batch_size = default_batch_size
        self.expected_input_planes = expected_input_planes
        self.expected_action_size = expected_action_size
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
            chunk = self._filter_chunk(chunk)
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= self.buffer_size:
                break

        self.data = [item for ch in reversed(chunks) for item in ch]
        self._trim()

    def append_from_file(self, file_path: str):
        with open(file_path, "rb") as fp:
            self.data.extend(self._filter_chunk(pickle.load(fp)))
        self._trim()

    def _filter_chunk(self, chunk):
        if self.expected_input_planes is None and self.expected_action_size is None:
            return chunk

        filtered = []
        for item in chunk:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                continue
            planes, pi, _ = item
            planes_arr = np.asarray(planes)
            pi_arr = np.asarray(pi)

            if self.expected_input_planes is not None:
                if planes_arr.ndim < 1 or int(planes_arr.shape[0]) != self.expected_input_planes:
                    continue

            if self.expected_action_size is not None:
                if int(pi_arr.size) != self.expected_action_size:
                    continue

            filtered.append(item)
        return filtered

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
