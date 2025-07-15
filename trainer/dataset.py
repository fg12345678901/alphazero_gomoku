# trainer/dataset.py
from __future__ import annotations
import os, pickle, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, BUFFER_SIZE, BATCH_SIZE

class ReplayBuffer(Dataset):
    def __init__(self):
        self.data = []  # (planes, pi, z)
        self.load_existing()

    def load_existing(self):
        """Load recent self-play data until BUFFER_SIZE is satisfied."""
        files = sorted(
            glob.glob(os.path.join(DATA_DIR, "selfplay_*.pkl")),
            key=os.path.getmtime,
        )
        chunks = []
        total = 0
        for f in reversed(files):  # 从最新文件开始向前找
            with open(f, "rb") as fp:
                chunk = pickle.load(fp)
            chunks.append(chunk)
            total += len(chunk)
            if total >= BUFFER_SIZE:
                break
        # 按时间顺序拼接并裁剪
        self.data = [item for ch in reversed(chunks) for item in ch]
        self._trim()

    def append_from_file(self, file_path):
        with open(file_path, "rb") as fp:
            self.data.extend(pickle.load(fp))
        self._trim()

    def _trim(self):
        if len(self.data) > BUFFER_SIZE:
            self.data = self.data[-BUFFER_SIZE:]

    # ----------- PyTorch Dataset ----------- #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        planes, pi, z = self.data[idx]
        return torch.tensor(planes, dtype=torch.float32), \
               torch.tensor(pi, dtype=torch.float32), \
               torch.tensor(z, dtype=torch.float32)

    # ----------- DataLoader ----------- #
    def loader(self, shuffle=True):
        return DataLoader(self, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
