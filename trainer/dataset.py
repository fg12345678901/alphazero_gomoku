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
        files = sorted(glob.glob(os.path.join(DATA_DIR, "selfplay_*.pkl")))
        for f in files:
            with open(f, "rb") as fp:
                self.data.extend(pickle.load(fp))
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
