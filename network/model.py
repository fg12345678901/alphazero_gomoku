# network/model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BOARD_SIZE, CHANNELS, NUM_RES, INPUT_PLANES

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, blocks=NUM_RES, channels=CHANNELS):
        super().__init__()
        self.board_size = board_size
        # 输入通道数由 HISTORY_STEPS 控制
        self.conv = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)
        self.res_layers = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # value head
        self.value_conv  = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(board_size * board_size, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, INPUT_PLANES, S, S)
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_layers(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)
