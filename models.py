# models.py
from __future__ import annotations
import torch
from torch import nn
from torch.nn.utils import spectral_norm as SN

class Generator(nn.Module):
    def __init__(self, c_in: int, c_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(c_hidden, c_hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(c_hidden, c_in, 1),
        )

    def forward(self, x):  # x: [B,C,T]
        return self.net(x)  # delta

class Critic(nn.Module):
    def __init__(self, c_in: int, c_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            SN(nn.Conv1d(c_in, c_hidden, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv1d(c_hidden, c_hidden, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            SN(nn.Conv1d(c_hidden, 1, 1)),
        )

    def forward(self, x):      # [B,C,T] -> [B]
        y = self.net(x).squeeze(-1).squeeze(1)
        return y
