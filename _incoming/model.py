# src/uqdiff/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LogSNRTimeMLP",
    "ResidualBlock",
    "ScoreNet",
    "MODEL_DEFAULTS",
    "LAST_LAYER_NAME",
]

MODEL_DEFAULTS = dict(
    hidden_dim=32,
    time_dim=32,
    n_blocks=2,
)
LAST_LAYER_NAME = "out"

class LogSNRTimeMLP(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim), nn.SiLU()
        )
    def forward(self, t_code_1d):
        return self.net(t_code_1d[:, None])

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale = nn.Linear(time_dim, hidden_dim)
        self.to_shift = nn.Linear(time_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)
    def forward(self, x, t_emb):
        h = self.fc(self.norm(x))
        h = F.silu(h)
        scale = self.to_scale(t_emb)
        shift = self.to_shift(t_emb)
        h = h * (1 + scale) + shift
        return x + h

class ScoreNet(nn.Module):
    def __init__(self, hidden_dim=512, time_dim=32, n_blocks=2):
        super().__init__()
        self.time_mlp = LogSNRTimeMLP(time_dim)
        self.inp = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t_code):
        t_emb = self.time_mlp(t_code)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = F.silu(self.norm(h))
        return self.out(h)

    def forward_with_feat(self, x, t_code):
        t_emb = self.time_mlp(t_code)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = F.silu(self.norm(h))
        y = self.out(h)
        return y, h
