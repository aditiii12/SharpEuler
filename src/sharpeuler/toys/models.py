"""Architectures used in toy 2D flow-matching experiments.

Three models are supported:

- :class:`FMNet`: residual MLP with FiLM-style time conditioning. Used for
  the branchy-flower dataset.
- :class:`FMNet2`: smaller MLP with raw + Fourier-feature inputs and
  sinusoidal time embeddings. Used for the rotated-grid dataset.
- :class:`FMNetBig`: deeper residual MLP with random Fourier features.
  Used for the harder spiral dataset.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# FMNet (branchy flower)
# =====================================================================
class TimeMLP(nn.Module):
    """Two-layer MLP that turns scalar t into a low-dim embedding."""

    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t[:, None])


class ResidualBlock(nn.Module):
    """LayerNorm -> Linear -> SiLU residual block with FiLM time conditioning."""

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale = nn.Linear(time_dim, hidden_dim)
        self.to_shift = nn.Linear(time_dim, hidden_dim)

        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc(self.norm(x)))
        h = h * (1.0 + self.to_scale(t_emb)) + self.to_shift(t_emb)
        return x + h


class FMNet(nn.Module):
    """Residual MLP with FiLM time conditioning. Default for branchy flower."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 32,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.time_mlp = TimeMLP(time_dim)
        self.inp = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, data_dim)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = F.silu(self.norm(h))
        return self.out(h)


# =====================================================================
# FMNet2 (rotated grid)
# =====================================================================
class FMNet2(nn.Module):
    """MLP with raw + Fourier-feature x and sinusoidal time embedding."""

    def __init__(
        self,
        hidden: int = 256,
        depth: int = 4,
        t_embed_dim: int = 64,
        x_embed_dim: int = 64,
        x_freq_max: float = 10.0,
    ):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.x_embed_dim = x_embed_dim
        n_freqs = x_embed_dim // 4
        x_freqs = torch.exp(torch.linspace(0, np.log(x_freq_max), n_freqs))
        self.register_buffer("x_freqs", x_freqs)

        x_emb_out = 2 * 2 * n_freqs
        in_dim = 2 + x_emb_out + t_embed_dim

        layers = []
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def t_embed(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_embed_dim // 2
        freqs = torch.exp(
            -np.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        ang = t[:, None] * freqs[None, :] * 2 * np.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def x_embed(self, x: torch.Tensor) -> torch.Tensor:
        ang = x[:, :, None] * self.x_freqs[None, None, :] * 2 * np.pi
        ang = ang.reshape(x.shape[0], -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(
            torch.cat([x, self.x_embed(x), self.t_embed(t)], dim=-1)
        )


# =====================================================================
# FMNetBig (spiral)
# =====================================================================
class FourierFeatures1D(nn.Module):
    """Random projection -> sin/cos features for low-dim inputs."""

    def __init__(self, in_dim: int, n_freqs: int, freq_max: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, n_freqs) * freq_max
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


def sinusoidal_t_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Standard transformer-style positional embedding for scalar t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FMNetBig(nn.Module):
    """Deeper residual MLP with random Fourier features. Used for spirals."""

    def __init__(
        self,
        x_dim: int = 2,
        hidden: int = 512,
        n_blocks: int = 6,
        t_embed_dim: int = 128,
        n_x_freqs: int = 64,
        x_freq_max: float = 10.0,
    ):
        super().__init__()
        self.t_embed_dim = t_embed_dim

        self.x_fourier = FourierFeatures1D(x_dim, n_x_freqs, freq_max=x_freq_max)
        x_feat_dim = x_dim + 2 * n_x_freqs

        self.in_proj = nn.Linear(x_feat_dim + t_embed_dim, hidden)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(n_blocks)
            ]
        )

        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(-1)
        t_emb = sinusoidal_t_embed(t, dim=self.t_embed_dim)
        x_feat = torch.cat([x, self.x_fourier(x)], dim=-1)
        h = self.in_proj(torch.cat([x_feat, t_emb], dim=-1))
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)
