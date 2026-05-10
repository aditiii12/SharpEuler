"""Uniform timestep schedule baseline."""
from __future__ import annotations

import torch

from sharpeuler.core.samplers import euler_fm_sample


def build_uniform_schedule(B: int, device: str = "cuda") -> torch.Tensor:
    """B-step uniform schedule on the descending time grid 1.0 -> 0.0."""
    return torch.linspace(1.0, 0.0, B + 1, device=device)


def build_uniform(model, BUDGETS, N_VIS, x1_vis, device):
    """Sample under a uniform schedule for each budget in ``BUDGETS``.

    Returns a dict ``{B: numpy_array(N_VIS, 2)}``.
    """
    out = {}
    for B in BUDGETS:
        sched = build_uniform_schedule(B, device=device)
        out[B] = (
            euler_fm_sample(model, N_VIS, sched, device=device, x1=x1_vis)
            .cpu()
            .numpy()
        )
    return out
