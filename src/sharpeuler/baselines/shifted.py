"""SD3-style shifted FlowMatch Euler schedule.

A static monotone shift of the form

.. math::

    t_{\\text{shifted}} = \\frac{\\alpha t}{1 + (\\alpha - 1) t}

with ``alpha = 3`` by default, applied between user-specified endpoints.
This is the same shift used in Stable Diffusion 3 / FLUX research-default
configs and serves as a profile-agnostic baseline that nonetheless biases
the schedule toward higher noise levels.

References: Esser et al., "Scaling Rectified Flow Transformers for
High-Resolution Image Synthesis" (2024).
"""
from __future__ import annotations

import torch


def build_shifted_schedule(
    B: int,
    alpha: float = 3.0,
    t_start: float = 1.0,
    t_end: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """B-step shifted schedule descending from ``t_start`` to ``t_end``.

    Args:
        B: Number of integration steps; returns ``B + 1`` boundary points.
        alpha: Shift parameter. ``alpha == 1`` recovers a uniform schedule;
            larger values push more density toward higher noise levels.
        t_start: Start (high-noise) endpoint, descending.
        t_end: End (low-noise) endpoint.
        device: Device for the returned tensor.
    """
    u = torch.linspace(0.0, 1.0, B + 1, device=device)
    shifted = alpha * u / (1.0 + (alpha - 1.0) * u)
    # interpolate from [t_start, t_end] using shifted as the parameter
    # u=0 -> t_start, u=1 -> t_end
    sched_asc = t_start + shifted * (t_end - t_start)
    return sched_asc
