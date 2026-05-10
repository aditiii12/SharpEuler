"""Euler integration of a flow-matching velocity field with a custom schedule."""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


@torch.no_grad()
def euler_fm_sample(
    model: torch.nn.Module,
    n: int,
    schedule: torch.Tensor,
    device: str = "cuda",
    x1: Optional[torch.Tensor] = None,
    return_traj: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, list, list]]:
    """Forward-Euler sampling along a specified descending time schedule.

    Integrates ``dx/dt = -v(x, t)`` from ``t = schedule[0]`` to ``t = schedule[-1]``
    with step sizes determined by consecutive entries of ``schedule``. Used
    consistently across all schedule variants (uniform, ETS, SharpEuler) so
    that comparisons are paired in the integrator.

    Args:
        model: Velocity field ``v(x, t)`` accepting ``x`` of shape ``(n, d)``
            and ``t`` of shape ``(n,)``.
        n: Number of samples (only used when ``x1`` is None).
        schedule: 1D tensor of timesteps in descending order. Length ``B + 1``
            for a ``B``-step integration.
        device: Device for integration.
        x1: Optional initial noise tensor of shape ``(n, 2)``. If None, draws
            a fresh standard-normal sample.
        return_traj: If True, also returns the per-step trajectory and
            velocity history (memory-heavy for large ``n``).

    Returns:
        Either ``x`` of shape ``(n, 2)``, or ``(x, traj, vel_hist)`` if
        ``return_traj`` is True.
    """
    schedule = schedule.to(device)
    dtype = next(model.parameters()).dtype
    if x1 is None:
        x = torch.randn((n, 2), device=device, dtype=dtype)
    else:
        x = x1.to(device=device, dtype=dtype).clone()

    traj = [x.clone()] if return_traj else None
    vel_hist = []
    for k in range(len(schedule) - 1):
        t_k, t_next = schedule[k], schedule[k + 1]
        h = t_k - t_next
        t_batch = torch.full((x.shape[0],), float(t_k), device=device, dtype=dtype)
        v = model(x, t_batch)
        x = x - h * v
        vel_hist.append(v.clone())
        if return_traj:
            traj.append(x.clone())

    if return_traj:
        return x, traj, vel_hist
    return x
