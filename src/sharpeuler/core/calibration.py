"""Sharpness profile calibration.

Given a trained flow-matching model, this module estimates the per-timestep
acceleration profile ``\\|a(t)\\|`` by integrating a dense uniform schedule
and finite-differencing consecutive velocity predictions. The resulting
profile is the input to :func:`sharpeuler.core.schedule.build_adaptive_schedule_from_signal`.
"""
from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def collect_dense_trajectories(
    model: torch.nn.Module,
    n_paths: int = 1024,
    N_dense: int = 1000,
    device: str = "cuda",
) -> Tuple[torch.Tensor, list, list]:
    """Integrate a dense uniform Euler schedule and record all velocities.

    Used as the first step of profile calibration. Returns the time grid,
    the per-step trajectory snapshots, and the per-step velocity predictions.

    Args:
        model: Trained flow-matching velocity field ``v(x, t)``. Must accept
            ``x`` of shape ``(n, d)`` and ``t`` of shape ``(n,)``, returning
            velocity of shape ``(n, d)``. Model is set to eval mode internally.
        n_paths: Number of independent integration paths.
        N_dense: Number of integration steps. Higher values give a finer
            profile but cost more compute.
        device: Device to run the integration on.

    Returns:
        Tuple of ``(sched_dense, traj, vel_hist)`` where
        - ``sched_dense`` is the dense schedule of shape ``(N_dense + 1,)``
          spanning ``1.0 -> 0.0``,
        - ``traj`` is a list of ``N_dense + 1`` tensors of shape ``(n_paths, d)``
          containing the integrated state at each time step,
        - ``vel_hist`` is a list of ``N_dense`` tensors containing the
          velocity predictions at each step.
    """
    model.eval()
    sched_dense = torch.linspace(1.0, 0.0, N_dense + 1, device=device)
    dtype = next(model.parameters()).dtype
    x = torch.randn((n_paths, 2), device=device, dtype=dtype)
    traj, vel_hist = [x.clone()], []
    for k in range(N_dense):
        t_k, t_next = sched_dense[k], sched_dense[k + 1]
        h = t_k - t_next
        t_batch = torch.full((n_paths,), float(t_k), device=device, dtype=dtype)
        v = model(x, t_batch)
        x = x - h * v
        vel_hist.append(v.clone())
        traj.append(x.clone())
    return sched_dense, traj, vel_hist


def compute_sharpness_profile(
    sched_dense: torch.Tensor,
    vel_hist: list,
) -> dict:
    """Compute the per-time acceleration profile from a velocity history.

    Estimates ``\\|a(t)\\|`` by central finite-difference of consecutive
    velocity predictions, averaged across paths. The output is defined on
    the *midpoint* grid of length ``N_dense - 1``.

    Args:
        sched_dense: Time schedule of shape ``(N_dense + 1,)`` from
            :func:`collect_dense_trajectories`.
        vel_hist: List of ``N_dense`` velocity tensors of shape
            ``(n_paths, d)`` from :func:`collect_dense_trajectories`.

    Returns:
        Dict with keys ``t_mid`` (midpoint time grid, shape
        ``(N_dense - 1,)``, descending) and ``accel_norm`` (mean
        ``\\|a(t)\\|`` per midpoint, same shape).
    """
    V = torch.stack(vel_hist, dim=0)  # (N_dense, n_paths, d)
    dt = (sched_dense[:-1] - sched_dense[1:]).abs()
    dt_mid = 0.5 * (dt[:-1] + dt[1:])
    dV = V[1:] - V[:-1]
    accel = dV / dt_mid[:, None, None]
    accel_norm = accel.norm(dim=-1).mean(dim=-1)
    t_mid = 0.5 * (sched_dense[:-2] + sched_dense[2:])
    return {"t_mid": t_mid, "accel_norm": accel_norm}
