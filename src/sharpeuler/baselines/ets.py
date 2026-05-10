"""Entropic time scheduler (ETS) wrapper for flow-matching models.

Wraps the BioNeMo ``EntropicInferenceSchedule`` with a small
modification: we take the absolute value of the entropy rate to handle
the sign convention used by negative-velocity flow-matching predictors,
matching the signed-rate convention in Stancevic et al. (2025).

References:
    - Stancevic, Handke, Ambrogioni. "Entropic Time Schedulers for
      Generative Diffusion Models." NeurIPS 2025.
    - NVIDIA BioNeMo MoCo, "Entropic Flow Matching for Optimal Time
      Scheduling" tutorial.
"""
from __future__ import annotations

import numpy as np
import torch

from sharpeuler.core.samplers import euler_fm_sample


try:
    from bionemo.moco.schedules.inference_time_schedules import (
        EntropicInferenceSchedule,
        TimeDirection,
    )

    _HAS_BIONEMO = True
except ImportError:  # pragma: no cover
    EntropicInferenceSchedule = object  # type: ignore[assignment]
    TimeDirection = None  # type: ignore[assignment]
    _HAS_BIONEMO = False


class AbsEntropicInferenceSchedule(EntropicInferenceSchedule):
    """ETS variant that takes ``|entropy_rate|`` to handle FM sign conventions."""

    def _calculate_entropy_rate(self, t_val):
        return abs(super()._calculate_entropy_rate(t_val))


def make_predictor_fn(model: torch.nn.Module):
    """Build a BioNeMo-compatible ``predictor_fn(t, x) -> v`` from an FM model.

    Adapter handles BioNeMo's UNIFIED time convention (``t = 0`` is noise,
    ``t = 1`` is data) by mapping to our descending FM convention via
    ``s = 1 - t`` and negating the velocity to preserve the sign of the
    integration direction.
    """

    def predictor_fn(t_ets, x):
        if t_ets.ndim == 0:
            t_ets = t_ets.unsqueeze(0)
        if t_ets.ndim == 2 and t_ets.shape[-1] == 1:
            t_ets = t_ets.squeeze(-1)
        if t_ets.shape[0] != x.shape[0]:
            t_ets = t_ets.expand(x.shape[0])
        s = 1.0 - t_ets
        return -model(x, s)

    return predictor_fn


def build_ets(model, X, BUDGETS, N_VIS, x1_vis, device, seed: int = 42):
    """Generate ETS-scheduled samples for each budget in ``BUDGETS``.

    Args:
        model: FM velocity field.
        X: Real-data array used as the ETS ``x_1`` distribution.
        BUDGETS: Iterable of integer step counts.
        N_VIS: Number of samples to draw per budget.
        x1_vis: Initial noise tensor of shape ``(N_VIS, 2)``.
        device: Device for sampling.
        seed: Seed for ETS construction reproducibility.

    Returns:
        Dict ``{B: numpy_array(N_VIS, 2)}``.

    Raises:
        ImportError: If ``bionemo.moco`` is not installed. Install it with
            ``pip install nvidia-bionemo-moco``.
    """
    if not _HAS_BIONEMO:
        raise ImportError(
            "ETS baseline requires bionemo.moco. "
            "Install with `pip install nvidia-bionemo-moco`."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    def x_0(n):
        return torch.randn((n, 2), device=device)

    def x_1(n):
        idx = np.random.choice(len(X), n, replace=True)
        return torch.from_numpy(X[idx]).to(device).float()

    pred = make_predictor_fn(model)
    out = {}
    for B in BUDGETS:
        sched_ets = (
            AbsEntropicInferenceSchedule(
                predictor=pred,
                x_0_sampler=x_0,
                x_1_sampler=x_1,
                nsteps=B,
                n_approx_entropy_points=100,
                batch_size=512,
                inclusive_end=False,
                min_t=0.01,
                direction=TimeDirection.UNIFIED,
                device=device,
            )
            .generate_schedule()
            .to(device)
        )
        sched = 1.0 - sched_ets
        out[B] = (
            euler_fm_sample(model, N_VIS, sched, device=device, x1=x1_vis)
            .cpu()
            .numpy()
        )
    return out
