"""Adaptive schedule construction from a sharpness signal.

The core of SharpEuler. Given a sharpness signal phi(t) (typically a power of
velocity-field acceleration), this module constructs a non-uniform timestep
schedule whose density tracks phi via inverse-CDF sampling of the smoothed
signal.

Reference: Sharpness-Aware Flow Matching (NeurIPS 2026 submission).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_smooth_1d(x: torch.Tensor, sigma: float = 5.0) -> torch.Tensor:
    """Gaussian smoothing along the last dim of a 1D tensor.

    Reflect-padded Gaussian convolution. ``sigma`` is in *index units* on
    the input grid, not in time units. Returns a tensor with the same
    length and dtype as ``x``.

    Args:
        x: 1D tensor of values to smooth.
        sigma: Gaussian bandwidth in index units. ``sigma <= 0`` returns
            a clone of ``x`` unmodified.
    """
    if sigma <= 0:
        return x.clone()
    radius = max(1, int(3 * sigma))
    grid = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()
    x_pad = F.pad(x[None, None, :], (radius, radius), mode="reflect")
    return F.conv1d(x_pad, kernel[None, None, :])[0, 0]


@torch.no_grad()
def build_adaptive_schedule_from_signal(
    t_support_desc: torch.Tensor,
    profile_vals: torch.Tensor,
    B: int,
    sigma: float = 5.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a B-step schedule by inverse-CDF sampling of a smoothed signal.

    Given a per-timestep signal phi(t) defined on a *descending* support grid
    (1.0 -> 0.0, the FLUX/FM convention), normalize to a density, integrate
    to a CDF, and invert at uniform quantiles to produce ``B+1`` schedule
    points spanning the same range. The result is returned in the same
    descending-time orientation as the input support.

    Args:
        t_support_desc: 1D tensor of time support points in *descending* order
            (e.g. ``torch.linspace(1.0, 0.0, N)``). Length ``N``.
        profile_vals: 1D tensor of nonnegative signal values phi at the
            support points; must have the same length as ``t_support_desc``.
            Negative values are clamped to zero.
        B: Number of integration steps. Returns ``B+1`` boundary points.
        sigma: Gaussian smoothing bandwidth in index units, applied to the
            signal before normalization.
        eps: Numerical floor for divisions and CDF normalization.

    Returns:
        Tensor of shape ``(B+1,)`` with timesteps in descending order
        (same orientation as ``t_support_desc``).
    """
    device, dtype = t_support_desc.device, t_support_desc.dtype

    # smooth in *index units* on the support; clamp negatives just in case
    phi = profile_vals.float().clamp_min(0.0)
    phi = gaussian_smooth_1d(phi, sigma=sigma)

    # density: phi normalized to sum to 1
    p_desc = phi / phi.sum().clamp_min(eps)

    # flip to ascending time for canonical CDF construction
    t_asc = torch.flip(t_support_desc, dims=[0])
    p_asc = torch.flip(p_desc, dims=[0])

    # cumulative sum -> CDF; prepend a 0 so cdf_aug[0] == 0
    cdf_asc = torch.cumsum(p_asc, dim=0)
    t_aug = torch.cat([torch.zeros(1, device=device, dtype=dtype), t_asc])
    cdf_aug = torch.cat([torch.zeros(1, device=device, dtype=dtype), cdf_asc])

    # B+1 uniform quantiles in [0, 1] -> invert through the CDF
    u = torch.linspace(0.0, 1.0, B + 1, device=device, dtype=dtype)
    idx = torch.searchsorted(cdf_aug, u).clamp(1, len(cdf_aug) - 1)
    c0, c1 = cdf_aug[idx - 1], cdf_aug[idx]
    t0, t1 = t_aug[idx - 1], t_aug[idx]
    w = (u - c0) / (c1 - c0 + eps)
    t_quant_asc = t0 + w * (t1 - t0)

    # return in descending orientation to match the input
    return torch.flip(t_quant_asc, dims=[0]).contiguous()
