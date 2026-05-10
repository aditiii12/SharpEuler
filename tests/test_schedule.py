"""Sanity tests for the core schedule construction."""
import torch

from sharpeuler.core.schedule import (
    build_adaptive_schedule_from_signal,
    gaussian_smooth_1d,
)


def test_uniform_signal_recovers_uniform_schedule():
    """A flat signal should produce a (near-)uniform schedule."""
    N = 1000
    t = torch.linspace(1.0, 0.0, N)
    phi = torch.ones(N)
    B = 16
    sched = build_adaptive_schedule_from_signal(t, phi, B=B, sigma=0.0)

    assert sched.shape == (B + 1,)
    assert torch.isclose(sched[0], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(sched[-1], torch.tensor(0.0), atol=1e-3)
    # descending
    diffs = sched[1:] - sched[:-1]
    assert (diffs <= 0).all()
    # near-uniform spacing
    spacing = sched[:-1] - sched[1:]
    assert (spacing.std() / spacing.mean()).item() < 0.05


def test_concentrated_signal_concentrates_schedule():
    """A peaked signal should put more schedule mass near the peak."""
    N = 1000
    t = torch.linspace(1.0, 0.0, N)
    # peak at t=0.3
    phi = torch.exp(-((t - 0.3) ** 2) / 0.01)
    B = 16
    sched = build_adaptive_schedule_from_signal(t, phi, B=B, sigma=1.0)

    assert sched.shape == (B + 1,)
    # majority of midpoints should fall near the peak
    midpoints = 0.5 * (sched[:-1] + sched[1:])
    near_peak = ((midpoints > 0.2) & (midpoints < 0.4)).sum().item()
    assert near_peak >= B // 2


def test_smoothing_zero_sigma_is_identity():
    x = torch.randn(50)
    out = gaussian_smooth_1d(x, sigma=0.0)
    assert torch.allclose(x, out)


def test_imports():
    """Top-level imports work."""
    import sharpeuler

    assert hasattr(sharpeuler, "build_adaptive_schedule_from_signal")
    assert hasattr(sharpeuler, "compute_sharpness_profile")
    assert hasattr(sharpeuler, "euler_fm_sample")
