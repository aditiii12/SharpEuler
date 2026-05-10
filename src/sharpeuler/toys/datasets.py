"""Synthetic 2D datasets for SharpEuler experiments."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_branchy_flower_dataset(
    n_samples: int = 100_000,
    n_petals: int = 8,
    depth: int = 6,
    branch_angle_deg: float = 24.0,
    length_decay: float = 0.76,
    angle_decay: float = 0.90,
    core_shrink: float = 0.65,
    core_levels: int = 2,
    tip_bias: float = 1.4,
    noise_decay: float = 0.90,
    base_noise: float = 0.015,
    seed: int = 0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a recursively branching ``flower'' manifold in 2D.

    Constructs ``n_petals`` radial trees, each grown by recursive bifurcation
    to ``depth`` levels. Sample points are drawn along segments with weights
    proportional to segment length, and a tip-bias factor pushes mass toward
    leaf segments.

    Returns:
        Tuple of ``(points, segments)`` where ``points`` has shape
        ``(n_samples, 2)`` and ``segments`` has shape ``(S, 2, 2)``
        containing each segment's start and end points.
    """
    rng = np.random.default_rng(seed)

    starts, ends, levels = [], [], []

    def grow(p, angle, length, level, branch_angle):
        if level < core_levels:
            length = length * core_shrink
        q = p + length * np.array([np.cos(angle), np.sin(angle)])
        starts.append(p)
        ends.append(q)
        levels.append(level)
        if level >= depth:
            return
        for sign in (-1.0, +1.0):
            grow(
                q,
                angle + sign * branch_angle,
                length * length_decay,
                level + 1,
                branch_angle * angle_decay,
            )

    base_branch_angle = np.deg2rad(branch_angle_deg)
    origin = np.zeros(2)
    for i in range(n_petals):
        grow(origin, 2 * np.pi * i / n_petals, 1.0, 0, base_branch_angle)

    starts = np.asarray(starts)
    ends = np.asarray(ends)
    levels = np.asarray(levels)
    seg_lengths = np.linalg.norm(ends - starts, axis=1)

    weights = seg_lengths / seg_lengths.sum()
    seg_idx = rng.choice(len(starts), size=n_samples, p=weights)

    u = rng.uniform(0.0, 1.0, size=n_samples)
    t = u ** (1.0 / tip_bias)

    s = starts[seg_idx]
    e = ends[seg_idx]
    pts = s + t[:, None] * (e - s)

    if normalize:
        scale = np.max(np.abs(pts))
        if scale > 0:
            pts = pts / scale

    segments = np.stack([starts, ends], axis=1)
    return pts.astype(np.float32), segments.astype(np.float32)


def make_rotated_gaussian_grid(
    n_samples: int = 100_000,
    grid_size: int = 5,
    spacing: float = 0.5,
    std: float = 0.05,
    angle_deg: float = 45.0,
    seed: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """A grid_size x grid_size grid of isotropic Gaussian modes, then rotated.

    Default config produces 25 well-separated Gaussians on a 45-deg-rotated
    lattice, normalized so that ``max(|x|) == 1``.
    """
    rng = np.random.default_rng(seed)
    coords = (
        np.linspace(-(grid_size - 1) / 2, (grid_size - 1) / 2, grid_size) * spacing
    )
    centers = np.array([(x, y) for x in coords for y in coords])

    K = len(centers)
    counts = np.full(K, n_samples // K)
    counts[: n_samples - counts.sum()] += 1
    pts = np.concatenate(
        [
            rng.normal(loc=c, scale=std, size=(n, 2))
            for c, n in zip(centers, counts)
        ],
        axis=0,
    )

    theta = np.deg2rad(angle_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pts = pts @ R.T

    if normalize:
        pts = pts / np.max(np.abs(pts))
    rng.shuffle(pts)
    return pts.astype(np.float32)


def make_spiral(
    n_samples: int = 30_000,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Two interleaved logarithmic spirals, normalized to unit std per axis."""
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.sqrt(rng.random(n)) * 3 * np.pi
    r_a = theta + np.pi
    x_a = np.stack([r_a * np.cos(theta), r_a * np.sin(theta)], axis=1)
    r_b = -theta - np.pi
    x_b = np.stack([r_b * np.cos(theta), r_b * np.sin(theta)], axis=1)
    X = np.vstack([x_a, x_b])
    X = X + rng.standard_normal(X.shape) * noise
    X = (X - X.mean(0)) / X.std(0)
    return X.astype(np.float32)
