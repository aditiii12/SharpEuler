"""Precision, Recall, Density, and Coverage metrics for sample quality.

Implements the four standard manifold-based metrics:

- Precision (Kynkaeaenniemi et al. 2019): fraction of fake samples that
  fall within any real-sample's k-NN ball.
- Recall (Kynkaeaenniemi et al. 2019): fraction of real samples covered
  by fake samples' k-NN balls.
- Density (Naeem et al. 2020): mean number of real-sample k-NN balls
  each fake sample falls into, normalized by ``k``.
- Coverage (Naeem et al. 2020): fraction of real samples whose nearest
  fake sample lies within their own k-NN radius.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_prdc(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    nearest_k: int = 5,
) -> dict:
    """Compute Precision, Recall, Density, and Coverage.

    Args:
        real_features: Array of shape ``(N_real, d)``.
        fake_features: Array of shape ``(N_fake, d)``.
        nearest_k: Number of nearest neighbors used to define manifold
            balls. Standard choice is ``k=5``.

    Returns:
        Dict with keys ``precision``, ``recall``, ``density``, ``coverage``.
    """
    real = np.asarray(real_features)
    fake = np.asarray(fake_features)

    real_tree = cKDTree(real)
    fake_tree = cKDTree(fake)
    real_kth, _ = real_tree.query(real, k=nearest_k + 1)
    fake_kth, _ = fake_tree.query(fake, k=nearest_k + 1)
    real_radii = real_kth[:, nearest_k]
    fake_radii = fake_kth[:, nearest_k]

    pairwise = np.linalg.norm(fake[:, None, :] - real[None, :, :], axis=-1)

    fake_in_real_ball = (pairwise < real_radii[None, :]).any(axis=1)
    precision = float(fake_in_real_ball.mean())

    real_in_fake_ball = (pairwise.T < fake_radii[None, :]).any(axis=1)
    recall = float(real_in_fake_ball.mean())

    n_balls_per_fake = (pairwise < real_radii[None, :]).sum(axis=1)
    density = float(n_balls_per_fake.mean() / nearest_k)

    nearest_fake_to_real, _ = fake_tree.query(real, k=1)
    coverage = float((nearest_fake_to_real < real_radii).mean())

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
    }
