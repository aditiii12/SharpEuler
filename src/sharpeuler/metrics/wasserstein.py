"""Wasserstein-2 distance via Sinkhorn entropy regularization."""
from __future__ import annotations

import numpy as np

try:
    import ot

    _HAS_POT = True
except ImportError:  # pragma: no cover
    _HAS_POT = False


def wasserstein2(
    real: np.ndarray,
    fake: np.ndarray,
    n: int = 2000,
    reg: float = 0.01,
    seed: int = 0,
) -> float:
    """Squared Wasserstein-2 between subsampled point clouds via Sinkhorn.

    Smaller values indicate better distributional match. We subsample to
    ``n`` points per side both for speed and for variance control.

    Args:
        real: Array of shape ``(N_real, d)``.
        fake: Array of shape ``(N_fake, d)``.
        n: Subsample size per cloud.
        reg: Sinkhorn entropy regularization. Smaller is closer to true
            optimal transport but less stable.
        seed: Subsampling seed.

    Returns:
        Squared Wasserstein-2 estimate.

    Raises:
        ImportError: If POT (``pip install POT``) is not installed.
    """
    if not _HAS_POT:
        raise ImportError(
            "Wasserstein-2 requires POT. Install with `pip install POT`."
        )
    rng = np.random.default_rng(seed)
    real = np.asarray(real)
    fake = np.asarray(fake)
    n_r = min(n, len(real))
    n_f = min(n, len(fake))
    real_sub = real[rng.choice(len(real), n_r, replace=False)]
    fake_sub = fake[rng.choice(len(fake), n_f, replace=False)]
    a = np.ones(n_r) / n_r
    b = np.ones(n_f) / n_f
    M = np.linalg.norm(real_sub[:, None, :] - fake_sub[None, :, :], axis=-1) ** 2
    return float(ot.sinkhorn2(a, b, M, reg=reg))
