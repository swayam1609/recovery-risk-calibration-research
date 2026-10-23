
"""Generate reproducible synthetic recovery-rate datasets.

All generated observations are synthetic and are intended for controlled
statistical experiments, not as representations of real credit portfolios.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def generate_synthetic_recoveries(
    n: int = 5_000,
    scenario: str = "unimodal",
    seed: int = 42,
) -> FloatArray:
    """Generate recovery rates in [0, 1].

    Parameters
    ----------
    n:
        Number of recovery observations.
    scenario:
        One of "unimodal", "bimodal", or "tail_heavy".
    seed:
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of synthetic recovery rates.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer greater than or equal to 2.")

    if scenario not in {"unimodal", "bimodal", "tail_heavy"}:
        raise ValueError(
            "scenario must be 'unimodal', 'bimodal', or 'tail_heavy'."
        )

    rng = np.random.default_rng(seed)

    if scenario == "unimodal":
        # One concentrated recovery population.
        recoveries = rng.beta(a=4.0, b=4.0, size=n)

    elif scenario == "bimodal":
        # Two distinct populations: low and high recovery.
        high_recovery_cluster = rng.random(n) < 0.5

        recoveries = np.empty(n, dtype=np.float64)
        n_high = int(high_recovery_cluster.sum())
        n_low = n - n_high

        recoveries[~high_recovery_cluster] = rng.beta(
            a=2.0, b=7.0, size=n_low
        )
        recoveries[high_recovery_cluster] = rng.beta(
            a=7.0, b=2.0, size=n_high
        )

    else:
        # A mixture with substantial probability near both boundaries.
        # This creates a broad, U-shaped recovery distribution.
        boundary_cluster = rng.random(n) < 0.5

        recoveries = np.empty(n, dtype=np.float64)
        n_boundary = int(boundary_cluster.sum())
        n_centre = n - n_boundary

        recoveries[boundary_cluster] = rng.beta(
            a=0.6, b=0.6, size=n_boundary
        )
        recoveries[~boundary_cluster] = rng.beta(
            a=4.0, b=4.0, size=n_centre
        )

    recoveries = np.asarray(recoveries, dtype=np.float64)

    if (
        recoveries.shape != (n,)
        or not np.all(np.isfinite(recoveries))
        or np.any(recoveries < 0.0)
        or np.any(recoveries > 1.0)
    ):
        raise RuntimeError("Generated recovery data failed validation.")

    return recoveries
