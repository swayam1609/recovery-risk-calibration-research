
"""Parametric and nonparametric recovery-density estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta
from sklearn.neighbors import KernelDensity


FloatArray = NDArray[np.float64]


def _validate_recoveries(recoveries: NDArray) -> FloatArray:
    """Validate recovery observations in the closed interval [0, 1]."""
    values = np.asarray(recoveries, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("recoveries must be a one-dimensional array.")

    if values.size < 2:
        raise ValueError("At least two recovery observations are required.")

    if not np.all(np.isfinite(values)):
        raise ValueError("recoveries must contain only finite values.")

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Recovery rates must lie in [0, 1].")

    if np.ptp(values) == 0.0:
        raise ValueError("Recovery observations must not all be identical.")

    return values


def fit_parametric_beta(
    recoveries: NDArray,
) -> tuple[float, float]:
    """Fit a Beta distribution by maximum likelihood.

    Observations at 0 and 1 are clipped only for fitting because the
    continuous Beta likelihood is defined on the open interval (0, 1).
    The original recovery data are not modified.
    """
    values = _validate_recoveries(recoveries)

    eps = np.finfo(np.float64).eps
    fit_values = np.clip(values, eps, 1.0 - eps)

    alpha, beta_param, _, _ = beta.fit(
        fit_values,
        floc=0.0,
        fscale=1.0,
    )

    if (
        not np.isfinite(alpha)
        or not np.isfinite(beta_param)
        or alpha <= 0.0
        or beta_param <= 0.0
    ):
        raise RuntimeError("Beta fitting returned invalid parameters.")

    return float(alpha), float(beta_param)


def fit_reflection_kde(
    recoveries: NDArray,
    bandwidth: float | None = None,
) -> KernelDensity:
    """Fit a Gaussian KDE using reflection at both boundaries.

    Reflection augments the observations with -x and 2-x so that
    boundary observations receive reflected kernel contributions.
    """
    values = _validate_recoveries(recoveries)

    if bandwidth is None:
        # Robust Silverman-style bandwidth for one-dimensional data.
        std = np.std(values, ddof=1)
        q25, q75 = np.percentile(values, [25, 75])
        robust_scale = min(std, (q75 - q25) / 1.349)

        if not np.isfinite(robust_scale) or robust_scale <= 0.0:
            robust_scale = std

        bandwidth = 0.9 * robust_scale * values.size ** (-1.0 / 5.0)

    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be a positive finite number.")

    reflected = np.concatenate(
        [values, -values, 2.0 - values]
    ).reshape(-1, 1)

    kde = KernelDensity(
        kernel="gaussian",
        bandwidth=float(bandwidth),
    )
    kde.fit(reflected)

    return kde


def bounded_kde_density(
    kde: KernelDensity,
    grid: NDArray,
) -> FloatArray:
    """Evaluate KDE on [0, 1] and normalize its numerical integral."""
    x = np.asarray(grid, dtype=np.float64)

    if x.ndim != 1 or x.size < 2:
        raise ValueError("grid must be one-dimensional with at least 2 points.")

    if not np.all(np.isfinite(x)):
        raise ValueError("grid must contain only finite values.")

    if np.any(np.diff(x) <= 0.0):
        raise ValueError("grid must be strictly increasing.")

    if x[0] < 0.0 or x[-1] > 1.0:
        raise ValueError("grid must lie within [0, 1].")

    log_density = kde.score_samples(x.reshape(-1, 1))
    density = np.exp(log_density)

    area = np.trapezoid(density, x)

    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("KDE density has an invalid numerical integral.")

    density /= area

    return density
