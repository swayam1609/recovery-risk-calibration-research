
"""Portfolio credit-loss and tail-risk metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_losses(losses: NDArray) -> FloatArray:
    """Validate a one-dimensional portfolio-loss sample."""
    values = np.asarray(losses, dtype=np.float64)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array.")

    if not np.all(np.isfinite(values)):
        raise ValueError("losses must contain only finite values.")

    if np.any(values < 0.0):
        raise ValueError("losses must be non-negative.")

    return values


def _validate_confidence(confidence: float) -> float:
    """Validate a tail-risk confidence level."""
    if (
        not np.isscalar(confidence)
        or not np.isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("confidence must be strictly between 0 and 1.")

    return float(confidence)


def expected_loss(losses: NDArray) -> float:
    """Estimate expected portfolio loss using the sample mean."""
    return float(np.mean(_validate_losses(losses)))


def value_at_risk(
    losses: NDArray,
    confidence: float = 0.99,
) -> float:
    """Estimate VaR using the empirical loss quantile."""
    values = _validate_losses(losses)
    confidence = _validate_confidence(confidence)

    return float(np.quantile(values, confidence, method="linear"))


def expected_shortfall(
    losses: NDArray,
    confidence: float = 0.99,
) -> float:
    """Estimate expected shortfall using the empirical quantile integral.

    This estimator integrates the empirical quantile function over the
    upper confidence tail. It handles discrete loss distributions and
    avoids assuming that every loss above VaR belongs entirely to the tail.
    """
    values = _validate_losses(losses)
    confidence = _validate_confidence(confidence)

    sorted_losses = np.sort(values)
    n = sorted_losses.size

    # Integrate the empirical quantile function from confidence to 1.
    start = confidence * n
    first_index = int(np.floor(start))
    fraction = start - first_index

    tail_sum = (1.0 - fraction) * sorted_losses[first_index]

    if first_index + 1 < n:
        tail_sum += np.sum(sorted_losses[first_index + 1:])

    return float(tail_sum / (n * (1.0 - confidence)))


def economic_capital(
    losses: NDArray,
    confidence: float = 0.99,
) -> float:
    """Calculate economic capital as VaR minus expected loss."""
    values = _validate_losses(losses)

    return max(
        0.0,
        value_at_risk(values, confidence) - expected_loss(values),
    )


def capital_gap(
    beta_losses: NDArray,
    kde_losses: NDArray,
    confidence: float = 0.99,
) -> dict[str, float]:
    """Compare economic capital estimated by Beta and KDE models.

    A positive signed gap means the KDE model estimates more capital
    than the Beta model. A negative gap means it estimates less.
    """
    beta_capital = economic_capital(beta_losses, confidence)
    kde_capital = economic_capital(kde_losses, confidence)

    return {
        "beta_economic_capital": beta_capital,
        "kde_economic_capital": kde_capital,
        "signed_capital_gap_kde_minus_beta": kde_capital - beta_capital,
        "absolute_capital_gap": abs(kde_capital - beta_capital),
    }
