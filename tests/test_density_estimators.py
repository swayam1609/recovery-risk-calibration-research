import numpy as np
import pytest

from src.data_simulator import generate_synthetic_recoveries
from src.density_estimators import (
    fit_parametric_beta,
    fit_reflection_kde,
    bounded_kde_density,
)


@pytest.fixture
def recoveries():
    return generate_synthetic_recoveries(
        n=1000,
        scenario="bimodal",
        seed=42,
    )


def test_beta_parameters_are_positive(recoveries):
    alpha, beta_param = fit_parametric_beta(recoveries)

    assert np.isfinite(alpha)
    assert np.isfinite(beta_param)
    assert alpha > 0
    assert beta_param > 0


def test_reflection_kde_returns_density(recoveries):
    kde = fit_reflection_kde(recoveries)
    grid = np.linspace(0.0, 1.0, 2001)

    density = bounded_kde_density(kde, grid)

    assert density.shape == grid.shape
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
    assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-6)


def test_beta_fit_rejects_invalid_recoveries():
    with pytest.raises(ValueError):
        fit_parametric_beta(np.array([0.2, 1.2, 0.4]))


def test_kde_rejects_invalid_bandwidth(recoveries):
    with pytest.raises(ValueError):
        fit_reflection_kde(recoveries, bandwidth=0)


def test_density_rejects_invalid_grid(recoveries):
    kde = fit_reflection_kde(recoveries)

    with pytest.raises(ValueError):
        bounded_kde_density(kde, np.array([0.0, 0.5, 0.4]))