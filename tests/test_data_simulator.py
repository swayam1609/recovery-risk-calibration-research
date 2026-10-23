import numpy as np
import pytest

from src.data_simulator import generate_synthetic_recoveries


def test_generates_correct_number_of_recoveries():
    recoveries = generate_synthetic_recoveries(n=500, seed=42)

    assert recoveries.shape == (500,)
    assert np.all(np.isfinite(recoveries))


@pytest.mark.parametrize(
    "scenario",
    ["unimodal", "bimodal", "tail_heavy"],
)
def test_recoveries_are_bounded(scenario):
    recoveries = generate_synthetic_recoveries(
        n=1000,
        scenario=scenario,
        seed=42,
    )

    assert np.all(recoveries >= 0.0)
    assert np.all(recoveries <= 1.0)


def test_same_seed_produces_same_results():
    first = generate_synthetic_recoveries(n=100, seed=123)
    second = generate_synthetic_recoveries(n=100, seed=123)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("n", [0, 1, -10, 2.5, True])
def test_invalid_observation_count_is_rejected(n):
    with pytest.raises(ValueError):
        generate_synthetic_recoveries(n=n)


def test_invalid_scenario_is_rejected():
    with pytest.raises(ValueError):
        generate_synthetic_recoveries(
            n=100,
            scenario="invalid",
        )