
import numpy as np
import pytest

from src.risk_metrics import (
    expected_loss,
    value_at_risk,
    expected_shortfall,
    economic_capital,
    capital_gap,
)


def test_expected_loss():
    losses = np.array([10.0, 20.0, 30.0, 40.0])
    assert expected_loss(losses) == pytest.approx(25.0)


def test_var_on_known_losses():
    losses = np.array([10.0, 20.0, 30.0, 40.0])
    assert value_at_risk(losses, 0.50) == pytest.approx(25.0)


def test_expected_shortfall_on_known_losses():
    losses = np.array([10.0, 20.0, 30.0, 40.0])
    assert expected_shortfall(losses, 0.50) == pytest.approx(35.0)


def test_economic_capital():
    losses = np.array([10.0, 20.0, 30.0, 40.0])
    assert economic_capital(losses, 0.50) == pytest.approx(0.0)


def test_capital_gap():
    beta_losses = np.array([10.0, 20.0, 30.0, 40.0])
    kde_losses = np.array([20.0, 30.0, 40.0, 50.0])

    result = capital_gap(beta_losses, kde_losses, confidence=0.50)

    assert result["beta_economic_capital"] == pytest.approx(0.0)
    assert result["kde_economic_capital"] == pytest.approx(0.0)
    assert result["signed_capital_gap_kde_minus_beta"] == pytest.approx(0.0)
    assert result["absolute_capital_gap"] == pytest.approx(0.0)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_invalid_confidence_is_rejected(confidence):
    losses = np.array([10.0, 20.0, 30.0, 40.0])

    with pytest.raises(ValueError):
        value_at_risk(losses, confidence)

    with pytest.raises(ValueError):
        expected_shortfall(losses, confidence)


def test_negative_losses_are_rejected():
    losses = np.array([10.0, -20.0, 30.0])

    with pytest.raises(ValueError):
        expected_loss(losses)