"""How sensitive are portfolio credit-loss estimates to the assumed recovery
distribution when expected recovery is held constant?

All data are synthetic. Results are illustrative, not estimates of any
real financial institution's credit risk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import beta

from src.data_simulator import generate_synthetic_recoveries
from src.density_estimators import (
    bounded_kde_density,
    fit_parametric_beta,
    fit_reflection_kde,
)
from src.risk_metrics import (
    economic_capital,
    expected_loss,
    expected_shortfall,
    value_at_risk,
)


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

SEED = 2026

N_OBSERVATIONS = 5_000
N_OBLIGORS = 100
EXPOSURE_PER_OBLIGOR = 100_000.0
DEFAULT_PROBABILITY = 0.05

N_SIMULATIONS = 10_000
N_REPEATED_EXPERIMENTS = 30

CONFIDENCE_LEVELS = (0.95, 0.99)
RECOVERY_GRID_SIZE = 20_001

RISK_METRICS = (
    "Expected loss",
    "VaR 95%",
    "ES 95%",
    "Economic capital 95%",
    "VaR 99%",
    "ES 99%",
    "Economic capital 99%",
)

REPEATED_GAP_METRICS = (
    "VaR 99%",
    "ES 99%",
    "Economic capital 99%",
)


# ---------------------------------------------------------------------
# 2. Density utilities
# ---------------------------------------------------------------------

def normalize_density(
    grid: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Validate and normalize a density using trapezoidal integration."""
    grid = np.asarray(grid, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    if grid.ndim != 1 or density.ndim != 1:
        raise ValueError("Grid and density must be one-dimensional.")

    if grid.shape != density.shape or grid.size < 2:
        raise ValueError("Grid and density must have equal valid lengths.")

    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(density)):
        raise ValueError("Grid and density must contain finite values.")

    if np.any(np.diff(grid) <= 0):
        raise ValueError("Grid must be strictly increasing.")

    if np.any(density < 0):
        raise ValueError("Density cannot contain negative values.")

    area = float(np.trapezoid(density, grid))

    if not np.isfinite(area) or area <= 0:
        raise ValueError("Density must have a positive finite integral.")

    return density / area


def match_density_mean(
    grid: np.ndarray,
    density: np.ndarray,
    target_mean: float,
) -> tuple[np.ndarray, float]:
    """Exponentially tilt a density to match a target recovery mean."""
    if not np.isfinite(target_mean) or not 0.0 < target_mean < 1.0:
        raise ValueError("Target mean must lie strictly between 0 and 1.")

    density = normalize_density(grid, density)

    def tilted_mean(lam: float) -> float:
        # Subtracting 0.5 improves numerical scaling.
        weights = density * np.exp(lam * (grid - 0.5))
        area = float(np.trapezoid(weights, grid))

        if not np.isfinite(area) or area <= 0:
            raise ValueError("Invalid tilted density.")

        return float(np.trapezoid(grid * weights, grid) / area)

    lower, upper = -100.0, 100.0

    if not tilted_mean(lower) <= target_mean <= tilted_mean(upper):
        raise ValueError("Target mean is outside the attainable range.")

    lam = brentq(
        lambda value: tilted_mean(value) - target_mean,
        lower,
        upper,
    )

    weights = density * np.exp(lam * (grid - 0.5))
    adjusted = normalize_density(grid, weights)

    return adjusted, float(lam)


def density_mean(
    grid: np.ndarray,
    density: np.ndarray,
) -> float:
    """Calculate the numerical mean represented by a density."""
    return float(np.trapezoid(grid * density, grid))


def density_cdf(
    grid: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """Construct a normalized numerical cumulative distribution."""
    density = normalize_density(grid, density)

    cdf = np.zeros_like(grid, dtype=np.float64)
    cdf[1:] = np.cumsum(
        0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    )

    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("Could not construct a valid CDF.")

    cdf /= cdf[-1]
    cdf = np.maximum.accumulate(cdf)
    cdf[-1] = 1.0

    return cdf


def sample_density(
    rng: np.random.Generator,
    grid: np.ndarray,
    density: np.ndarray,
    size: int,
) -> np.ndarray:
    """Sample recovery rates by inverse interpolation of a numerical CDF."""
    if size < 1:
        raise ValueError("Sample size must be positive.")

    cdf = density_cdf(grid, density)

    # Remove repeated CDF values before inverse interpolation.
    unique_cdf, indices = np.unique(cdf, return_index=True)

    uniforms = rng.random(size)

    return np.interp(
        uniforms,
        unique_cdf,
        grid[indices],
    )


# ---------------------------------------------------------------------
# 3. Portfolio simulation and risk metrics
# ---------------------------------------------------------------------

def simulate_losses(
    rng: np.random.Generator,
    recovery_grid: np.ndarray,
    recovery_density: np.ndarray,
    defaults: np.ndarray,
) -> np.ndarray:
    """Simulate total portfolio loss from fixed default outcomes."""
    if defaults.ndim != 2 or defaults.shape[1] != N_OBLIGORS:
        raise ValueError(
            f"defaults must have shape (n_simulations, {N_OBLIGORS})."
        )

    if defaults.shape[0] == 0:
        raise ValueError("At least one portfolio scenario is required.")

    recoveries = sample_density(
        rng,
        recovery_grid,
        recovery_density,
        defaults.size,
    ).reshape(defaults.shape)

    individual_losses = (
        defaults.astype(np.float64)
        * EXPOSURE_PER_OBLIGOR
        * (1.0 - recoveries)
    )

    return individual_losses.sum(axis=1)


def calculate_metrics(
    losses: np.ndarray,
) -> dict[str, float]:
    """Calculate expected loss, VaR, ES, and economic capital."""
    results = {
        "Expected loss": expected_loss(losses),
    }

    for confidence in CONFIDENCE_LEVELS:
        label = f"{confidence:.0%}"

        results[f"VaR {label}"] = value_at_risk(
            losses,
            confidence,
        )

        results[f"ES {label}"] = expected_shortfall(
            losses,
            confidence,
        )

        results[f"Economic capital {label}"] = economic_capital(
            losses,
            confidence,
        )

    return results


def generate_default_scenarios(
    rng: np.random.Generator,
    n_simulations: int,
) -> np.ndarray:
    """Generate independent obligor defaults for each portfolio scenario."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive.")

    return (
        rng.random((n_simulations, N_OBLIGORS))
        < DEFAULT_PROBABILITY
    )


# ---------------------------------------------------------------------
# 4. Repeated Monte Carlo experiments
# ---------------------------------------------------------------------

def run_repeated_experiments(
    grid: np.ndarray,
    beta_density: np.ndarray,
    kde_density: np.ndarray,
    n_experiments: int = N_REPEATED_EXPERIMENTS,
) -> pd.DataFrame:
    """Compare model risk estimates across independent simulation runs.

    The fitted recovery densities remain fixed. Each experiment generates
    new default scenarios and new recovery samples. Both models share the
    same default outcomes within each experiment.
    """
    if n_experiments < 2:
        raise ValueError("At least two repeated experiments are required.")

    rows: list[dict[str, float | int]] = []

    for experiment in range(n_experiments):
        # Independent and reproducible random stream for each experiment.
        rng = np.random.default_rng(SEED + 10_000 + experiment)

        defaults = generate_default_scenarios(
            rng,
            N_SIMULATIONS,
        )

        beta_losses = simulate_losses(
            rng,
            grid,
            beta_density,
            defaults,
        )

        kde_losses = simulate_losses(
            rng,
            grid,
            kde_density,
            defaults,
        )

        beta_metrics = calculate_metrics(beta_losses)
        kde_metrics = calculate_metrics(kde_losses)

        row: dict[str, float | int] = {
            "Experiment": experiment + 1,
        }

        # Preserve all risk metrics for both models.
        for metric in RISK_METRICS:
            row[f"Beta {metric}"] = beta_metrics[metric]
            row[f"KDE {metric}"] = kde_metrics[metric]
            row[f"Gap {metric}"] = (
                kde_metrics[metric] - beta_metrics[metric]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_repeated_experiments(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the mean gap and its Monte Carlo variability."""
    summary_rows = []

    for metric in REPEATED_GAP_METRICS:
        column = f"Gap {metric}"
        gaps = results[column].to_numpy(dtype=np.float64)

        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps, ddof=1))
        standard_error = std_gap / np.sqrt(gaps.size)

        # Approximate normal 95% confidence interval for the mean gap.
        lower = mean_gap - 1.96 * standard_error
        upper = mean_gap + 1.96 * standard_error

        summary_rows.append(
            {
                "Risk metric": metric,
                "Mean gap ($)": mean_gap,
                "Std. dev. ($)": std_gap,
                "Standard error ($)": standard_error,
                "Approx. 95% CI lower ($)": lower,
                "Approx. 95% CI upper ($)": upper,
                "Experiments with positive gap": int(np.sum(gaps > 0)),
                "Experiments": int(gaps.size),
            }
        )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------
# 5. Diagnostic visualization
# ---------------------------------------------------------------------

def create_diagnostic_plot(
    recoveries: np.ndarray,
    grid: np.ndarray,
    beta_density: np.ndarray,
    kde_density: np.ndarray,
    beta_losses: np.ndarray,
    kde_losses: np.ndarray,
    repeated_results: pd.DataFrame,
) -> Path:
    """Create recovery, portfolio-loss, and repeated-gap diagnostics."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5.8),
    )

    # Panel A: recovery distributions.
    axes[0].hist(
        recoveries,
        bins=70,
        density=True,
        alpha=0.25,
        color="grey",
        label="Synthetic observations",
    )

    axes[0].plot(
        grid,
        beta_density,
        linewidth=2.2,
        label="Mean-matched Beta",
    )

    axes[0].plot(
        grid,
        kde_density,
        linewidth=2.2,
        label="Mean-matched reflected KDE",
    )

    axes[0].set(
        title="Recovery distributions",
        xlabel="Recovery rate",
        ylabel="Density",
        xlim=(0, 1),
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # Panel B: portfolio-loss distributions.
    max_loss = max(
        float(np.max(beta_losses)),
        float(np.max(kde_losses)),
    )

    bins = np.linspace(0.0, max_loss, 55)

    axes[1].hist(
        beta_losses,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Beta portfolio losses",
    )

    axes[1].hist(
        kde_losses,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="KDE portfolio losses",
    )

    beta_var95 = value_at_risk(beta_losses, 0.95)
    kde_var95 = value_at_risk(kde_losses, 0.95)

    axes[1].axvline(
        beta_var95,
        linestyle="--",
        linewidth=1.5,
        label="Beta VaR 95%",
    )

    axes[1].axvline(
        kde_var95,
        linestyle=":",
        linewidth=1.8,
        label="KDE VaR 95%",
    )

    # Shade the empirical upper 5% region of the KDE loss distribution.
    tail_bins = np.linspace(
        kde_var95,
        float(np.max(kde_losses)),
        30,
    )

    if tail_bins.size >= 2 and tail_bins[-1] > tail_bins[0]:
        tail_density, tail_edges = np.histogram(
            kde_losses,
            bins=tail_bins,
            density=True,
        )

        tail_centres = 0.5 * (tail_edges[:-1] + tail_edges[1:])

        axes[1].fill_between(
            tail_centres,
            tail_density,
            step="mid",
            alpha=0.18,
            label="KDE upper 5% tail",
        )

    axes[1].set(
        title="Portfolio loss distributions",
        xlabel="Portfolio loss ($)",
        ylabel="Density",
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    # Panel C: repeated-run tail-risk differences.
    labels = {
        "Gap VaR 99%": "VaR 99%",
        "Gap ES 99%": "ES 99%",
        "Gap Economic capital 99%": "Economic capital 99%",
    }

    for column, label in labels.items():
        axes[2].plot(
            repeated_results["Experiment"],
            repeated_results[column],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )

    axes[2].axhline(0.0, linestyle="--", linewidth=1, color="black")

    axes[2].set(
        title="Repeated-run risk gaps",
        xlabel="Experiment",
        ylabel="KDE minus Beta ($)",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    fig.suptitle(
        "Recovery Risk Calibration — Synthetic Experiment",
        fontsize=14,
    )
    fig.tight_layout()

    plot_path = REPORTS / "diagnostic_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return plot_path


# ---------------------------------------------------------------------
# 6. Research brief
# ---------------------------------------------------------------------

def write_research_brief(
    target_mean: float,
    beta_alpha: float,
    beta_beta: float,
    beta_mean: float,
    kde_mean: float,
    beta_tilt: float,
    kde_tilt: float,
    comparison: pd.DataFrame,
    repeated_summary: pd.DataFrame,
) -> Path:
    """Write the reproducible research brief."""
    comparison_text = comparison.round(2).to_string()
    repeated_text = repeated_summary.round(2).to_string()

    brief = f"""# Recovery Risk Calibration: Research Brief

## Research question

How sensitive are portfolio credit-loss estimates to the assumed recovery
distribution when expected recovery is held constant?

## Executive summary

This experiment compares a parametric Beta recovery distribution with a
bounded reflected kernel density estimate (KDE). Both distributions are
calibrated to the same empirical mean recovery before portfolio losses
are simulated.

The repeated experiments hold the fitted recovery densities fixed and
vary the Monte Carlo random streams. They measure simulation variability,
not uncertainty from refitting the recovery models to new datasets.

All results use synthetic data and illustrative portfolio assumptions.
They are not estimates of real-world credit risk.

## Data and portfolio assumptions

- Synthetic recovery observations: {N_OBSERVATIONS:,}
- Recovery data-generating scenario: bimodal mixture
- Empirical target mean recovery: {target_mean:.6f}
- Number of obligors: {N_OBLIGORS}
- Exposure per obligor: ${EXPOSURE_PER_OBLIGOR:,.2f}
- Total portfolio exposure: ${N_OBLIGORS * EXPOSURE_PER_OBLIGOR:,.2f}
- Default probability per obligor: {DEFAULT_PROBABILITY:.2%}
- Portfolio simulations per experiment: {N_SIMULATIONS:,}
- Repeated experiments: {N_REPEATED_EXPERIMENTS}
- Random seed for the primary experiment: {SEED}

The same default outcomes are used for both recovery models within each
experiment. Recovery samples differ between the two models.

## Recovery-model calibration

- Fitted Beta alpha: {beta_alpha:.6f}
- Fitted Beta beta: {beta_beta:.6f}
- Beta adjusted mean recovery: {beta_mean:.6f}
- KDE adjusted mean recovery: {kde_mean:.6f}
- Beta exponential-tilt parameter: {beta_tilt:.6f}
- KDE exponential-tilt parameter: {kde_tilt:.6f}

Both densities are exponentially tilted to match the empirical recovery
mean. The comparison therefore isolates distributional-shape differences
after this calibration, subject to the estimation and numerical methods.

## Primary portfolio-risk comparison

{comparison_text}

A positive KDE-minus-Beta gap means the KDE estimates a higher value for
that risk metric in the primary simulation.

## Repeated-experiment results

{repeated_text}

The reported confidence intervals are approximate normal intervals for
the mean risk gap across repeated Monte Carlo experiments. They quantify
simulation variability conditional on the fitted densities and model
assumptions; they are not confidence intervals for real-world portfolio
risk or for uncertainty in the fitted recovery distributions.

The fraction of positive gaps helps show how consistently the KDE
estimate exceeds the Beta estimate across these simulation runs.

## Interpretation guidance

A positive mean gap suggests higher estimated risk under KDE for the
specified synthetic experiment. It does not prove that KDE is universally
more conservative or more accurate.

A confidence interval that crosses zero indicates that the average gap
is not clearly distinguishable from zero at the approximate 95% level
under this repeated-simulation design. A confidence interval that excludes
zero still only supports a conditional conclusion for these assumptions.

## Limitations

1. All recovery observations are synthetic, not historical loan data.
2. The fitted recovery densities remain fixed across repeated experiments.
3. The portfolio uses a homogeneous default probability and fixed exposure.
4. Dependence between obligor defaults is not modelled.
5. Recovery dependence on default conditions and exposure is not modelled.
6. The KDE result depends on the bandwidth and boundary correction.
7. The Beta density is evaluated on a finite grid; endpoint values are
   approximated numerically where necessary.
8. Continuous recovery distributions do not explicitly model point masses
   at exactly zero or one.
9. The approximate confidence intervals rely on a normal approximation
   for the repeated-run mean gap.
10. Results depend on the chosen synthetic scenario and random seeds.

## Reproducibility

Run `python main.py` from the project root.

Generated outputs:

- `reports/diagnostic_plot.png`
- `reports/research_brief.md`
- `reports/repeated_experiments.csv`
- `reports/repeated_experiment_summary.csv`

The script uses fixed random seeds to make its results reproducible under
the same software environment and configuration.
"""

    brief_path = REPORTS / "research_brief.md"
    brief_path.write_text(brief, encoding="utf-8")

    return brief_path


# ---------------------------------------------------------------------
# 7. Master pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete recovery calibration research experiment."""
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Step 1: generate synthetic recovery observations.
    recoveries = generate_synthetic_recoveries(
        n=N_OBSERVATIONS,
        scenario="bimodal",
        seed=SEED,
    )

    target_mean = float(np.mean(recoveries))

    # Step 2: fit the two recovery-density estimators.
    beta_alpha, beta_beta = fit_parametric_beta(recoveries)
    kde = fit_reflection_kde(recoveries)

    # Step 3: evaluate both densities on the bounded interval [0, 1].
    grid = np.linspace(
        0.0,
        1.0,
        RECOVERY_GRID_SIZE,
    )

    beta_density = beta.pdf(
        grid,
        beta_alpha,
        beta_beta,
        loc=0.0,
        scale=1.0,
    )

    # Approximate endpoints when the fitted Beta density is singular there.
    beta_density[0] = beta_density[1]
    beta_density[-1] = beta_density[-2]

    beta_density = normalize_density(
        grid,
        beta_density,
    )

    kde_density = bounded_kde_density(
        kde,
        grid,
    )

    # Step 4: match the recovery means.
    beta_adjusted, beta_tilt = match_density_mean(
        grid,
        beta_density,
        target_mean,
    )

    kde_adjusted, kde_tilt = match_density_mean(
        grid,
        kde_density,
        target_mean,
    )

    beta_mean = density_mean(grid, beta_adjusted)
    kde_mean = density_mean(grid, kde_adjusted)

    if not np.isclose(beta_mean, target_mean, atol=1e-5):
        raise RuntimeError("Beta recovery mean did not match its target.")

    if not np.isclose(kde_mean, target_mean, atol=1e-5):
        raise RuntimeError("KDE recovery mean did not match its target.")

    # Step 5: run the primary portfolio experiment.
    primary_rng = np.random.default_rng(SEED + 1)

    defaults = generate_default_scenarios(
        primary_rng,
        N_SIMULATIONS,
    )

    beta_losses = simulate_losses(
        primary_rng,
        grid,
        beta_adjusted,
        defaults,
    )

    kde_losses = simulate_losses(
        primary_rng,
        grid,
        kde_adjusted,
        defaults,
    )

    beta_results = calculate_metrics(beta_losses)
    kde_results = calculate_metrics(kde_losses)

    comparison = pd.DataFrame(
        {
            "Beta model": beta_results,
            "Reflected KDE": kde_results,
        }
    )

    comparison["KDE minus Beta"] = (
        comparison["Reflected KDE"] - comparison["Beta model"]
    )

    # Step 6: run independent repeated Monte Carlo experiments.
    repeated_results = run_repeated_experiments(
        grid,
        beta_adjusted,
        kde_adjusted,
    )

    repeated_summary = summarize_repeated_experiments(
        repeated_results,
    )

    repeated_results_path = REPORTS / "repeated_experiments.csv"
    repeated_summary_path = REPORTS / "repeated_experiment_summary.csv"

    repeated_results.to_csv(
        repeated_results_path,
        index=False,
    )

    repeated_summary.to_csv(
        repeated_summary_path,
        index=False,
    )

    # Step 7: print diagnostics and results.
    print("\n" + "=" * 72)
    print("RECOVERY RISK CALIBRATION")
    print("=" * 72)

    print("\nRECOVERY MODEL DIAGNOSTICS")
    print(f"Synthetic observations: {N_OBSERVATIONS:,}")
    print(f"Target recovery mean:   {target_mean:.6f}")
    print(
        f"Beta parameters:        "
        f"alpha={beta_alpha:.6f}, beta={beta_beta:.6f}"
    )
    print(f"Beta adjusted mean:     {beta_mean:.6f}")
    print(f"KDE adjusted mean:      {kde_mean:.6f}")
    print(f"Beta tilt parameter:    {beta_tilt:.6f}")
    print(f"KDE tilt parameter:     {kde_tilt:.6f}")

    print("\nPORTFOLIO ASSUMPTIONS")
    print(f"Obligors:               {N_OBLIGORS}")
    print(f"Exposure per obligor:   ${EXPOSURE_PER_OBLIGOR:,.2f}")
    print(
        f"Total exposure:         "
        f"${N_OBLIGORS * EXPOSURE_PER_OBLIGOR:,.2f}"
    )
    print(f"Default probability:    {DEFAULT_PROBABILITY:.2%}")
    print(f"Simulations per run:    {N_SIMULATIONS:,}")
    print(f"Repeated experiments:   {N_REPEATED_EXPERIMENTS}")

    print("\nPRIMARY PORTFOLIO RISK COMPARISON")
    print(comparison.round(2).to_string())

    print("\nREPEATED-EXPERIMENT SUMMARY")
    print(repeated_summary.round(2).to_string(index=False))

    positive_gap_frequency = (
        repeated_results["Gap Economic capital 99%"] > 0
    ).mean()

    print(
        "\nPositive 99% economic-capital gaps: "
        f"{positive_gap_frequency:.1%} of repeated experiments"
    )

    # Step 8: create the diagnostic figure.
    plot_path = create_diagnostic_plot(
        recoveries=recoveries,
        grid=grid,
        beta_density=beta_adjusted,
        kde_density=kde_adjusted,
        beta_losses=beta_losses,
        kde_losses=kde_losses,
        repeated_results=repeated_results,
    )

    # Step 9: write the research brief.
    brief_path = write_research_brief(
        target_mean=target_mean,
        beta_alpha=beta_alpha,
        beta_beta=beta_beta,
        beta_mean=beta_mean,
        kde_mean=kde_mean,
        beta_tilt=beta_tilt,
        kde_tilt=kde_tilt,
        comparison=comparison,
        repeated_summary=repeated_summary,
    )

    print("\nGENERATED OUTPUTS")
    print(f"Diagnostic figure:      {plot_path}")
    print(f"Research brief:         {brief_path}")
    print(f"Repeated results:       {repeated_results_path}")
    print(f"Repeated summary:       {repeated_summary_path}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
