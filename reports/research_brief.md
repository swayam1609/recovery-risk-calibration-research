# Recovery Risk Calibration: Research Brief

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

- Synthetic recovery observations: 5,000
- Recovery data-generating scenario: bimodal mixture
- Empirical target mean recovery: 0.496011
- Number of obligors: 100
- Exposure per obligor: $100,000.00
- Total portfolio exposure: $10,000,000.00
- Default probability per obligor: 5.00%
- Portfolio simulations per experiment: 10,000
- Repeated experiments: 30
- Random seed for the primary experiment: 2026

The same default outcomes are used for both recovery models within each
experiment. Recovery samples differ between the two models.

## Recovery-model calibration

- Fitted Beta alpha: 1.008270
- Fitted Beta beta: 1.025455
- Beta adjusted mean recovery: 0.496011
- KDE adjusted mean recovery: 0.496011
- Beta exponential-tilt parameter: 0.002857
- KDE exponential-tilt parameter: -0.001206

Both densities are exponentially tilted to match the empirical recovery
mean. The comparison therefore isolates distributional-shape differences
after this calibration, subject to the estimation and numerical methods.

## Primary portfolio-risk comparison

                      Beta model  Reflected KDE  KDE minus Beta
Expected loss          249925.29      250257.71          332.42
VaR 95%                478593.16      486460.00         7866.84
ES 95%                 554422.25      559092.76         4670.50
Economic capital 95%   228667.87      236202.29         7534.43
VaR 99%                595659.36      603072.14         7412.79
ES 99%                 659112.57      667314.95         8202.38
Economic capital 99%   345734.07      352814.43         7080.37

A positive KDE-minus-Beta gap means the KDE estimates a higher value for
that risk metric in the primary simulation.

## Repeated-experiment results

            Risk metric  Mean gap ($)  Std. dev. ($)  Standard error ($)  Approx. 95% CI lower ($)  Approx. 95% CI upper ($)  Experiments with positive gap  Experiments
0               VaR 99%       9901.35        7726.43             1410.65                   7136.49                  12666.22                             25           30
1                ES 99%      12417.68        9507.12             1735.76                   9015.60                  15819.76                             27           30
2  Economic capital 99%       9776.02        7219.62             1318.12                   7192.51                  12359.53                             28           30

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
