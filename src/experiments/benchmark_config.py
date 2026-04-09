from pathlib import Path

import numpy as np

from simulation import ExperimentSeries


series = [
    ExperimentSeries(
        policy_name="optimistic-hire-auto",
        label="Optimistic-Hire",
    ),
    ExperimentSeries(
        policy_name="AHT",
        label="AgrawalHegdeTeneketzis"),
    ExperimentSeries(
        policy_name="OMM",
        label="OMM"),
    ExperimentSeries(
        policy_name="SemiAnnualReview",
        label="Semi-Annual Review",
    ),
    ExperimentSeries(
        policy_name="WorkTrial",
        label="WorkTrial",
    ),
    ExperimentSeries(
        policy_name="Threshold-0.6",
        label="Threshold (0.6)"),
]

k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
n_runs = 25
rng = np.random.default_rng(12345)
means = rng.uniform(0.3, 0.7, size=k).tolist()

simulate_kwargs = {
    "k": k,
    "m": m,
    "T": T,
    "means": means,
    "c": c,
    "omega_max": omega_max,
    "delay_lower": 8,
    "n_runs": n_runs,
    "n_jobs": 4,
    "review_interval": 6 * 30 * 24,
    "work_trial_periods": 1,
    "work_trial_rotation_periods": 90 * 24,
}

cumulative_title = "Cumulative Regret of Optimistic-Hire Compared to Benchmarks"
average_title = "Normalized Loss of Optimistic-Hire Compared to Benchmarks"

# Curve cache mode for benchmark.py:
#   - "auto": load benchmark_curves.npz when present, otherwise rerun simulations
#   - "load": rebuild plots/tables from benchmark_curves.npz only
#   - "regenerate": rerun simulations and overwrite benchmark_curves.npz
curve_cache_mode = "regenerate"

planning_horizons = [
    ("1 month", 30 * 24),
    ("12 months", 365 * 24),
    ("3 years", 3 * 365 * 24),
    ("5 years", 5 * 365 * 24),
]

output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "benchmark"
