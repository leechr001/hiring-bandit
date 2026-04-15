from experiments.benchmark_common import (
    DEFAULT_AVERAGE_TITLE,
    DEFAULT_CUMULATIVE_TITLE,
    DEFAULT_PLANNING_HORIZONS,
    make_benchmark_means,
    run_benchmark,
)
from simulation import ExperimentSeries

import numpy as np

k = 150
m = 50
T = 5 * 365 * 24
c = 0
omega_max = 8
delay_lower = 8
n_runs = 20
n_jobs = min(4, n_runs)

rng = np.random.default_rng(12345)
means = rng.uniform(0.3, 0.7, size=k).tolist()

series = [
    ExperimentSeries(
        policy_name="optimistic-hire",
        label="Optimistic-Hire",
        sim_kwargs={"gamma": "auto"},
    ),
    ExperimentSeries(
        policy_name="AHT",
        label="AgrawalHegdeTeneketzis",
    ),
    ExperimentSeries(
        policy_name="OMM",
        label="OMM",
    ),
    # ExperimentSeries(
    #     policy_name="SemiAnnualReview",
    #     label="Semi-Annual Review",
    # ),
    # ExperimentSeries(
    #     policy_name="WorkTrial",
    #     label="WorkTrial",
    # ),
    # ExperimentSeries(
    #     policy_name="Threshold",
    #     label="Threshold (0.6)",
    #     sim_kwargs={"threshold": 0.6},
    # ),
]

simulate_kwargs = {
    "k": k,
    "m": m,
    "T": T,
    "means": means,
    "c": c,
    "omega_max": omega_max,
    "delay_lower": delay_lower,
    "n_runs": n_runs,
    "n_jobs": n_jobs,
    "review_interval": 6 * 30 * 24,
    "work_trial_periods": 1,
    "work_trial_rotation_periods": 90 * 24,
}

cumulative_title = DEFAULT_CUMULATIVE_TITLE
average_title = DEFAULT_AVERAGE_TITLE
curve_cache_mode = "disabled"
planning_horizons = DEFAULT_PLANNING_HORIZONS
show_plots = True
save_artifacts = False


def main() -> None:
    run_benchmark(
        series=series,
        simulate_kwargs=simulate_kwargs,
        cumulative_title=cumulative_title,
        average_title=average_title,
        curve_cache_mode=curve_cache_mode,
        planning_horizons=planning_horizons,
        save_artifacts=save_artifacts,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
