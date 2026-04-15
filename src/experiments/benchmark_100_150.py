from experiments.benchmark_common import (
    DEFAULT_AVERAGE_TITLE,
    DEFAULT_CUMULATIVE_TITLE,
    DEFAULT_PLANNING_HORIZONS,
    benchmark_output_dir,
    default_benchmark_series,
    make_benchmark_means,
    run_benchmark,
)

k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
delay_lower = 8
n_runs = 25
n_jobs = 4

means = make_benchmark_means(k=k)
series = default_benchmark_series(
    threshold_value=0.35,
    threshold_label="Tuned Threshold (0.35)",
)

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
curve_cache_mode = "regenerate"
planning_horizons = DEFAULT_PLANNING_HORIZONS
output_dir = benchmark_output_dir(module_file=__file__, output_subdir="benchmark_100_150")

def main() -> None:
    run_benchmark(
        series=series,
        simulate_kwargs=simulate_kwargs,
        output_dir=output_dir,
        cumulative_title=cumulative_title,
        average_title=average_title,
        curve_cache_mode=curve_cache_mode,
        planning_horizons=planning_horizons,
    )


if __name__ == "__main__":
    main()
