from experiments.benchmark_common import benchmark_output_dir, make_benchmark_means
from simulation import ExperimentSeries, plot_regret_series, run_series_simulations


k = 150
m = 50
T = 5 * 365 * 24
c = 8
omega_max = 8
delay_lower = 8
n_runs = 20
n_jobs = min(4, n_runs)
base_seed = 12345

series = [
    ExperimentSeries(
        policy_name="optimistic-hire",
        label="Optimistic-Hire",
        sim_kwargs={"gamma": "auto"},
    ),
    ExperimentSeries(
        policy_name="Threshold",
        label="Threshold (0.3)",
        sim_kwargs={"threshold": 0.3},
    ),
    ExperimentSeries(
        policy_name="Threshold",
        label="Threshold (0.4)",
        sim_kwargs={"threshold": 0.4},
    ),
    ExperimentSeries(
        policy_name="Threshold",
        label="Threshold (0.5)",
        sim_kwargs={"threshold": 0.5},
    ),
    ExperimentSeries(
        policy_name="Threshold",
        label="Threshold (0.6)",
        sim_kwargs={"threshold": 0.6},
    ),
    ExperimentSeries(
        policy_name="Threshold",
        label="Threshold (0.7)",
        sim_kwargs={"threshold": 0.7},
    ),
]

mean_specs = [
    ("uniform_0p1_0p5", 0.1, 0.5),
    ("uniform_0p5_0p9", 0.5, 0.9),
    ("uniform_0p4_0p6", 0.4, 0.6),
    ("uniform_0p1_0p9", 0.1, 0.9),
]

output_dir = benchmark_output_dir(
    module_file=__file__,
    output_subdir="threshold_robustness",
)
output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for mean_slug, low, high in mean_specs:
        means = make_benchmark_means(
            k=k,
            seed=base_seed,
            low=low,
            high=high,
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
            "seed0": base_seed,
        }

        print(f"Running threshold robustness for means ~ U[{low}, {high}]")
        means_out, results = run_series_simulations(
            series=series,
            simulate_kwargs=simulate_kwargs,
        )

        output_path = output_dir / f"{mean_slug}_cumulative_regret.png"
        output_path.unlink(missing_ok=True)
        plot_regret_series(
            series=series,
            simulate_kwargs=simulate_kwargs,
            title=(
                "Threshold Robustness: Cumulative Regret "
                f"(means ~ U[{low}, {high}])"
            ),
            ylim=(0, 100000),
            save_path=str(output_path),
            show_plot=False,
            precomputed=(means_out, results),
        )

        print(f"Saved cumulative regret plot to {output_path}")


if __name__ == "__main__":
    main()
