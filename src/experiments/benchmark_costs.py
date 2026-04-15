from pathlib import Path

import numpy as np

from simulation import ExperimentSeries, plot_regret_series, run_series_simulations


k = 150
m = 100
T = 5 * 365 * 24
c_values = [0, 24, 168]
omega_max = 8
delay_lower = 8
n_runs = 20
n_jobs = min(4, n_runs)
base_seed = 12345
show_plots = False

rng = np.random.default_rng(base_seed)
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
]

output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "benchmark_costs"
output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for c in c_values:
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

        means_out, results = run_series_simulations(
            series=series,
            simulate_kwargs=simulate_kwargs,
        )

        output_path = output_dir / f"benchmark_cost_{c}.png"
        output_path.unlink(missing_ok=True)

        plot_regret_series(
            series=series,
            simulate_kwargs=simulate_kwargs,
            title=f"Cumulative Regret Benchmark (c = {c})",
            ylim=(0, 250000) if c == 168 else None,
            save_path=str(output_path),
            show_plot=show_plots,
            precomputed=(means_out, results),
        )

        print(f"Saved cumulative regret plot to {output_path}")


if __name__ == "__main__":
    main()
