from pathlib import Path

import numpy as np

from simulation import (
    ExperimentSeries,
    compute_optimistic_hire_auto_gamma,
    plot_regret_series,
    run_series_simulations,
)


k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_values = [
    # 0, 
    # 24, 
    168]
delay_processes = ["stochastic"]
n_runs = 20
n_jobs = 4
base_seed = 12345
show_plots = False

rng = np.random.default_rng(base_seed)
means = rng.uniform(0.3, 0.7, size=k).tolist()

output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "benchmark_delays"
output_dir.mkdir(parents=True, exist_ok=True)


def build_series(*, omega_max: int) -> list[ExperimentSeries]:
    oh_gamma = compute_optimistic_hire_auto_gamma(
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=0,
    )

    return [
        ExperimentSeries(
            policy_name="optimistic-hire",
            label="Optimistic-Hire",
            sim_kwargs={"gamma": oh_gamma},
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


def main() -> None:
    for omega_max in omega_values:
        for delay_process_name in delay_processes:
            series = build_series(omega_max=omega_max)
            simulate_kwargs = {
                "k": k,
                "m": m,
                "T": T,
                "means": means,
                "c": c,
                "omega_max": omega_max,
                "delay_process_name": delay_process_name,
                "n_runs": n_runs,
                "n_jobs": n_jobs,
                "seed0": base_seed,
            }

            means_out, results = run_series_simulations(
                series=series,
                simulate_kwargs=simulate_kwargs,
            )

            output_path = (
                output_dir / f"benchmark_delay_{delay_process_name}_omega_{omega_max}.png"
            )
            output_path.unlink(missing_ok=True)

            plot_regret_series(
                series=series,
                simulate_kwargs=simulate_kwargs,
                title=(
                    "Cumulative Regret Benchmark "
                    f"({delay_process_name} delays, "
                    rf"$\omega_{{\max}} = {omega_max}$)"
                ),
                save_path=str(output_path),
                show_plot=show_plots,
                precomputed=(means_out, results),
            )

            print(f"Saved cumulative regret plot to {output_path}")


if __name__ == "__main__":
    main()
