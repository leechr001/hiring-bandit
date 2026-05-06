from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    bandit_series,
    benchmark_simulate_kwargs,
)
from simulation import plot_regret_series, run_series_simulations


omega_means = [0, 3, 7, 21]

show_plots = False
output_dir = benchmark_output_dir(module_file=__file__, output_subdir="benchmark_delays")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for omega_mean in omega_means:
        series = bandit_series(omega_mean=float(omega_mean))
        simulate_kwargs = benchmark_simulate_kwargs(omega_mean=float(omega_mean))

        means_out, results = run_series_simulations(
            series=series,
            simulate_kwargs=simulate_kwargs,
        )

        output_path = output_dir / f"benchmark_delay_omega_{omega_mean}.png"
        output_path.unlink(missing_ok=True)

        plot_regret_series(
            series=series,
            simulate_kwargs=simulate_kwargs,
            title=(
                "Cumulative Regret Benchmark "
                rf"($\bar\omega = {omega_mean}$)"
            ),
            save_path=str(output_path),
            show_plot=show_plots,
            precomputed=(means_out, results),
        )

        print(f"Saved cumulative regret plot to {output_path}")


if __name__ == "__main__":
    main()
