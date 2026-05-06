from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    BANDIT_SERIES,
    benchmark_simulate_kwargs,
)
from simulation import plot_regret_series, run_series_simulations


c_values = [0, 3, 7, 21]

series = BANDIT_SERIES
show_plots = False
output_dir = benchmark_output_dir(module_file=__file__, output_subdir="benchmark_costs")


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in c_values:
        simulate_kwargs = benchmark_simulate_kwargs(c=float(c))
        means_out, results = run_series_simulations(
            series=series,
            simulate_kwargs=simulate_kwargs,
        )

        output_path = output_dir / f"benchmark_cost_{c}.png"
        output_path.unlink(missing_ok=True)

        plot_regret_series(
            series=series,
            simulate_kwargs=simulate_kwargs,
            title=f"Regret Benchmark (c = {c})",
            save_path=str(output_path),
            show_plot=show_plots,
            precomputed=(means_out, results),
        )

        print(f"Saved cumulative regret plot to {output_path}")


if __name__ == "__main__":
    main()
