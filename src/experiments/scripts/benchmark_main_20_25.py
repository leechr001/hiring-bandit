from experiments.helpers import benchmark_output_dir, run_benchmark
from experiments.simulation_setups.config_main_20_25 import (
    CONFIG_NAME,
    BENCHMARK_SERIES,
    DEFAULT_AVERAGE_TITLE,
    DEFAULT_CUMULATIVE_TITLE,
    DEFAULT_PLANNING_HORIZONS,
    SIMULATE_KWARGS,
)


series = BENCHMARK_SERIES
simulate_kwargs = SIMULATE_KWARGS

cumulative_title = DEFAULT_CUMULATIVE_TITLE
average_title = DEFAULT_AVERAGE_TITLE
curve_cache_mode = "regenerate"
planning_horizons = DEFAULT_PLANNING_HORIZONS
cumulative_ylim = None
normalized_ylim = (0, 1)
output_dir = benchmark_output_dir(module_file=__file__, output_subdir=f"benchmark_{CONFIG_NAME}")


def main() -> None:
    run_benchmark(
        series=series,
        simulate_kwargs=simulate_kwargs,
        output_dir=output_dir,
        cumulative_title=cumulative_title,
        average_title=average_title,
        curve_cache_mode=curve_cache_mode,
        planning_horizons=planning_horizons,
        cumulative_ylim=cumulative_ylim,
        normalized_ylim=normalized_ylim,
    )


if __name__ == "__main__":
    main()
