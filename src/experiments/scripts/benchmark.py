from experiments.helpers import run_benchmark
from experiments.simulation_setups.config_main import (
    BENCHMARK_SERIES,
    HEURISTIC_SERIES,
    BANDIT_SERIES,
    DEFAULT_AVERAGE_TITLE,
    DEFAULT_CUMULATIVE_TITLE,
    DEFAULT_PLANNING_HORIZONS,
    SIMULATE_KWARGS,
)


series = BENCHMARK_SERIES

simulate_kwargs = SIMULATE_KWARGS

cumulative_title = DEFAULT_CUMULATIVE_TITLE
average_title = DEFAULT_AVERAGE_TITLE
curve_cache_mode = "disabled"
planning_horizons = DEFAULT_PLANNING_HORIZONS
show_plots = True
save_artifacts = False
cumulative_ylim = None
normalized_ylim = (0, 1)


def main() -> None:
    run_benchmark(
        series=series,
        simulate_kwargs=simulate_kwargs,
        cumulative_title=cumulative_title,
        average_title=average_title,
        curve_cache_mode=curve_cache_mode,
        planning_horizons=planning_horizons,
        cumulative_ylim=cumulative_ylim,
        normalized_ylim=normalized_ylim,
        save_artifacts=save_artifacts,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    main()
