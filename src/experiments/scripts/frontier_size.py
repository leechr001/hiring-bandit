import matplotlib.pyplot as plt
import numpy as np

from choose_target import ChooseTargetFrontierSizeRecord
from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    BASE_SEED,
    HORIZON,
    K,
    M,
    N_JOBS,
    N_RUNS,
    OMEGA_MEAN,
    PERFORMANCE_MEANS,
    SWITCHING_COST,
)
from simulation import simulate


show_plot = False

output_dir = benchmark_output_dir(module_file=__file__, output_subdir="frontier_size")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "delayed_replace_ucb_frontier_size.png"
trend_grid_size = 300
trend_bandwidth = HORIZON / 40.0


def _decision_time(record: ChooseTargetFrontierSizeRecord) -> int:
    if record.current_period is not None:
        return record.current_period
    return record.time_index


def _collect_max_frontier_points(
    frontier_log: list[ChooseTargetFrontierSizeRecord],
) -> tuple[np.ndarray, np.ndarray]:
    max_frontier_by_seed_and_time: dict[int, dict[int, int]] = {}

    for record in frontier_log:
        episode_seed = record.episode_seed
        if episode_seed is None:
            episode_seed = -1
        decision_time = _decision_time(record)

        per_time = max_frontier_by_seed_and_time.setdefault(episode_seed, {})
        previous = per_time.get(decision_time)
        if previous is None or record.frontier_size > previous:
            per_time[decision_time] = record.frontier_size

    times: list[float] = []
    sizes: list[float] = []
    for seed in sorted(max_frontier_by_seed_and_time):
        for decision_time, frontier_size in sorted(max_frontier_by_seed_and_time[seed].items()):
            times.append(float(decision_time))
            sizes.append(float(frontier_size))

    return np.asarray(times, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def _kernel_smooth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        raise ValueError("Need at least one point to smooth.")
    if x.size == 1:
        return x.copy(), y.copy()

    effective_bandwidth = max(float(bandwidth), 1.0)
    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), num=grid_size)
    smoothed = np.empty_like(x_grid)

    for idx, x_center in enumerate(x_grid):
        weights = np.exp(-0.5 * ((x - x_center) / effective_bandwidth) ** 2)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            smoothed[idx] = float(np.mean(y))
        else:
            smoothed[idx] = float(np.dot(weights, y) / weight_sum)

    return x_grid, smoothed


def main() -> None:
    frontier_log: list[ChooseTargetFrontierSizeRecord] = []

    simulate(
        policies=["delayed-replace-ucb"],
        k=K,
        m=M,
        T=HORIZON,
        means=PERFORMANCE_MEANS,
        c=SWITCHING_COST,
        delay_process_name="geometric",
        omega_mean=OMEGA_MEAN,
        gamma=f"auto={OMEGA_MEAN}",
        n_runs=N_RUNS,
        n_jobs=N_JOBS,
        seed0=BASE_SEED,
        frontier_size_log=frontier_log,
    )

    if not frontier_log:
        raise RuntimeError("No frontier-size records were collected.")

    decision_times, frontier_sizes = _collect_max_frontier_points(frontier_log)
    trend_x, trend_y = _kernel_smooth(
        decision_times,
        frontier_sizes,
        bandwidth=trend_bandwidth,
        grid_size=trend_grid_size,
    )
    output_path.unlink(missing_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(
        decision_times,
        frontier_sizes,
        color="#1f77b4",
        s=18,
        alpha=0.35,
        edgecolors="none",
        label="Observed max frontier size",
    )
    ax.plot(
        trend_x,
        trend_y,
        color="#d62728",
        linewidth=2.5,
        label="Smoothed trend",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("Frontier size")
    ax.set_title(f"Frontier Size Over Time ({N_RUNS} runs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)
    print(f"Saved frontier-size plot to {output_path}")


if __name__ == "__main__":
    main()
