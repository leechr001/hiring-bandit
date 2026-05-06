import matplotlib.pyplot as plt

from experiments.helpers import benchmark_output_dir

from experiments.simulation_setups.config_delay import (
    HORIZON,
    bandit_series,
    benchmark_simulate_kwargs,
)
from simulation import run_series_simulations


mean_delay_values = [0, 1, 3, 7]

output_dir = benchmark_output_dir(module_file=__file__, output_subdir="sweep_delay")
output_path = output_dir / "final_regret_vs_delay.png"


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [spec.label or spec.policy_name for spec in bandit_series()]
    final_means = {label: [] for label in labels}
    final_stds = {label: [] for label in labels}

    for mean_delay in mean_delay_values:
        series = bandit_series(omega_mean=float(mean_delay))
        _, results = run_series_simulations(
            series=series,
            simulate_kwargs=benchmark_simulate_kwargs(omega_mean=float(mean_delay)),
        )

        for spec in series:
            label = spec.label or spec.policy_name
            mean_curve, std_curve = results[label]
            final_means[label].append(float(mean_curve[-1]))
            final_stds[label].append(float(std_curve[-1]))

    plt.figure(figsize=(7, 5))
    for label in labels:
        plt.errorbar(
            mean_delay_values,
            final_means[label],
            yerr=final_stds[label],
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel(r"Mean geometric delay $\bar{\omega}$")
    plt.ylabel(f"Regret at T = {HORIZON}")
    plt.title("Final Regret vs Delay")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved final-regret delay sweep to {output_path}")


if __name__ == "__main__":
    main()
