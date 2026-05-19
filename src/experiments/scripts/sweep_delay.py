import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from experiments.helpers import benchmark_output_dir

from experiments.simulation_setups.config_delay import (
    HORIZON,
    bandit_series,
    benchmark_simulate_kwargs,
)
from simulation import _average_regret_results, run_series_simulations


mean_delay_values = [0, 1, 3, 7]

output_dir = benchmark_output_dir(module_file=__file__, output_subdir="sweep_delay")
regret_output_path = output_dir / "final_regret_vs_delay.png"
normalized_output_path = output_dir / "final_normalized_loss_vs_delay.png"


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [spec.label or spec.policy_name for spec in bandit_series()]
    final_means = {label: [] for label in labels}
    final_stds = {label: [] for label in labels}
    final_normalized_means = {label: [] for label in labels}
    final_normalized_stds = {label: [] for label in labels}

    for mean_delay in mean_delay_values:
        series = bandit_series(omega_mean=float(mean_delay))
        simulate_kwargs = benchmark_simulate_kwargs(omega_mean=float(mean_delay))
        means, results = run_series_simulations(
            series=series,
            simulate_kwargs=simulate_kwargs,
        )
        normalized_results = _average_regret_results(
            means,
            int(simulate_kwargs["m"]),
            results,
        )

        for spec in series:
            label = spec.label or spec.policy_name
            mean_curve, std_curve = results[label]
            normalized_mean_curve, normalized_std_curve = normalized_results[label]
            final_means[label].append(float(mean_curve[-1]))
            final_stds[label].append(float(std_curve[-1]))
            final_normalized_means[label].append(float(normalized_mean_curve[-1]))
            final_normalized_stds[label].append(float(normalized_std_curve[-1]))

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

    regret_output_path.unlink(missing_ok=True)
    plt.savefig(regret_output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved final-regret delay sweep to {regret_output_path}")

    plt.figure(figsize=(7, 5))
    for label in labels:
        plt.errorbar(
            mean_delay_values,
            final_normalized_means[label],
            yerr=final_normalized_stds[label],
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel(r"Mean geometric delay $\bar{\omega}$")
    plt.ylabel(f"Normalized loss at T = {HORIZON}")
    plt.title("Final Normalized Loss vs Delay")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    normalized_output_path.unlink(missing_ok=True)
    plt.savefig(normalized_output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved normalized-loss delay sweep to {normalized_output_path}")


if __name__ == "__main__":
    main()
