import matplotlib.pyplot as plt

from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    BANDIT_SERIES,
    HORIZON,
    benchmark_simulate_kwargs,
)
from simulation import run_series_simulations


c_values = [1,3,7,30,60,90]

series = BANDIT_SERIES
output_dir = benchmark_output_dir(module_file=__file__, output_subdir="sweep_c")
output_path = output_dir / "final_regret_vs_c.png"


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_means = {spec.label or spec.policy_name: [] for spec in series}
    final_stds = {spec.label or spec.policy_name: [] for spec in series}

    for c in c_values:
        _, results = run_series_simulations(
            series=series,
            simulate_kwargs=benchmark_simulate_kwargs(c=float(c)),
        )

        for spec in series:
            label = spec.label or spec.policy_name
            mean_curve, std_curve = results[label]
            final_means[label].append(float(mean_curve[-1]))
            final_stds[label].append(float(std_curve[-1]))

    plt.figure(figsize=(7, 5))
    for spec in series:
        label = spec.label or spec.policy_name
        plt.errorbar(
            c_values,
            final_means[label],
            yerr=final_stds[label],
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel("Switching cost c")
    plt.ylabel(f"Regret at T = {HORIZON}")
    plt.yscale("log")
    plt.title("Final Regret vs Switching Cost")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved final-regret sweep to {output_path}")


if __name__ == "__main__":
    main()
