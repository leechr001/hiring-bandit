from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    HORIZON,
    bandit_series,
    benchmark_simulate_kwargs,
)
from experiments.scripts.sweep_c_replacements import _count_completed_replacements
from simulation import _merge_sim_kwargs


mean_delay_values = [0, 1, 3, 7]

output_dir = benchmark_output_dir(
    module_file=__file__,
    output_subdir="sweep_delay_replacements",
)
output_path = output_dir / "completed_replacements_vs_delay.png"


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [spec.label or spec.policy_name for spec in bandit_series()]
    completed_means = {label: [] for label in labels}
    completed_stds = {label: [] for label in labels}

    for mean_delay in mean_delay_values:
        series = bandit_series(omega_mean=float(mean_delay))
        base_simulate_kwargs = benchmark_simulate_kwargs(omega_mean=float(mean_delay))
        n_runs = int(base_simulate_kwargs["n_runs"])
        seed0 = int(base_simulate_kwargs["seed0"])

        for spec in series:
            label = spec.label or spec.policy_name
            simulate_kwargs = _merge_sim_kwargs(base_simulate_kwargs, spec.sim_kwargs)
            completed_counts = [
                _count_completed_replacements(
                    policy_name=spec.policy_name,
                    simulate_kwargs=simulate_kwargs,
                    seed=seed0 + run_idx,
                )
                for run_idx in range(n_runs)
            ]
            completed = np.asarray(completed_counts, dtype=np.float64)
            completed_means[label].append(float(completed.mean()))
            completed_stds[label].append(float(completed.std()))

    plt.figure(figsize=(7, 5))
    for label in labels:
        plt.errorbar(
            mean_delay_values,
            completed_means[label],
            yerr=completed_stds[label],
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel(r"Mean geometric delay $\bar{\omega}$")
    plt.ylabel(f"Replacements by T = {HORIZON}")
    plt.title("Replacements vs Delay")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved completed-replacements delay sweep to {output_path}")


if __name__ == "__main__":
    main()
