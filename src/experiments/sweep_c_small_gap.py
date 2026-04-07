import numpy as np

from simulation import ExperimentSeries, plot_regret_series


k = 10
m = 3
T = 20000
means = np.linspace(0.45, 0.5, k).tolist()

omega_max = 5
c_values = [1, 5, 10, 15, 30]
n_runs = 20
n_jobs = 4

for policy_name in ["optimistic-hire-gamma-1", "optimistic-hire-gamma-2"]:
    series = [
        ExperimentSeries(
            policy_name=policy_name,
            label=rf"$c={float(c):.2f}$",
            sim_kwargs={"c": float(c)},
            plot_kwargs={"linewidth": 2},
        )
        for c in c_values
    ]

    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "omega_max": omega_max,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
        },
        title=rf"Average regret over {n_runs} runs with $\omega_\max = {omega_max}$",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, 3500),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
    )
