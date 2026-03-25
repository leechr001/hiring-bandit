import numpy as np

from simulation import ExperimentSeries, plot_regret_series


k = 10
m = 3
T = 20000
means = np.linspace(0.3, 0.7, k).tolist()

c = 10
omega_max = 10
gammas = [1, 5, 10, 15, 30, 100]
n_runs = 20

series = [
    ExperimentSeries(
        policy_name="optimistic-hire",
        label=rf"$\gamma={float(gamma):.2f}$",
        sim_kwargs={"gamma": float(gamma)},
        plot_kwargs={"linewidth": 2},
    )
    for gamma in gammas
]

plot_regret_series(
    series=series,
    simulate_kwargs={
        "k": k,
        "m": m,
        "T": T,
        "means": means,
        "c": c,
        "omega_max": omega_max,
        "n_runs": n_runs,
    },
    title=rf"Average regret over {n_runs} runs with $c = {c}$ and $\omega_\max = {omega_max}$",
    xlabel="t",
    ylabel="Cumulative regret",
    figure_kwargs={"figsize": (8, 5)},
    ylim=(0, 2500),
    grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
)
