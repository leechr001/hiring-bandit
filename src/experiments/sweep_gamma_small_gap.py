import numpy as np

from simulation import (
    ExperimentSeries, 
    plot_regret_series, 
    compute_optimistic_hire_auto_gamma
)


k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
n_runs = 5

base_seed = 12345

rng = np.random.default_rng(base_seed)
means = rng.uniform(0.55, 0.6, size=k).tolist()

gamma_auto = compute_optimistic_hire_auto_gamma(
    k=k,
    m=m,
    T=T,
    c=c,
    omega_max=omega_max,
)

factors = [0.25, 0.5, 1, 2, 4]
gammas = [gamma_auto * f for f in factors]

series = [
    ExperimentSeries(
        policy_name="optimistic-hire",
        label=rf"$\gamma={f}\cdot \gamma^*$",
        sim_kwargs={"gamma": float(gamma_auto * f)},
        plot_kwargs={"linewidth": 2},
    )
    for f in factors
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
    title=rf"Cumulative regret for a small gap instance ($\Delta\leq 0.05$).",
    xlabel="t",
    ylabel="Cumulative regret",
    figure_kwargs={"figsize": (8, 5)},
    grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
)
