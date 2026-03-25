import numpy as np

from simulation import ExperimentSeries, plot_regret_series


series = [
    ExperimentSeries(
        policy_name="optimistic-hire-gamma-1",
        label=r"Optimistic-Hire, $\gamma=(c+\omega_\max)^2 m$",
    ),
    ExperimentSeries(
        policy_name="optimistic-hire-gamma-2",
        label=r"Optimistic-Hire, $\gamma=(c+\omega_\max) m$",
    ),
    ExperimentSeries(policy_name="AHT", label="AgrawalHegdeTeneketzis"),
]

k = 10
m = 3
T = 20000
c = 5
omega_max = 5
n_runs = 25
means = np.linspace(0.3, 0.7, k).tolist()

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
    title="Regret by Policy",
)
