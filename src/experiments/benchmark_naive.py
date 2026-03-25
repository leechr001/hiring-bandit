import numpy as np

from simulation import ExperimentSeries, plot_regret_series


series = [
    # ExperimentSeries(
    #     policy_name="optimistic-hire-gamma-1",
    #     label=r"Optimistic-Hire, $\gamma=(c+\omega_\max)^2 m$",
    # ),
    # ExperimentSeries(
    #     policy_name="optimistic-hire-gamma-2",
    #     label=r"Optimistic-Hire, $\gamma=(c+\omega_\max) m$",
    # ),
    ExperimentSeries(
        policy_name="optimistic-hire-auto",
        label=r"Optimistic-Hire, $\gamma=$Numerical Minimum",
    ),
    ExperimentSeries(policy_name="OMM", label="OMM"),
    ExperimentSeries(policy_name="Epsilon-Greedy", label=r"$\epsilon$-Greedy"),
]

k = 10
m = 3
T = 20000
c = 50
omega_max = 50
n_runs = 20
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
    title=rf"Performance of Optimistic-Hire for two theoretical choices of $\gamma$.",
)
