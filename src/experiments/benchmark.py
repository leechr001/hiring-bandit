import numpy as np

from simulation import ExperimentSeries, plot_regret_series


series = [
    ExperimentSeries(
        policy_name="optimistic-hire-auto",
        label="Optimistic-Hire",
    ),
    ExperimentSeries(policy_name="AHT", label="AgrawalHegdeTeneketzis"),
    ExperimentSeries(policy_name="OMM", label="OMM"),
    ExperimentSeries(policy_name="threshold-0.9", label="Threshold (0.9)"),
    ExperimentSeries(policy_name="threshold-0.75", label="Threshold (0.75)"),
    ExperimentSeries(policy_name="threshold-0.5", label="Threshold (0.5)"),
]

k = 200
m = 20
T = 10000
c = 2
omega_max = 5
n_runs = 20
rng = np.random.default_rng(12345)
means = rng.uniform(0.0, 1.0, size=k).tolist()

plot_regret_series(
    series=series,
    simulate_kwargs={
        "k": k,
        "m": m,
        "T": T,
        "means": means,
        "c": c,
        "omega_max": omega_max,
        "epsilon": 0,
        "n_runs": n_runs,
    },
    title=rf"Performance of Optimistic-Hire Compared to Benchmarks",
)
