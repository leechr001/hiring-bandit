import random

from simulation import ExperimentSeries, plot_regret_series


series = [
    ExperimentSeries(policy_name="omm-rm", label="OMM with rank-matching bijection"),
    ExperimentSeries(policy_name="omm", label="OMM with random bijection"),
    ExperimentSeries(policy_name="omm-rmm", label="OMM with rank-mismatching bijection"),
]

k = 100
m = 75
T = 10000
c = 0
omega_max = 500
n_runs = 5
n_jobs = 4

rng = random.Random(12345)
means = [rng.uniform(0.3, 0.7) for _ in range(k)]

for delay_process_name in ["stochastic", "adversarial"]:
    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "c": c,
            "omega_max": omega_max,
            "delay_process_name": delay_process_name,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
        },
        title=f"Regret of OMM by bijection with {delay_process_name} delays.",
    )
