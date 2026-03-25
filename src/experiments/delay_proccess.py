import random

from simulation import ExperimentSeries, make_delay_sampler_factory, plot_regret_series


series = [
    ExperimentSeries(policy_name="omm-rm", label="OMM with rank-matching bijection"),
    ExperimentSeries(policy_name="omm", label="OMM with random bijection"),
    ExperimentSeries(policy_name="omm-rmm", label="OMM with rank-mismatching bijection"),
]

k = 20
m = 10
T = 1000
c = 0
omega_max = 50
n_runs = 20

rng = random.Random(123)
means = sorted([rng.uniform(0.1, 0.9) for _ in range(k)], reverse=True)

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
            "delay_sampler_factory": make_delay_sampler_factory(
                delay_process_name,
                means=means,
                omega_max=omega_max,
            ),
            "n_runs": n_runs,
        },
        title=f"Comparison of regret for OMM with different bijections and {delay_process_name} delays.",
    )
