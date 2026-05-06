import numpy as np

from simulation import ExperimentSeries, plot_regret_series


k = 10
m = 3
T = 20000
means = np.linspace(0.3, 0.8, k).tolist()

delay_upper = 5
c_values = [1, 3, 5, 10, 15]
n_runs = 20
n_jobs = 4

gamma_specs = [
    ("gamma_1", lambda c: (c + delay_upper) ** 2 * m),
    ("gamma_2", lambda c: (c + delay_upper) * m),
]

for gamma_label, gamma_fn in gamma_specs:
    series = [
        ExperimentSeries(
            policy_name="delayed-replace-ucb",
            label=rf"$c={float(c):.2f}$",
            sim_kwargs={"c": float(c), "gamma": float(gamma_fn(float(c)))},
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
            "delay_upper": delay_upper,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
        },
        title=rf"Average regret over {n_runs} runs with delay upper = {delay_upper} ({gamma_label})",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, 3500),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
    )
