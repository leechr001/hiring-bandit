import numpy as np

from simulation import (
    HighlightPoint,
    compute_optimistic_hire_auto_gamma,
    plot_final_regret_sweep,
)


k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
n_runs = 5
n_jobs = 4

base_seed = 12345

rng = np.random.default_rng(base_seed)
means = rng.uniform(0.3, 0.7, size=k).tolist()

gamma_1 = (c + omega_max) ** 2 * m
gamma_2 = (c + omega_max) * m

gamma_auto = compute_optimistic_hire_auto_gamma(
    k=k,
    m=m,
    T=T,
    c=c,
    omega_max=omega_max,
)

gamma_min = 1.0
gamma_max = gamma_1 * 1.5
base_gammas = np.linspace(gamma_min, gamma_max, num=25).tolist()
gammas = sorted(
    set(
        [1, 2, 3, 5, 10, 15]
        + base_gammas
        + [gamma_1, gamma_2, gamma_auto]
    )
)

plot_final_regret_sweep(
    policy_name="optimistic-hire",
    parameter_name="gamma",
    values=gammas,
    simulate_kwargs={
        "k": k,
        "m": m,
        "T": T,
        "means": means,
        "c": c,
        "omega_max": omega_max,
        "n_runs": n_runs,
        "n_jobs": n_jobs,
        "seed0": base_seed,
    },
    title=(
        r"Regret($T$) vs $\gamma$"
    ),
    xlabel=r"$\gamma$",
    ylabel=rf"Total cumulative regret at $T = {T}$",
    figure_kwargs={"figsize": (7, 5)},
    highlight_points=[
        HighlightPoint(
            value=gamma_1,
            label=r"$\gamma_1 = (\omega_{\max} + c)^2 m$",
            plot_kwargs={"marker": "s"},
        ),
        HighlightPoint(
            value=gamma_2,
            label=r"$\gamma_2 = (\omega_{\max} + c) m$",
            plot_kwargs={"marker": "D"},
        ),
        HighlightPoint(
            value=gamma_auto,
            label=rf"$\gamma^* \approx {gamma_auto:.2f}$",
            plot_kwargs={"marker": "*", "s": 180, "color": "crimson"},
        ),
    ],
)
