import numpy as np

from simulation import (
    HighlightPoint,
    compute_optimistic_hire_auto_gamma,
    plot_final_regret_sweep,
)


k = 10
m = 3
T = 20000
means = np.linspace(0.3, 0.7, k).tolist()

c = 5
omega_max = 5
n_runs = 30
base_seed = 12345

gamma_theoretical_1 = (c + omega_max) ** 2 * m
gamma_theoretical_2 = (c + omega_max) * m
gamma_auto = compute_optimistic_hire_auto_gamma(
    k=k,
    m=m,
    T=T,
    c=c,
    omega_max=omega_max,
)

gamma_min = 1.0
gamma_max = 1000.0
base_gammas = np.linspace(gamma_min, gamma_max, num=25).tolist()
gammas = sorted(
    set(
        [1, 2, 3, 5, 10, 15]
        + base_gammas
        + [gamma_theoretical_1, gamma_theoretical_2, gamma_auto]
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
        "seed0": base_seed,
    },
    title=(
        r"Final cumulative regret vs $\gamma$"
        + "\n"
        + rf"($k={k}$, $m={m}$, $c={c}$, $\omega_\max={omega_max}$, ${n_runs}$ runs per $\gamma$)"
    ),
    xlabel=r"$\gamma$",
    ylabel=rf"Total cumulative regret at $T = {T}$",
    figure_kwargs={"figsize": (7, 5)},
    highlight_points=[
        HighlightPoint(
            value=gamma_theoretical_1,
            label=r"Theoretical Choice: $\gamma = (\omega_{\max} + c)^2 m$",
            plot_kwargs={"marker": "s"},
        ),
        HighlightPoint(
            value=gamma_theoretical_2,
            label=r"Theoretical Choice: $\gamma = (\omega_{\max} + c) m$",
            plot_kwargs={"marker": "D"},
        ),
        HighlightPoint(
            value=gamma_auto,
            label=rf"Numerical Minimum: $\gamma \approx {gamma_auto:.2f}$",
            plot_kwargs={"marker": "*", "s": 180, "color": "crimson"},
        ),
    ],
)
