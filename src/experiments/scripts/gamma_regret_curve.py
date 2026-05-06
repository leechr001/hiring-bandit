import numpy as np

from simulation import (
    HighlightPoint,
    compute_delayed_replace_ucb_auto_gamma,
    plot_final_regret_sweep,
)

from experiments.simulation_setups.config_gamma_tune import (
    SIMULATE_KWARGS,
    K,
    M,
    HORIZON,
    SWITCHING_COST,
    OMEGA_MEAN
)

gamma_1 = (OMEGA_MEAN + SWITCHING_COST)**2 * M
gamma_2 = (OMEGA_MEAN + SWITCHING_COST) * M
gamma_auto = compute_delayed_replace_ucb_auto_gamma(
    k=K,
    m=M,
    T=HORIZON,
    c=SWITCHING_COST,
    omega_mean=OMEGA_MEAN,
)

gamma_max = max([gamma_1, gamma_2, gamma_auto])*2
max_power = np.log2(gamma_max)
base_gammas = [2**p for p in np.linspace(0, max_power, 25)]
gammas = sorted(
    set(
        base_gammas
        + [gamma_1, gamma_2, gamma_auto]
    )
)

plot_final_regret_sweep(
    policy_name="delayed-replace-ucb",
    parameter_name="gamma",
    values=gammas,
    simulate_kwargs=SIMULATE_KWARGS,
    title=(
        r"Regret($T$) vs $\gamma$"
    ),
    xlabel=r"$\gamma$",
    xscale="log",
    ylabel=rf"Total cumulative regret at $T = {HORIZON}$",
    figure_kwargs={"figsize": (7, 5)},
    highlight_points=[
        HighlightPoint(
            value=gamma_1,
            label=r"$\gamma_1 = (\bar\omega + c)^2 m$",
            plot_kwargs={"marker": "s"},
        ),
        HighlightPoint(
            value=gamma_2,
            label=r"$\gamma_2 = (\bar\omega + c) m$",
            plot_kwargs={"marker": "D"},
        ),
        HighlightPoint(
            value=gamma_auto,
            label=rf"$\gamma^* \approx {gamma_auto:.2f}$",
            plot_kwargs={"marker": "*", "s": 180, "color": "crimson"},
        ),
    ],
)
