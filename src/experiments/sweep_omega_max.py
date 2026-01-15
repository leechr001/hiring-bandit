import numpy as np

from simulation import run_omega_sweep

# Problem setup
k = 30
m = 5
T = 1000
# Example true means (you can swap this out for whatever instance you like)
# Sorted so that workers 1..m are optimal, etc.

means = np.linspace(0.1, 0.9, k).tolist()

c = 0
omega_max_values = [1, 5, 10, 30]

run_omega_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='ucb',
    means=means,
    c=c,
    omega_max_values=omega_max_values,
    omega_process = "stochastic",
    n_runs=20,
    y_up_lim=1000
)

run_omega_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='ucb',
    means=means,
    c=c,
    omega_max_values=omega_max_values,
    omega_process = "adversarial",
    n_runs=20,
    y_up_lim=1000
)

