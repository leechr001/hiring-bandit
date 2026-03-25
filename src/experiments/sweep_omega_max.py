import numpy as np

from simulation import run_omega_sweep

# Problem setup
k = 50
m = 5
T = 1000

# set common upper limit on y axis for compairson
y_up_lim = 1500

means = np.linspace(0.1, 0.9, k).tolist()

c = 0
omega_max_values = [1, 5, 10, 20, 50]

run_omega_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='omm',
    means=means,
    c=c,
    omega_max_values=omega_max_values,
    omega_process = "stochastic",
    n_runs=20,
    y_up_lim=y_up_lim
)

run_omega_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='omm',
    means=means,
    c=c,
    omega_max_values=omega_max_values,
    omega_process = "adversarial",
    n_runs=20,
    y_up_lim=y_up_lim
)
