import numpy as np

from simulation import run_c_sweep

# Problem setup
k = 10
m = 3
T = 20000

means = np.linspace(0.3, 0.8, k).tolist()

omega_max = 5
c_values = [1, 3, 5, 10, 15]

run_c_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='optimistic-hire-gamma-1',
    means=means,
    c_values=c_values,
    omega_max=omega_max,
    n_runs=20,
)

run_c_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='optimistic-hire-gamma-2',
    means=means,
    c_values=c_values,
    omega_max=omega_max,
    n_runs=20,
)
