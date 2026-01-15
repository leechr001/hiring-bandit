import numpy as np

from simulation import run_c_sweep

# Problem setup
k = 10
m = 3
T = 1000

means = np.linspace(0.1, 0.9, k).tolist()

omega_max = 3
c_values = [1,5, 10, 15, 30]

run_c_sweep(
    k=k,
    m=m,
    T=T,
    policy_name='hiring-ucb-gamma-2',
    means=means,
    c_values=c_values,
    omega_max=omega_max,
    n_runs=20,
)

