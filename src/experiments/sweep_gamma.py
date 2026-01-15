import numpy as np

from simulation import run_gamma_sweep

# Problem setup
k = 10
m = 3
T = 1000
# Example true means (you can swap this out for whatever instance you like)
# Sorted so that workers 1..m are optimal, etc.

means = np.linspace(0.1, 0.9, k).tolist()

c = 3
omega_max = 3
gammas = [1,5, 10, 15, 30, 100]

run_gamma_sweep(
    k=k,
    m=m,
    T=T,
    means=means,
    gammas=gammas,
    c=c,
    omega_max=omega_max,
    n_runs=20,
)

