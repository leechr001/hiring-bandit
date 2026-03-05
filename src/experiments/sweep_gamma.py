import numpy as np

from simulation import run_gamma_sweep

# Problem setup
k = 10
m = 3
T = 20000

means = np.linspace(0.3, 0.7, k).tolist()

c = 10
omega_max = 10
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

