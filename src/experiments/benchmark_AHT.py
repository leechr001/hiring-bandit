from simulation import run_policy_comparisons

import random
import numpy as np

policies = [
    "hiring-ucb-gamma-1", 
    "hiring-ucb-gamma-2",
    "AHT"
]

labels = [
    r"Hiring-UCB, $\gamma=(c+\omega_\max)^2 m$",
    r"Hiring-UCB, $\gamma=(c+\omega_\max) m$",
    "AgrawalHegdeTeneketzis"
]

k = 10
m = 3
T = 20000
c = 5
omega_max = 5

rng = random.Random(123)
means = np.linspace(0.3, 0.7, k).tolist()

run_policy_comparisons(
    policies=policies,
    labels=labels,
    means=means,
    k=k,
    m=m,
    T=T,
    c=c,
    omega_max=omega_max,
    n_runs=25,
)
