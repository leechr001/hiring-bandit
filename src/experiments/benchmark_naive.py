from simulation import run_policy_comparisons

import random
import numpy as np

policies = [
    "optimistic-hire-gamma-1", 
    "optimistic-hire-gamma-2",
    "UCB", 
    "Epsilon-Greedy",
]

labels = [
    r"Optimistic-Hire, $\gamma=(c+\omega_\max)^2 m$",
    r"Optimistic-Hire, $\gamma=(c+\omega_\max) m$",
    "UCB",
    r"$\epsilon$-Greedy",
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
    n_runs=20,
    title=rf'Performance of Optimistic-Hire for two theoretical choices of $\gamma$.'
)
