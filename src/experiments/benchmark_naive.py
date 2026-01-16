from simulation import run_policy_comparisons

import random
import numpy as np

policies = [
    "hiring-ucb-gamma-1", 
    "hiring-ucb-gamma-2",
    "UCB", 
    "Epsilon-Greedy",
]

labels = [
    r"Hiring-UCB, $\gamma=(c+\omega_\max)^2 m$",
    r"Hiring-UCB, $\gamma=(c+\omega_\max) m$",
    "UCB",
    r"$\epsilon$-Greedy",
]

k = 10
m = 3
T = 20000
c = 10
omega_max = 50

rng = random.Random(123)
means = sorted([rng.uniform(0.1, 0.9) for _ in range(k)], reverse=True)
#means = np.linspace(0.1, 0.9, k).tolist()

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
    title=rf'Performance of Hiring-UCB for two theoretical choices of $\gamma$.'
)
