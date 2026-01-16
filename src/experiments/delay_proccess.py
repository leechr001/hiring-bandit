from simulation import run_policy_comparisons

import random
import numpy as np

policies = [
    "ucb-rm",
    "ucb",
    "ucb-rmm"
]

labels = [
    "UCB with rank-matching bijection",
    "UCB with random bijection",
    "UCB with rank-mismatching bijection"
]

k = 20
m = 10
T = 1000
c = 0
omega_max = 50

rng = random.Random(123)
means = sorted([rng.uniform(0.1, 0.9) for _ in range(k)], reverse=True)
#means = np.linspace(0.1, 0.9, k).tolist()

delay_process = ["stochastic", "adversarial"]

for dp in delay_process:
    run_policy_comparisons(
        policies=policies,
        labels=labels,
        means=means,
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=omega_max,
        delay_process_name=dp,
        n_runs=20,
        title=f"Comparison of regret for UCB with different bijections and {dp} delays."
    )
