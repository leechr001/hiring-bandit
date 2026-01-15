from simulation import run_policy_comparisons


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
T = 1000
c = 10
omega_max = 3

run_policy_comparisons(
    policies=policies,
    labels=labels,
    k=k,
    m=m,
    T=T,
    c=c,
    omega_max=omega_max,
    n_runs=25,
)
