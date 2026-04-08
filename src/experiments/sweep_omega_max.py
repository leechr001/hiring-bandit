import numpy as np

from simulation import run_omega_sweep

# Problem setup
k = 150
m = 100
T = 5 * 365 * 24
c = 8
n_runs = 4
n_jobs = min(n_runs, 4)

# set common upper limit on y axis for compairson
y_up_lim = 130000

rng = np.random.default_rng(12345)
means = rng.uniform(0.3, 0.7, size=k).tolist()

omega_values = [1, 8, 24, 7*24, 30*24, 3*30*24, 6*30*24]
policies = ["AHT", "OMM", "Optimistic-Hire"]
# In the stochastic sweep, each value is treated as a target mean delay.
# The simulator converts it into a uniform integer interval centered at that mean
# with radius 1.2 * mean, clipped to start at 1.
omega_sweeps = [
    ("stochastic", "mean"),
    ("adversarial", "max"),
]

for omega_process, omega_value_type in omega_sweeps:
    for policy_name in policies:
        run_omega_sweep(
            k=k,
            m=m,
            T=T,
            policy_name=policy_name,
            means=means,
            c=c,
            omega_values=omega_values,
            omega_process=omega_process,
            omega_value_type=omega_value_type,
            n_runs=n_runs,
            n_jobs=n_jobs,
            y_up_lim=y_up_lim,
        )
