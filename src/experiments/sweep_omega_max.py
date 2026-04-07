import numpy as np

from simulation import run_omega_sweep

# Problem setup
k = 150
m = 100
T = 5 * 365 * 24
c = 8
n_runs = 4
n_jobs = 4

# set common upper limit on y axis for compairson
y_up_lim = 130000

rng = np.random.default_rng(12345)
means = rng.uniform(0.3, 0.7, size=k).tolist()

omega_max_values = [1, 8, 24, 7*24, 30*24, 3*30*24, 6*30*24]

for delay_process in ("adversarial", "stochastic"):

    run_omega_sweep(
        k=k,
        m=m,
        T=T,
        policy_name='AHT',
        means=means,
        c=c,
        omega_max_values=omega_max_values,
        omega_process=delay_process,
        n_runs=n_runs,
        n_jobs=n_jobs,
        y_up_lim=y_up_lim
    )

    run_omega_sweep(
        k=k,
        m=m,
        T=T,
        policy_name='OMM',
        means=means,
        c=c,
        omega_max_values=omega_max_values,
        omega_process=delay_process,
        n_runs=n_runs,
        n_jobs=n_jobs,
        y_up_lim=y_up_lim
    )

    run_omega_sweep(
        k=k,
        m=m,
        T=T,
        policy_name='Optimistic-Hire',
        means=means,
        c=c,
        omega_max_values=omega_max_values,
        omega_process=delay_process,
        n_runs=n_runs,
        n_jobs=n_jobs,
        y_up_lim=y_up_lim
    )





