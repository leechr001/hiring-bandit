from pathlib import Path

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

omega_max_values = [1, 8, 24, 7*24]
policies = [
    "AHT", "OMM", 
    "Optimistic-Hire"
    ]
omega_sweeps = ["stochastic", "adversarial",]

plot_file_names = [
    "AHT_delay_sweep_stoch",
    "OMM_delay_sweep_stoch",
    "OH_delay_sweep_stoch",
    "AHT_delay_sweep_wc",
    "OMM_delay_sweep_wc",
    "OH_delay_sweep_wc",
]

output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "sweep_omega_max"
output_dir.mkdir(parents=True, exist_ok=True)

for file_name, (omega_process, policy_name) in zip(
    plot_file_names,
    [(omega_process, policy_name) for omega_process in omega_sweeps for policy_name in policies],
):
        output_path = output_dir / f"{file_name}.png"
        output_path.unlink(missing_ok=True)
        run_omega_sweep(
            k=k,
            m=m,
            T=T,
            policy_name=policy_name,
            means=means,
            c=c,
            omega_values=omega_max_values,
            omega_process=omega_process,
            n_runs=n_runs,
            n_jobs=n_jobs,
            y_up_lim=y_up_lim,
            save_path=str(output_path),
            show_plot=False,
        )
