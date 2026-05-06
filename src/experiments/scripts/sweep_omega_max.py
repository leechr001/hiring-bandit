import numpy as np

from experiments.helpers import benchmark_output_dir
from simulation import run_omega_sweep

# Problem setup
k = 150
m = 100
T = 5 * 365 * 24
c = 8
n_runs = 20
n_jobs = min(n_runs, 4)

# set common upper limit on y axis for compairson
y_up_lim = 90000

rng = np.random.default_rng(12345)
means = rng.uniform(0.3, 0.7, size=k).tolist()

delay_upper_values = [0, 8, 24, 7*24]
policies = [
    "a-aht",
    "a-omm",
    "delayed-replace-ucb",
    ]
omega_sweeps = ["stochastic", "adversarial",]

plot_file_names = [
    "A-AHT_delay_sweep_stoch",
    "A-OMM_delay_sweep_stoch",
    "DR-UCB_delay_sweep_stoch",
    "A-AHT_delay_sweep_wc",
    "A-OMM_delay_sweep_wc",
    "DR-UCB_delay_sweep_wc",
]

output_dir = benchmark_output_dir(module_file=__file__, output_subdir="sweep_delay_upper")
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
            omega_values=delay_upper_values,
            omega_process=omega_process,
            n_runs=n_runs,
            n_jobs=n_jobs,
            y_up_lim=y_up_lim,
            save_path=str(output_path),
            show_plot=False,
        )
