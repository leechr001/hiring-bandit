from pathlib import Path

import numpy as np

from simulation import run_c_sweep


# Problem setup
k = 150
m = 100
T = 5 * 365 * 24
omega_max = 1
delay_lower = 1
n_runs = 20
n_jobs = min(n_runs, 4)

base_seed = 12345
rng = np.random.default_rng(base_seed)
means = rng.uniform(0.3, 0.7, size=k).tolist()

c_values = [0, 8, 24, 7*24]

policy_specs = [
    ("optimistic-hire-auto", "OH"),
    ("AHT", "AHT"),
    ("OMM", "OMM"),
]

output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "sweep_c"
output_dir.mkdir(parents=True, exist_ok=True)

for policy_name, file_stem in policy_specs:
    output_path = output_dir / f"{file_stem}_c_sweep.png"
    output_path.unlink(missing_ok=True)
    run_c_sweep(
        k=k,
        m=m,
        T=T,
        policy_name=policy_name,
        means=means,
        c_values=c_values,
        omega_max=omega_max,
        delay_process_name="uniform",
        delay_lower=delay_lower,
        n_runs=n_runs,
        n_jobs=n_jobs,
        base_seed=base_seed,
        y_up_lim=100000,
        save_path=str(output_path),
        show_plot=False,
    )
