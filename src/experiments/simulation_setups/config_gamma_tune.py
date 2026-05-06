from __future__ import annotations

import numpy as np
from samplers import make_bernoulli_samplers


BASE_SEED = 1

K = 25
M = 10
T = HORIZON = 365*3
SWITCHING_COST = 3

OMEGA_MEAN = 3

N_RUNS = 100
N_JOBS = 1

rng = np.random.default_rng(BASE_SEED)
PERFORMANCE_MEANS = np.clip(
    rng.normal(0.5, 0.3, size=K), 
    0,1).tolist()

PERFORMANCE_SAMPLERS = make_bernoulli_samplers(
    means=PERFORMANCE_MEANS,
    rng=rng)



def delay_kwargs_for_omega_mean(
    *,
    omega_mean: float,
) -> dict[str, float | int | str]:
    if omega_mean < 0:
        raise ValueError("omega_mean must be non-negative.")

    if omega_mean > 0:
        return {
            "delay_process_name": "geometric",
            "delay_geom_p": 1/omega_mean,
        }
    return {
        "delay_process_name": "uniform",
        "delay_upper": 0,
        "delay_lower": 0,
    }


def benchmark_simulate_kwargs(
    *,
    omega_mean: float = OMEGA_MEAN,
    **overrides,
) -> dict:
    kwargs = {
        "k": K,
        "m": M,
        "T": HORIZON,
        "means": PERFORMANCE_MEANS,
        "reward_samplers": PERFORMANCE_SAMPLERS,
        "c": SWITCHING_COST,
        "n_runs": N_RUNS,
        "n_jobs": N_JOBS,
        "seed0": BASE_SEED,
        **delay_kwargs_for_omega_mean(omega_mean=omega_mean),
    }
    kwargs.update(overrides)
    return kwargs


SIMULATE_KWARGS = benchmark_simulate_kwargs()
