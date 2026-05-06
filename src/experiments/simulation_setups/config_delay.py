from __future__ import annotations

import numpy as np

from simulation import ExperimentSeries
from samplers import make_bernoulli_samplers


BASE_SEED = 12345

K = 25
M = 10
T = HORIZON = 365
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



def delayed_replace_ucb_series(
    *,
    omega_mean: float = OMEGA_MEAN,
    label: str = "DR-UCB",
) -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="delayed-replace-ucb",
        label=label,
        sim_kwargs={"gamma": f"auto={float(omega_mean)}"}
    )


def adapted_aht_series(*, label: str = "Adapted-AHT") -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="a-aht",
        label=label,
    )


def adapted_omm_series(*, label: str = "Adapted-OMM") -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="a-omm",
        label=label,
    )


def bandit_series(*, omega_mean: float = OMEGA_MEAN) -> list[ExperimentSeries]:
    return [
        delayed_replace_ucb_series(omega_mean=omega_mean),
        adapted_aht_series(),
        adapted_omm_series(),
    ]


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
        "delay_upper": int(omega_mean),
        "delay_lower": int(omega_mean),
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
