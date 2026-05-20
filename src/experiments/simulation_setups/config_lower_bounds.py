from __future__ import annotations

import numpy as np

from simulation import ExperimentSeries
from samplers import make_bernoulli_samplers


BASE_SEED = 12345

CONFIG_NAME = "main_10_25"

K = 25
M = 10
T = HORIZON = 365
SWITCHING_COST = 5

OMEGA_MEAN = 3

N_RUNS = 250
N_JOBS = 1

rng = np.random.default_rng(BASE_SEED)
PERFORMANCE_MEANS = [1/2 for _ in range(M)] + [1/4 for _ in range(K-M)]


PERFORMANCE_SAMPLERS = make_bernoulli_samplers(
    means=PERFORMANCE_MEANS,
    rng=rng)


DEFAULT_CUMULATIVE_TITLE = "Regret of DR-UCB Compared to Benchmarks"
DEFAULT_AVERAGE_TITLE = "Normalized Loss of DR-UCB Compared to Benchmarks"
DEFAULT_PLANNING_HORIZONS = (
    ("1 month", int(365 / 12)),
    ("3 months", int(365 / 4)),
    ("6 months", int(365 / 2)),
    ("12 months", 365),
)


def delayed_replace_ucb_series(
    *,
    omega_mean: float = OMEGA_MEAN,
    label: str = "DR-UCB",
) -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="delayed-replace-ucb",
        label=label,
        sim_kwargs={"gamma": f"auto={float(omega_mean)}"},
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

def lower_bound_instance_ind() -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="instance-independent-lower-bound",
        label="Instance Independent Lower Bound"
    )

def lower_bound_instance_dep() -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="instance-dependent-lower-bound",
        label="Instance Dependent Lower Bound"
    )

def ck() -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="startup-cost",
        label="Startup Cost"
    )


def lower_bound_series(*, omega_mean: float = OMEGA_MEAN) -> list[ExperimentSeries]:
    return [
        delayed_replace_ucb_series(omega_mean=omega_mean),
        adapted_aht_series(),
        adapted_omm_series(),
        lower_bound_instance_dep(),
        #ck()
    ]



def delay_kwargs_for_omega_mean(
    *,
    omega_mean: float,
) -> dict[str, float | int | str]:
    if omega_mean < 0:
        raise ValueError("omega_mean must be non-negative.")

    if omega_mean == 0:
        return {
            "delay_process_name": "uniform",
            "delay_upper": 0,
            "delay_lower": 0,
        }

    return {
        "delay_process_name": "geometric",
        "omega_mean": float(omega_mean),
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


DELAYED_REPLACE_UCB = delayed_replace_ucb_series()
A_AHT = adapted_aht_series()
A_OMM = adapted_omm_series()

LOWER_BOUND_SERIES = lower_bound_series()
SIMULATE_KWARGS = benchmark_simulate_kwargs()
