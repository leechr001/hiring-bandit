from __future__ import annotations

import numpy as np

from simulation import ExperimentSeries
from samplers import make_bernoulli_samplers


BASE_SEED = 12345

CONFIG_NAME = "main_20_25"

K = 25
M = 20
T = HORIZON = 365
SWITCHING_COST = 3

OMEGA_MEAN = 3

N_RUNS = 250
N_JOBS = 1
WORK_TRIAL_ROTATION_PERIODS = 90

rng = np.random.default_rng(BASE_SEED)
PERFORMANCE_MEANS = np.clip(
    rng.normal(0.5, 0.3, size=K), 
    0,1).tolist()

PERFORMANCE_SAMPLERS = make_bernoulli_samplers(
    means=PERFORMANCE_MEANS,
    rng=rng)

INTERVIEW_RHO = 0.3
INTERVIEW_COST = SWITCHING_COST

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


def interview_screen_series() -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="pre-screen",
        label="InterviewScreen",
        sim_kwargs={"rho": INTERVIEW_RHO, "cost": INTERVIEW_COST * K},
    )


def work_trial_series() -> ExperimentSeries:
    return ExperimentSeries(
        policy_name="WorkTrial",
        label="WorkTrial",
        sim_kwargs={"rho": INTERVIEW_RHO, "cost": INTERVIEW_COST * K},
    )


def bandit_series(*, omega_mean: float = OMEGA_MEAN) -> list[ExperimentSeries]:
    return [
        delayed_replace_ucb_series(omega_mean=omega_mean),
        adapted_aht_series(),
        adapted_omm_series(),
    ]


def heuristic_series(*, omega_mean: float = OMEGA_MEAN) -> list[ExperimentSeries]:
    return [
        delayed_replace_ucb_series(omega_mean=omega_mean),
        interview_screen_series(),
        work_trial_series(),
    ]


def benchmark_series(*, omega_mean: float = OMEGA_MEAN) -> list[ExperimentSeries]:
    return bandit_series(omega_mean=omega_mean) + [
        interview_screen_series(),
        work_trial_series(),
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
        "work_trial_rotation_periods": WORK_TRIAL_ROTATION_PERIODS,
        **delay_kwargs_for_omega_mean(omega_mean=omega_mean),
    }
    kwargs.update(overrides)
    return kwargs


DELAYED_REPLACE_UCB = delayed_replace_ucb_series()
A_AHT = adapted_aht_series()
A_OMM = adapted_omm_series()
INTERVIEW_SCREEN = interview_screen_series()
WORK_TRIAL = work_trial_series()

BANDIT_SERIES = bandit_series()
HEURISTIC_SERIES = heuristic_series()
BENCHMARK_SERIES = benchmark_series()
ALL_SERIES = BENCHMARK_SERIES
SIMULATE_KWARGS = benchmark_simulate_kwargs()
