from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bandit_environment import TemporaryHiringBanditEnv
from experiments.helpers import benchmark_output_dir
from experiments.simulation_setups.config_main import (
    BANDIT_SERIES,
    HORIZON,
    benchmark_simulate_kwargs,
)
from simulation import (
    ExperimentSeries,
    _merge_sim_kwargs,
    _resolve_policy_omega_mean,
    make_delay_sampler_factory,
    make_policy,
    make_reward_sampler_factory,
)


c_values = [0, 3, 7, 30, 60, 90]

series = BANDIT_SERIES
output_dir = benchmark_output_dir(module_file=__file__, output_subdir="sweep_c_replacements")
output_path = output_dir / "completed_replacements_vs_c.png"


def _episode_reward_samplers(simulate_kwargs: Mapping[str, Any], seed: int) -> Sequence:
    if simulate_kwargs.get("reward_samplers") is not None:
        return simulate_kwargs["reward_samplers"]

    reward_sampler_factory = simulate_kwargs.get("reward_sampler_factory")
    if reward_sampler_factory is not None:
        return reward_sampler_factory(seed)

    return make_reward_sampler_factory(
        str(simulate_kwargs.get("reward_process_name", "bernoulli")),
        means=simulate_kwargs["means"],
        reward_stddev=float(simulate_kwargs.get("reward_stddev", 0.1)),
        reward_lower=float(simulate_kwargs.get("reward_lower", 0.0)),
        reward_upper=float(simulate_kwargs.get("reward_upper", 1.0)),
    )(seed)


def _episode_delay_sampler(simulate_kwargs: Mapping[str, Any], seed: int):
    if simulate_kwargs.get("delay_sampler") is not None:
        return simulate_kwargs["delay_sampler"]

    delay_sampler_factory = simulate_kwargs.get("delay_sampler_factory")
    if delay_sampler_factory is not None:
        return delay_sampler_factory(seed)

    delay_process_name = str(simulate_kwargs.get("delay_process_name", "uniform"))
    delay_lower = int(simulate_kwargs.get("delay_lower", 0))
    delay_upper = simulate_kwargs.get("delay_upper")
    delay_geom_p = simulate_kwargs.get("delay_geom_p")
    omega_mean = _resolve_policy_omega_mean(
        omega_mean=simulate_kwargs.get("omega_mean"),
        delay_process_name=delay_process_name,
        delay_lower=delay_lower,
        delay_upper=delay_upper,
        delay_geom_p=delay_geom_p,
    )

    return make_delay_sampler_factory(
        delay_process_name,
        means=simulate_kwargs["means"],
        delay_upper=delay_upper,
        delay_lower=delay_lower,
        delay_geom_p=delay_geom_p,
        omega_mean=omega_mean,
        calendar_frequency=simulate_kwargs.get("calendar_frequency"),
        calendar_distribution=str(simulate_kwargs.get("calendar_distribution", "geom")),
        calendar_geom_p=float(simulate_kwargs.get("calendar_geom_p", 0.5)),
    )(seed)


def _count_completed_replacements(
    *,
    policy_name: str,
    simulate_kwargs: Mapping[str, Any],
    seed: int,
) -> int:
    k = int(simulate_kwargs["k"])
    m = int(simulate_kwargs["m"])
    T = int(simulate_kwargs["T"])
    means = list(simulate_kwargs["means"])
    c = float(simulate_kwargs.get("c", 0.0))
    omega_mean = _resolve_policy_omega_mean(
        omega_mean=simulate_kwargs.get("omega_mean"),
        delay_process_name=str(simulate_kwargs.get("delay_process_name", "uniform")),
        delay_lower=int(simulate_kwargs.get("delay_lower", 0)),
        delay_upper=simulate_kwargs.get("delay_upper"),
        delay_geom_p=simulate_kwargs.get("delay_geom_p"),
    )

    rng = random.Random(seed)
    initial_workforce = rng.sample(list(range(1, k + 1)), m)
    env = TemporaryHiringBanditEnv(
        k=k,
        m=m,
        reward_samplers=_episode_reward_samplers(simulate_kwargs, seed),
        delay_sampler=_episode_delay_sampler(simulate_kwargs, seed),
        c=c,
        omega_mean=omega_mean,
        rng=rng,
        true_means=means,
        initial_workforce=initial_workforce,
    )
    policy = make_policy(
        policy_name,
        k=k,
        m=m,
        T=T,
        c=c,
        rng=rng,
        omega_mean=omega_mean,
        true_means=means,
        epsilon=float(simulate_kwargs.get("epsilon", 0.1)),
        review_interval=int(simulate_kwargs.get("review_interval", 6 * 30 * 24)),
        work_trial_rotation_periods=int(
            simulate_kwargs.get("work_trial_rotation_periods", 90 * 24)
        ),
        threshold=float(simulate_kwargs.get("threshold", 0.5)),
        pre_screen_rho=float(simulate_kwargs.get("pre_screen_rho", 1.0)),
        pre_screen_cost=float(simulate_kwargs.get("pre_screen_cost", 0.0)),
        rho=simulate_kwargs.get("rho"),
        cost=simulate_kwargs.get("cost"),
        ucb_coef=float(simulate_kwargs.get("ucb_coef", 2.0)),
        gamma=simulate_kwargs.get("gamma", "auto"),
    )

    completed_replacements = 0
    for _ in range(T):
        replacements = policy.act(env)
        _, _, _, feedback = env.step(replacements)
        policy.update(feedback)
        completed_replacements += len(feedback.completed_this_period)

    return completed_replacements


def _run_policy_sweep(
    spec: ExperimentSeries,
    *,
    base_simulate_kwargs: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []

    n_runs = int(base_simulate_kwargs["n_runs"])
    seed0 = int(base_simulate_kwargs["seed0"])
    for c in c_values:
        simulate_kwargs = _merge_sim_kwargs(
            benchmark_simulate_kwargs(c=float(c)),
            spec.sim_kwargs,
        )
        completed_counts = [
            _count_completed_replacements(
                policy_name=spec.policy_name,
                simulate_kwargs=simulate_kwargs,
                seed=seed0 + run_idx,
            )
            for run_idx in range(n_runs)
        ]
        completed = np.asarray(completed_counts, dtype=np.float64)
        means.append(float(completed.mean()))
        stds.append(float(completed.std()))

    return means, stds


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_simulate_kwargs = benchmark_simulate_kwargs()
    completed_means: dict[str, list[float]] = {}
    completed_stds: dict[str, list[float]] = {}

    for spec in series:
        label = spec.label or spec.policy_name
        completed_means[label], completed_stds[label] = _run_policy_sweep(
            spec,
            base_simulate_kwargs=base_simulate_kwargs,
        )

    plt.figure(figsize=(7, 5))
    for spec in series:
        label = spec.label or spec.policy_name
        plt.errorbar(
            c_values,
            completed_means[label],
            yerr=completed_stds[label],
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    plt.xlabel("Switching cost c")
    plt.ylabel(f"Replacements by T = {HORIZON}")
    plt.ylim((0,400))
    plt.title("Replacements vs Switching Cost")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_path.unlink(missing_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved completed-replacements sweep to {output_path}")


if __name__ == "__main__":
    main()
