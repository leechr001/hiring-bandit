from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from experiments.benchmark_common import benchmark_output_dir, make_benchmark_means
from simulation import (
    make_delay_sampler_factory,
    make_policy,
    make_reward_sampler_factory,
)
from bandit_environment import TemporaryHiringBanditEnv
from optimistic_hire import OptimisticHire


@dataclass(frozen=True)
class DiagnosticsSeries:
    policy_name: str
    label: str
    pairing: str


@dataclass(frozen=True)
class EpisodeDiagnostics:
    policy_name: str
    label: str
    pairing: str
    final_regret: float
    normalized_loss: float
    requested_replacements: int
    rejected_replacements: int
    accepted_replacements: int
    switch_periods: int
    pending_period_share: float
    accepted_delay_mean: float
    episode_runtime_sec: float
    selection_runtime_ms_mean: float
    selection_runtime_ms_max: float
    max_frontier_size: int


@dataclass(frozen=True)
class AggregateDiagnostics:
    policy_name: str
    label: str
    pairing: str
    n_runs: int
    final_regret_mean: float
    final_regret_std: float
    normalized_loss_mean: float
    normalized_loss_std: float
    rejected_share_mean: float
    rejected_share_std: float
    accepted_replacements_mean: float
    accepted_replacements_std: float
    switch_periods_mean: float
    switch_periods_std: float
    pending_period_share_mean: float
    pending_period_share_std: float
    accepted_delay_mean: float
    accepted_delay_std: float
    episode_runtime_sec_mean: float
    episode_runtime_sec_std: float
    selection_runtime_ms_mean: float
    selection_runtime_ms_std: float
    selection_runtime_ms_max: float
    frontier_size_max: int


SERIES: tuple[DiagnosticsSeries, ...] = (
    DiagnosticsSeries(policy_name="aht", label="AHT", pairing="Random"),
    DiagnosticsSeries(policy_name="aht-rm", label="AHT", pairing="Rank-matching"),
    DiagnosticsSeries(policy_name="omm", label="OMM", pairing="Random"),
    DiagnosticsSeries(policy_name="omm-rm", label="OMM", pairing="Rank-matching"),
    DiagnosticsSeries(policy_name="optimistic-hire", label="Optimistic-Hire", pairing="Rank-matching"),
)

k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
delay_lower = 8
n_runs = 2
n_jobs = 4
seed0 = 12345
review_interval = 6 * 30 * 24
work_trial_periods = 1
work_trial_rotation_periods = 90 * 24

means = make_benchmark_means(k=k)
reward_sampler_factory = make_reward_sampler_factory("bernoulli", means=means)
delay_sampler_factory = make_delay_sampler_factory(
    "uniform",
    means=means,
    omega_max=omega_max,
    delay_lower=delay_lower,
)
output_dir = benchmark_output_dir(
    module_file=__file__,
    output_subdir="benchmark_diagnostics",
)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _worker(spec: DiagnosticsSeries, seed: int) -> EpisodeDiagnostics:
    episode_reward_samplers = reward_sampler_factory(seed)
    episode_delay_sampler = delay_sampler_factory(seed)
    rng_seed = int(seed)

    policy_rng = random.Random(rng_seed)
    initial_workforce = policy_rng.sample(list(range(1, k + 1)), m)

    env = TemporaryHiringBanditEnv(
        k=k,
        m=m,
        reward_samplers=episode_reward_samplers,
        delay_sampler=episode_delay_sampler,
        c=c,
        omega_max=omega_max,
        true_means=means,
        initial_workforce=initial_workforce,
    )

    policy = make_policy(
        spec.policy_name,
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=omega_max,
        rng=policy_rng,
        review_interval=review_interval,
        work_trial_periods=work_trial_periods,
        work_trial_rotation_periods=work_trial_rotation_periods,
        gamma="auto",
        log_frontier_sizes=spec.policy_name == "optimistic-hire",
    )

    oracle_reward = env.optimal_expected_reward()
    if oracle_reward is None:
        raise ValueError("Expected true means to be available for diagnostics.")

    requested_replacements = 0
    rejected_replacements = 0
    accepted_replacements = 0
    switch_periods = 0
    pending_periods = 0
    accepted_delays: list[float] = []
    cumulative_regret = 0.0

    start_time = time.perf_counter()
    for _ in range(T):
        replacements = policy.act(env)
        requested_replacements += len(getattr(policy, "last_requested_replacements", []))
        rejected_replacements += (
            len(getattr(policy, "last_requested_replacements", []))
            - len(getattr(policy, "last_feasible_replacements", []))
        )

        _, _, cost_incurred, feedback = env.step(replacements)
        policy.update(feedback)

        accepted_count = len(feedback.accepted_replacements)
        accepted_replacements += accepted_count
        if accepted_count > 0:
            switch_periods += 1
            accepted_delays.extend(float(delay) for delay in feedback.accepted_delays)

        if feedback.pending_count > 0:
            pending_periods += 1

        active_expected = sum(means[i - 1] for i in feedback.active_set)
        cumulative_regret += (oracle_reward - active_expected) + cost_incurred

    episode_runtime_sec = time.perf_counter() - start_time

    selection_runtime_ms_mean = 0.0
    selection_runtime_ms_max = 0.0
    max_frontier_size = 0
    if isinstance(policy, OptimisticHire):
        if policy.selection_runtime_log:
            selection_runtime_ms_mean = 1000.0 * _safe_mean(policy.selection_runtime_log)
            selection_runtime_ms_max = 1000.0 * max(policy.selection_runtime_log)
        if policy.frontier_size_log:
            max_frontier_size = max(record.frontier_size for record in policy.frontier_size_log)

    return EpisodeDiagnostics(
        policy_name=spec.policy_name,
        label=spec.label,
        pairing=spec.pairing,
        final_regret=float(cumulative_regret),
        normalized_loss=float(cumulative_regret / (T * oracle_reward)),
        requested_replacements=requested_replacements,
        rejected_replacements=rejected_replacements,
        accepted_replacements=accepted_replacements,
        switch_periods=switch_periods,
        pending_period_share=float(pending_periods / T),
        accepted_delay_mean=_safe_mean(accepted_delays),
        episode_runtime_sec=episode_runtime_sec,
        selection_runtime_ms_mean=selection_runtime_ms_mean,
        selection_runtime_ms_max=selection_runtime_ms_max,
        max_frontier_size=max_frontier_size,
    )


def _worker_from_tuple(args: tuple[DiagnosticsSeries, int]) -> EpisodeDiagnostics:
    return _worker(*args)


def _aggregate(records: Sequence[EpisodeDiagnostics]) -> AggregateDiagnostics:
    if not records:
        raise ValueError("No diagnostic records were provided.")

    rejected_shares = [
        (record.rejected_replacements / record.requested_replacements)
        if record.requested_replacements > 0
        else 0.0
        for record in records
    ]

    return AggregateDiagnostics(
        policy_name=records[0].policy_name,
        label=records[0].label,
        pairing=records[0].pairing,
        n_runs=len(records),
        final_regret_mean=_safe_mean([record.final_regret for record in records]),
        final_regret_std=_std([record.final_regret for record in records]),
        normalized_loss_mean=_safe_mean([record.normalized_loss for record in records]),
        normalized_loss_std=_std([record.normalized_loss for record in records]),
        rejected_share_mean=_safe_mean(rejected_shares),
        rejected_share_std=_std(rejected_shares),
        accepted_replacements_mean=_safe_mean([record.accepted_replacements for record in records]),
        accepted_replacements_std=_std([record.accepted_replacements for record in records]),
        switch_periods_mean=_safe_mean([record.switch_periods for record in records]),
        switch_periods_std=_std([record.switch_periods for record in records]),
        pending_period_share_mean=_safe_mean([record.pending_period_share for record in records]),
        pending_period_share_std=_std([record.pending_period_share for record in records]),
        accepted_delay_mean=_safe_mean([record.accepted_delay_mean for record in records]),
        accepted_delay_std=_std([record.accepted_delay_mean for record in records]),
        episode_runtime_sec_mean=_safe_mean([record.episode_runtime_sec for record in records]),
        episode_runtime_sec_std=_std([record.episode_runtime_sec for record in records]),
        selection_runtime_ms_mean=_safe_mean([record.selection_runtime_ms_mean for record in records]),
        selection_runtime_ms_std=_std([record.selection_runtime_ms_mean for record in records]),
        selection_runtime_ms_max=max(record.selection_runtime_ms_max for record in records),
        frontier_size_max=max(record.max_frontier_size for record in records),
    )


def _format_regret(mean: float, std: float) -> str:
    return f"${int(round(mean)):,} \\pm {int(round(std)):,}$".replace(",", "{,}")


def _format_percent(mean: float, std: float) -> str:
    return f"${100.0 * mean:.2f}\\% \\pm {100.0 * std:.2f}\\%$"


def _format_float(mean: float, std: float, decimals: int = 1) -> str:
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _build_latex_table(aggregates: Sequence[AggregateDiagnostics]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(
        r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l r r r r r}"
    )
    lines.append(r"    \toprule")
    lines.append(
        r"    Policy & Pairing & 5-year regret & Rejected share & Accepted repl. & Switch periods & Pending-period share \\"
    )
    lines.append(r"    \midrule")
    for row in aggregates:
        lines.append(
            "    "
            + " & ".join(
                [
                    row.label,
                    row.pairing,
                    _format_regret(row.final_regret_mean, row.final_regret_std),
                    _format_percent(row.rejected_share_mean, row.rejected_share_std),
                    _format_float(row.accepted_replacements_mean, row.accepted_replacements_std, decimals=1),
                    _format_float(row.switch_periods_mean, row.switch_periods_std, decimals=1),
                    _format_percent(row.pending_period_share_mean, row.pending_period_share_std),
                ]
            )
            + r" \\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular*}")
    lines.append(
        r"    \caption{Diagnostics for the delayed-action adaptations of the benchmark policies in the benchmark setting of Section~\ref{numerical:benchmarks}. Rejected share reports the fraction of requested replacement pairs that were dropped because they conflicted with in-progress replacements.}"
    )
    lines.append(r"    \label{tab:benchmark-adaptation-diagnostics}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _build_runtime_summary(aggregates: Sequence[AggregateDiagnostics]) -> dict[str, Any]:
    runtime_summary: dict[str, Any] = {}
    for row in aggregates:
        if row.policy_name != "optimistic-hire":
            continue
        runtime_summary = {
            "policy": row.label,
            "pairing": row.pairing,
            "n_runs": row.n_runs,
            "episode_runtime_sec_mean": row.episode_runtime_sec_mean,
            "episode_runtime_sec_std": row.episode_runtime_sec_std,
            "selection_runtime_ms_mean": row.selection_runtime_ms_mean,
            "selection_runtime_ms_std": row.selection_runtime_ms_std,
            "selection_runtime_ms_max": row.selection_runtime_ms_max,
            "frontier_size_max": row.frontier_size_max,
            "accepted_delay_mean": row.accepted_delay_mean,
        }
        break
    return runtime_summary


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_policy: dict[str, list[EpisodeDiagnostics]] = {
        spec.policy_name: []
        for spec in SERIES
    }

    tasks = [
        (spec, seed0 + run_idx)
        for spec in SERIES
        for run_idx in range(n_runs)
    ]

    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(_worker_from_tuple, tasks):
                per_policy[result.policy_name].append(result)
    except (OSError, PermissionError):
        for task in tasks:
            result = _worker_from_tuple(task)
            per_policy[result.policy_name].append(result)

    ordered_aggregates = [
        _aggregate(per_policy[spec.policy_name])
        for spec in SERIES
    ]

    diagnostics_payload = {
        "config": {
            "k": k,
            "m": m,
            "T": T,
            "c": c,
            "omega_max": omega_max,
            "delay_lower": delay_lower,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
            "seed0": seed0,
        },
        "aggregates": [asdict(row) for row in ordered_aggregates],
        "optimistic_hire_runtime": _build_runtime_summary(ordered_aggregates),
    }

    (output_dir / "benchmark_diagnostics.json").write_text(
        json.dumps(diagnostics_payload, indent=2)
    )
    (output_dir / "benchmark_adaptation_diagnostics.tex").write_text(
        _build_latex_table(ordered_aggregates[:4])
    )

    print(json.dumps(diagnostics_payload["optimistic_hire_runtime"], indent=2))


if __name__ == "__main__":
    main()
