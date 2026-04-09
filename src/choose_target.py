"""
Dynamic program for the horizon-aware ChooseTarget subroutine.
"""

from __future__ import annotations

from bijections import (
    optimistic_hire_replacement_is_admissible,
    optimistic_hire_switching_threshold,
    rank_matching_add_order,
    rank_matching_remove_order,
)

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


Score = Tuple[int, float]


@dataclass(frozen=True)
class ChooseTargetResult:
    target: frozenset[int]
    matched_pairs: Tuple[Tuple[int, int], ...]


def _validate_inputs(
    *,
    active_set: Sequence[int],
    counts: Sequence[int],
    empirical_means: Sequence[float],
    current_period: int | None,
    time_index: int | None,
    horizon: int | None,
    switching_cost: float,
    ucb_coef: float,
) -> None:
    if len(counts) == 0:
        raise ValueError("counts must be non-empty.")
    if len(empirical_means) != len(counts):
        raise ValueError("empirical_means and counts must have the same length.")

    active = sorted(set(int(worker_id) for worker_id in active_set))
    if len(active) != len(active_set):
        raise ValueError("active_set must contain distinct worker IDs.")

    k = len(counts)
    for worker_id in active:
        if worker_id < 1 or worker_id > k:
            raise ValueError("active_set must contain worker IDs in [1, k].")

    if current_period is not None and current_period < 0:
        raise ValueError("current_period must be non-negative.")
    if time_index is not None and time_index < 0:
        raise ValueError("time_index must be non-negative.")
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive when provided.")
    if switching_cost < 0.0:
        raise ValueError("switching_cost must be non-negative.")
    if ucb_coef < 0.0:
        raise ValueError("ucb_coef must be non-negative.")


def _confidence_radius(
    count: int,
    *,
    time_index: int,
    ucb_coef: float,
) -> float:
    if count <= 0:
        return float("inf")
    if ucb_coef == 0.0:
        return 0.0

    t_for_log = max(1, time_index)
    log_term = math.log(t_for_log)
    return math.sqrt(ucb_coef * log_term / count)


def _compute_confidence_bounds(
    *,
    counts: Sequence[int],
    empirical_means: Sequence[float],
    time_index: int,
    ucb_coef: float,
) -> Tuple[np.ndarray, np.ndarray]:
    counts_arr = np.asarray(counts, dtype=np.int64)
    means_arr = np.asarray(empirical_means, dtype=np.float64)
    k = int(len(counts_arr))

    ucb = np.empty(k, dtype=np.float64)
    lcb = np.empty(k, dtype=np.float64)

    for idx in range(k):
        count = int(counts_arr[idx])
        if count <= 0:
            ucb[idx] = float("inf")
            lcb[idx] = -float("inf")
            continue

        rad = _confidence_radius(
            count,
            time_index=time_index,
            ucb_coef=ucb_coef,
        )
        ucb[idx] = means_arr[idx] + rad
        lcb[idx] = means_arr[idx] - rad

    return ucb, lcb


def _objective_score(ucb_value: float) -> Score:
    if math.isinf(ucb_value) and ucb_value > 0.0:
        return (1, 0.0)
    return (0, float(ucb_value))


def _score_add(left: Score, right: Score) -> Score:
    return (left[0] + right[0], left[1] + right[1])


def _score_sub(left: Score, right: Score) -> Score:
    return (left[0] - right[0], left[1] - right[1])


def _score_better(left: Score, right: Score, *, atol: float = 1e-12) -> bool:
    if left[0] != right[0]:
        return left[0] > right[0]
    return left[1] > right[1] + atol


def choose_target(
    *,
    active_set: Sequence[int],
    counts: Sequence[int],
    empirical_means: Sequence[float],
    current_period: int | None,
    horizon: int | None,
    switching_cost: float,
    ucb_coef: float = 1.0,
    time_index: int | None = None,
) -> ChooseTargetResult:
    """
    Solve the ChooseTarget optimization via dynamic programming.

    The DP treats the replacement decision as an order-preserving matching between
    active workers sorted by descending LCB and inactive workers sorted by descending UCB.
    Matching worker ``j`` to worker ``i`` corresponds to selecting a rank-matched
    replacement ``i -> j`` and is only allowed when the horizon-aware switching
    condition is satisfied.
    """
    _validate_inputs(
        active_set=active_set,
        counts=counts,
        empirical_means=empirical_means,
        current_period=current_period,
        time_index=time_index,
        horizon=horizon,
        switching_cost=switching_cost,
        ucb_coef=ucb_coef,
    )

    active = sorted(set(int(worker_id) for worker_id in active_set))
    k = len(counts)
    m = len(active)

    effective_time_index = current_period if time_index is None else time_index
    assert effective_time_index is not None

    ucb_values, lcb_values = _compute_confidence_bounds(
        counts=counts,
        empirical_means=empirical_means,
        time_index=effective_time_index,
        ucb_coef=ucb_coef,
    )
    ucb_sequence = ucb_values.tolist()
    lcb_sequence = lcb_values.tolist()

    threshold = optimistic_hire_switching_threshold(
        horizon=horizon,
        current_period=current_period,
        switching_cost=switching_cost,
    )
    if threshold is not None and math.isinf(threshold):
        return ChooseTargetResult(target=frozenset(active), matched_pairs=())
    if threshold is None:
        threshold = 0.0

    active_set_lookup = set(active)
    inactive = [
        worker_id
        for worker_id in range(1, k + 1)
        if worker_id not in active_set_lookup
    ]

    active_ranked = rank_matching_remove_order(
        active,
        lcb_values=lcb_sequence,
    )
    inactive_ranked = rank_matching_add_order(
        inactive,
        ucb_values=ucb_sequence,
    )

    active_scores = {
        worker_id: _objective_score(float(ucb_values[worker_id - 1]))
        for worker_id in active_ranked
    }
    inactive_scores = {
        worker_id: _objective_score(float(ucb_values[worker_id - 1]))
        for worker_id in inactive_ranked
    }

    n_active = len(active_ranked)
    n_inactive = len(inactive_ranked)

    dp: List[List[Score]] = [
        [(0, 0.0) for _ in range(n_inactive + 1)]
        for _ in range(n_active + 1)
    ]
    parent: List[List[str | None]] = [
        [None for _ in range(n_inactive + 1)]
        for _ in range(n_active + 1)
    ]

    for i in range(1, n_active + 1):
        parent[i][0] = "skip_active"
    for j in range(1, n_inactive + 1):
        parent[0][j] = "skip_inactive"

    for i in range(1, n_active + 1):
        remove_id = active_ranked[i - 1]
        remove_score = active_scores[remove_id]

        for j in range(1, n_inactive + 1):
            add_id = inactive_ranked[j - 1]
            add_score = inactive_scores[add_id]

            best = dp[i - 1][j]
            choice = "skip_active"

            if _score_better(dp[i][j - 1], best):
                best = dp[i][j - 1]
                choice = "skip_inactive"

            if optimistic_hire_replacement_is_admissible(
                remove_id,
                add_id,
                lcb_values=lcb_sequence,
                ucb_values=ucb_sequence,
                threshold=threshold,
            ):
                gain = _score_sub(add_score, remove_score)
                candidate = _score_add(dp[i - 1][j - 1], gain)
                if _score_better(candidate, best):
                    best = candidate
                    choice = "match"

            dp[i][j] = best
            parent[i][j] = choice

    removed_workers: List[int] = []
    added_workers: List[int] = []
    i = n_active
    j = n_inactive
    while i > 0 or j > 0:
        choice = parent[i][j]
        if choice == "match":
            removed_workers.append(active_ranked[i - 1])
            added_workers.append(inactive_ranked[j - 1])
            i -= 1
            j -= 1
        elif choice == "skip_active":
            i -= 1
        elif choice == "skip_inactive":
            j -= 1
        else:
            break

    removed_workers.reverse()
    added_workers.reverse()

    target = set(active)
    for worker_id in removed_workers:
        target.remove(worker_id)
    for worker_id in added_workers:
        target.add(worker_id)

    return ChooseTargetResult(
        target=frozenset(target),
        matched_pairs=tuple(zip(removed_workers, added_workers)),
    )
