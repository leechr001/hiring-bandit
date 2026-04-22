"""
Dynamic program for the horizon-aware ChooseTarget subroutine.
"""

from __future__ import annotations

from bijections import (
    optimistic_hire_switching_threshold,
    rank_matching_add_order,
    rank_matching_remove_order,
)

import math
from dataclasses import dataclass
from typing import List, MutableSequence, Sequence, Tuple

import numpy as np


Score = Tuple[int, float]


@dataclass(frozen=True)
class ChooseTargetResult:
    target: frozenset[int]
    matched_pairs: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class ChooseTargetFrontierSizeRecord:
    time_index: int
    current_period: int | None
    active_prefix_size: int
    replacement_count: int
    candidate_count: int
    frontier_size: int
    decision_iteration: int | None = None
    policy_name: str | None = None
    episode_seed: int | None = None


@dataclass(frozen=True)
class _RemovalLcbSum:
    has_negative_infinity: bool
    finite_sum: float


@dataclass(frozen=True)
class _RemovalFrontierEntry:
    removal_lcb: _RemovalLcbSum
    removal_ucb: Score
    prev_index: int | None
    took_worker: bool


def _validate_inputs(
    *,
    active_set: Sequence[int],
    counts: Sequence[int],
    empirical_means: Sequence[float],
    current_period: int,
    time_index: int | None,
    horizon: int,
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
    if horizon <= 0:
        raise ValueError("horizon-aware ChooseTarget requires a positive finite horizon.")
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


def _score_less(left: Score, right: Score, *, atol: float = 1e-12) -> bool:
    if left[0] != right[0]:
        return left[0] < right[0]
    return left[1] < right[1] - atol


def _score_leq(left: Score, right: Score, *, atol: float = 1e-12) -> bool:
    if left[0] != right[0]:
        return left[0] < right[0]
    return left[1] <= right[1] + atol


def _removal_lcb_of_value(lcb_value: float) -> _RemovalLcbSum:
    if math.isinf(lcb_value) and lcb_value < 0.0:
        return _RemovalLcbSum(has_negative_infinity=True, finite_sum=0.0)
    return _RemovalLcbSum(has_negative_infinity=False, finite_sum=float(lcb_value))


def _removal_lcb_add(left: _RemovalLcbSum, right: _RemovalLcbSum) -> _RemovalLcbSum:
    if left.has_negative_infinity or right.has_negative_infinity:
        return _RemovalLcbSum(has_negative_infinity=True, finite_sum=0.0)
    return _RemovalLcbSum(
        has_negative_infinity=False,
        finite_sum=left.finite_sum + right.finite_sum,
    )


def _removal_lcb_sort_key(value: _RemovalLcbSum) -> tuple[int, float]:
    if value.has_negative_infinity:
        return (0, 0.0)
    return (1, value.finite_sum)


def _removal_lcb_within_budget(
    removal_lcb: _RemovalLcbSum,
    *,
    budget: float,
    atol: float = 1e-12,
) -> bool:
    if removal_lcb.has_negative_infinity:
        return True
    return removal_lcb.finite_sum <= budget + atol


def _prune_removal_frontier(
    candidates: Sequence[_RemovalFrontierEntry],
) -> List[_RemovalFrontierEntry]:
    ordered = sorted(
        candidates,
        key=lambda entry: (
            *_removal_lcb_sort_key(entry.removal_lcb),
            entry.removal_ucb[0],
            entry.removal_ucb[1],
        ),
    )

    frontier: List[_RemovalFrontierEntry] = []
    best_removal_ucb: Score | None = None
    for candidate in ordered:
        if best_removal_ucb is not None and _score_leq(
            best_removal_ucb,
            candidate.removal_ucb,
        ):
            continue
        frontier.append(candidate)
        if best_removal_ucb is None or _score_less(
            candidate.removal_ucb,
            best_removal_ucb,
        ):
            best_removal_ucb = candidate.removal_ucb

    return frontier


def _best_feasible_frontier_index(
    frontier: Sequence[_RemovalFrontierEntry],
    *,
    incoming_ucb: Score,
    threshold: float,
    n_replacements: int,
) -> int | None:
    if not frontier:
        return None

    # On the pruned frontier, removal UCB strictly improves as removal LCB grows,
    # so the best feasible state is the last state within the LCB budget.
    if incoming_ucb[0] > 0:
        return len(frontier) - 1

    budget = incoming_ucb[1] - n_replacements * threshold
    lo = 0
    hi = len(frontier)
    while lo < hi:
        mid = (lo + hi) // 2
        if _removal_lcb_within_budget(frontier[mid].removal_lcb, budget=budget):
            lo = mid + 1
        else:
            hi = mid

    if lo == 0:
        return None
    return lo - 1


def _record_frontier_size(
    frontier_size_log: MutableSequence[ChooseTargetFrontierSizeRecord] | None,
    *,
    time_index: int,
    current_period: int | None,
    active_prefix_size: int,
    replacement_count: int,
    candidate_count: int,
    frontier_size: int,
) -> None:
    if frontier_size_log is None:
        return

    frontier_size_log.append(
        ChooseTargetFrontierSizeRecord(
            time_index=time_index,
            current_period=current_period,
            active_prefix_size=active_prefix_size,
            replacement_count=replacement_count,
            candidate_count=candidate_count,
            frontier_size=frontier_size,
        )
    )


def choose_target(
    *,
    active_set: Sequence[int],
    counts: Sequence[int],
    empirical_means: Sequence[float],
    current_period: int,
    horizon: int,
    switching_cost: float,
    ucb_coef: float = 1.0,
    time_index: int | None = None,
    frontier_size_log: MutableSequence[ChooseTargetFrontierSizeRecord] | None = None,
) -> ChooseTargetResult:
    """
    Solve the aggregate ChooseTarget optimization via dynamic programming.

    The incoming workers satisfy the monotonicity property from the exact
    selection rule: for any replacement count ``r``, it is enough to consider the
    top-``r`` inactive workers by UCB. For each such prefix, we solve the active
    side exactly via a cardinality-constrained knapsack frontier that minimizes
    removed UCB subject to the aggregate LCB budget.

    The aggregate horizon-aware screening constraint is:

        sum_{(j, i) in pi^R} (UCB_i - LCB_j) >= (# replacements) * c / (T - t)

    where T is the known horizon and t is the current period.
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
    max_replacements = min(m, k - m)

    effective_time_index = current_period if time_index is None else time_index
    assert effective_time_index is not None

    ucb_values, lcb_values = _compute_confidence_bounds(
        counts=counts,
        empirical_means=empirical_means,
        time_index=effective_time_index,
        ucb_coef=ucb_coef,
    )
    ucb_sequence = ucb_values.tolist()

    threshold = optimistic_hire_switching_threshold(
        horizon=horizon,
        current_period=current_period,
        switching_cost=switching_cost,
    )
    assert threshold is not None
    if math.isinf(threshold):
        return ChooseTargetResult(target=frozenset(active), matched_pairs=())

    active_set_lookup = set(active)
    inactive = [
        worker_id
        for worker_id in range(1, k + 1)
        if worker_id not in active_set_lookup
    ]

    active_ranked = rank_matching_remove_order(
        active,
        ucb_values=ucb_sequence,
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
    active_total_score = (0, 0.0)
    for worker_id in active_ranked:
        active_total_score = _score_add(active_total_score, active_scores[worker_id])

    incoming_prefix_scores: List[Score] = [(0, 0.0)]
    running_incoming_score = (0, 0.0)
    for worker_id in inactive_ranked[:max_replacements]:
        running_incoming_score = _score_add(
            running_incoming_score,
            inactive_scores[worker_id],
        )
        incoming_prefix_scores.append(running_incoming_score)

    n_active = len(active_ranked)
    removal_frontiers: List[List[List[_RemovalFrontierEntry]]] = [
        [[] for _ in range(max_replacements + 1)]
        for _ in range(n_active + 1)
    ]
    removal_frontiers[0][0] = [
        _RemovalFrontierEntry(
            removal_lcb=_RemovalLcbSum(has_negative_infinity=False, finite_sum=0.0),
            removal_ucb=(0, 0.0),
            prev_index=None,
            took_worker=False,
        )
    ]
    _record_frontier_size(
        frontier_size_log,
        time_index=effective_time_index,
        current_period=current_period,
        active_prefix_size=0,
        replacement_count=0,
        candidate_count=1,
        frontier_size=1,
    )

    for i in range(1, n_active + 1):
        worker_id = active_ranked[i - 1]
        worker_ucb = active_scores[worker_id]
        worker_lcb = _removal_lcb_of_value(float(lcb_values[worker_id - 1]))
        max_size_here = min(i, max_replacements)

        for r in range(max_size_here + 1):
            candidates: List[_RemovalFrontierEntry] = []

            for prev_index, previous in enumerate(removal_frontiers[i - 1][r]):
                candidates.append(
                    _RemovalFrontierEntry(
                        removal_lcb=previous.removal_lcb,
                        removal_ucb=previous.removal_ucb,
                        prev_index=prev_index,
                        took_worker=False,
                    )
                )

            if r > 0:
                for prev_index, previous in enumerate(removal_frontiers[i - 1][r - 1]):
                    candidates.append(
                        _RemovalFrontierEntry(
                            removal_lcb=_removal_lcb_add(
                                previous.removal_lcb,
                                worker_lcb,
                            ),
                            removal_ucb=_score_add(
                                previous.removal_ucb,
                                worker_ucb,
                            ),
                            prev_index=prev_index,
                            took_worker=True,
                        )
                    )

            removal_frontiers[i][r] = _prune_removal_frontier(candidates)
            _record_frontier_size(
                frontier_size_log,
                time_index=effective_time_index,
                current_period=current_period,
                active_prefix_size=i,
                replacement_count=r,
                candidate_count=len(candidates),
                frontier_size=len(removal_frontiers[i][r]),
            )

    best_score = active_total_score
    best_r = 0
    best_frontier_index: int | None = None

    for r in range(1, max_replacements + 1):
        best_index_for_r = _best_feasible_frontier_index(
            removal_frontiers[n_active][r],
            incoming_ucb=incoming_prefix_scores[r],
            threshold=threshold,
            n_replacements=r,
        )
        if best_index_for_r is None:
            continue

        removal_entry = removal_frontiers[n_active][r][best_index_for_r]
        candidate_score = _score_add(
            _score_sub(active_total_score, removal_entry.removal_ucb),
            incoming_prefix_scores[r],
        )
        if _score_better(candidate_score, best_score):
            best_score = candidate_score
            best_r = r
            best_frontier_index = best_index_for_r

    if best_r == 0 or best_frontier_index is None:
        return ChooseTargetResult(target=frozenset(active), matched_pairs=())

    removed_workers: List[int] = []
    frontier_index = best_frontier_index
    r = best_r
    for i in range(n_active, 0, -1):
        entry = removal_frontiers[i][r][frontier_index]
        if entry.took_worker:
            removed_workers.append(active_ranked[i - 1])
            r -= 1
        assert entry.prev_index is not None
        frontier_index = entry.prev_index

    removed_workers.reverse()
    added_workers = inactive_ranked[:best_r]

    target = set(active)
    for worker_id in removed_workers:
        target.remove(worker_id)
    for worker_id in added_workers:
        target.add(worker_id)

    removed_ranked = rank_matching_remove_order(
        removed_workers,
        ucb_values=ucb_sequence,
    )

    return ChooseTargetResult(
        target=frozenset(target),
        matched_pairs=tuple(zip(removed_ranked, added_workers)),
    )
