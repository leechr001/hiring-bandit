from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Optional

import math
import random

def _ignore_rng(
    bijection_fn: Callable[[Sequence[int], Sequence[int]], List[Tuple[int, int]]],
) -> Callable[[Sequence[int], Sequence[int]], List[Tuple[int, int]]]:
    def wrapped(
        current: Sequence[int],
        target: Sequence[int],
        **_kwargs,
    ) -> List[Tuple[int, int]]:
        return bijection_fn(current, target)

    return wrapped

def make_bijection(bijection_name:str) -> Callable:
    """
    Factory to build bijections that be reused between policies.
    """
    if bijection_name.lower() in ('random', 'stochastic'):
        return random_bijection
    elif bijection_name.lower() in ('oracle-match', 'rank-matching', 'rm', 'best'):
        return _ignore_rng(oracle_rank_matching_bijection)
    elif bijection_name.lower() in ('oracle-mismatch', 'rank-mismatching', 'rmm', 'worst'):
        return _ignore_rng(oracle_rank_mismatching_bijection)
    
    raise ValueError(f"{bijection_name} is not recognized")

def random_bijection(
    current: Sequence[int],
    target: Sequence[int],
    *,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    """
    Construct a random bijection, needed to adapt policies
    """
    cur = set(current)
    tar = set(target)
    remove = sorted(cur - tar)
    add = sorted(tar - cur)
    if len(remove) != len(add):
        raise ValueError("current and target must differ by equal counts.")
    
    if rng is None:
        random.shuffle(add)
    else:
        rng.shuffle(add)
    return list(zip(remove, add))

def oracle_rank_matching_bijection(
    current: Sequence[int],
    target: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Construct a rank matching objective based on true means
    """
    cur_set = set(current)
    tar_set = set(target)

    remove = sorted(cur_set - tar_set)
    add = sorted(tar_set - cur_set)

    # Defensive truncation if environment sizes mismatch.
    if len(remove) != len(add):
        n = min(len(remove), len(add))
        remove, add = remove[:n], add[:n]

    return list(zip(remove, add))

def oracle_rank_mismatching_bijection(
    current: Sequence[int],
    target: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Construct a rank matching objective based on true means
    """
    cur_set = set(current)
    tar_set = set(target)

    remove = sorted(cur_set - tar_set)
    add = sorted(tar_set - cur_set)

    # reverse the list so ranks are most different
    add.reverse()

    # Defensive truncation if environment sizes mismatch.
    if len(remove) != len(add):
        n = min(len(remove), len(add))
        remove, add = remove[:n], add[:n]

    return list(zip(remove, add))


def rank_matching_remove_order(
    workers: Sequence[int],
    *,
    ucb_values: Sequence[float],
) -> List[int]:
    """Order active workers for pi^R by descending UCB, breaking ties by worker ID."""
    return sorted(
        (int(worker_id) for worker_id in workers),
        key=lambda worker_id: (float(ucb_values[worker_id - 1]), -worker_id),
        reverse=True,
    )


def rank_matching_add_order(
    workers: Sequence[int],
    *,
    ucb_values: Sequence[float],
) -> List[int]:
    """Order inactive workers for pi^R by descending UCB, breaking ties by worker ID."""
    return sorted(
        (int(worker_id) for worker_id in workers),
        key=lambda worker_id: (float(ucb_values[worker_id - 1]), -worker_id),
        reverse=True,
    )


def delayed_replace_ucb_switching_threshold(
    *,
    horizon: Optional[int],
    current_period: Optional[int],
    switching_cost: float,
) -> Optional[float]:
    """Return c / (T - t) when a finite horizon is active, else None."""
    if horizon is None or current_period is None:
        return None

    remaining_periods = int(horizon) - int(current_period)
    if remaining_periods <= 0:
        return math.inf

    return float(switching_cost) / float(remaining_periods)


def delayed_replace_ucb_rank_matching_bijection(
    current: Sequence[int],
    target: Sequence[int],
    *,
    ucb_values: Sequence[float],
) -> List[Tuple[int, int]]:
    """
    Construct the delayed-replace-ucb rank-matching bijection pi^R.

    Removed workers are ordered by descending UCB and added workers are ordered by
    descending UCB.
    """
    cur_set = set(current)
    tar_set = set(target)

    remove = rank_matching_remove_order(
        list(cur_set - tar_set),
        ucb_values=ucb_values,
    )
    add = rank_matching_add_order(
        list(tar_set - cur_set),
        ucb_values=ucb_values,
    )

    if not remove or not add:
        return []

    if len(remove) != len(add):
        n = min(len(remove), len(add))
        remove = remove[:n]
        add = add[:n]

    return list(zip(remove, add))
