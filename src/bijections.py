from __future__ import annotations

from typing import List, Tuple, Sequence, Callable, Optional

import random

def make_bijection(bijection_name:str) -> Callable:
    """
    Factory to build bijections that be reused between policies.
    """
    if bijection_name.lower() in ('random', 'stochastic'):
        return random_bijection
    elif bijection_name.lower() in ('oracle-match', 'rank-matching', 'rm', 'best'):
        return oracle_rank_matching_bijection
    elif bijection_name.lower() in ('oracle-mismatch', 'rank-mismatching', 'rmm', 'worst'):
        return oracle_rank_mismatching_bijection
    
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
    *,
    rng: Optional[random.Random] = None,
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
    *,
    rng: Optional[random.Random] = None,
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
