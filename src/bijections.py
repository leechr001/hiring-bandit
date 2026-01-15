from __future__ import annotations

from typing import List, Tuple, Sequence

import random

def random_bijection(
    current: Sequence[int],
    target: Sequence[int]
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
    
    random.shuffle(add)
    return list(zip(remove, add))

def oracle_rank_matching_bijection(
    current: Sequence[int],
    target: Sequence[int]
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