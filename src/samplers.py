import random
from typing import Sequence, Tuple

"""
Defines a number of samplers that will be helpful througout
"""

def make_bernoulli_samplers(means: Sequence[float], rng: random.Random):
    """
    Used for generation of rewards
    """
    def sampler(p):
        return lambda: 1.0 if rng.random() < p else 0.0
    return [sampler(p) for p in means]

def make_uniform_delay_sampler(omega_max: int, rng=None):
    """
    Generates delay completion times as iid uniform process.
    """
    rng = rng or random.Random()

    # iid so pair and t are not used.
    def sampler(pair: Tuple[int, int], t: int) -> int:
        return rng.randint(1, omega_max)
    return sampler

def make_adversarial_delay(means: Sequence[float], omega_max:int):
    """
    Computes and returns worst case delay from true means.
    Not really a "sampler" but keeping name for consistency.
    """
    def sampler(pair: Tuple[int, int], t: int) -> int:
        i,j = pair
        # if replacement is worse, execute imediately.
        # 1-indexed because matches paper and I like confusing code.
        if means[i-1] >= means[j-1]:
            return 1
        return omega_max

    return sampler
