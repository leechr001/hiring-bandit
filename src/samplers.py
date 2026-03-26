import math
import random
from typing import Optional, Sequence, Tuple

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


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _truncated_normal_mean(
    location: float,
    *,
    stddev: float,
    lower: float,
    upper: float,
) -> float:
    alpha = (lower - location) / stddev
    beta = (upper - location) / stddev
    z = _normal_cdf(beta) - _normal_cdf(alpha)
    if z <= 0.0:
        raise ValueError("Truncated normal normalization constant must be positive.")
    return location + stddev * (_normal_pdf(alpha) - _normal_pdf(beta)) / z


def _calibrate_truncated_normal_location(
    target_mean: float,
    *,
    stddev: float,
    lower: float,
    upper: float,
) -> float:
    if not (lower <= target_mean <= upper):
        raise ValueError("Each truncated-normal mean must lie within [lower, upper].")

    lo = lower - 12.0 * stddev
    hi = upper + 12.0 * stddev
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _truncated_normal_mean(mid, stddev=stddev, lower=lower, upper=upper) < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_truncated_normal_samplers(
    means: Sequence[float],
    rng: random.Random,
    *,
    stddev: float = 0.1,
    lower: float = 0.0,
    upper: float = 1.0,
):
    """
    Generate reward samplers from truncated normal distributions on [lower, upper].

    The provided ``means`` are treated as the desired expected rewards. Each arm's
    underlying normal location is calibrated so the truncated distribution matches
    that mean up to numerical tolerance.
    """
    if stddev <= 0.0:
        raise ValueError("stddev must be positive.")
    if lower >= upper:
        raise ValueError("Require lower < upper for truncated normal rewards.")

    locations = [
        _calibrate_truncated_normal_location(
            float(mean),
            stddev=stddev,
            lower=lower,
            upper=upper,
        )
        for mean in means
    ]

    def sampler(location: float):
        def draw() -> float:
            while True:
                sample = rng.gauss(location, stddev)
                if lower <= sample <= upper:
                    return float(sample)

        return draw

    return [sampler(location) for location in locations]

def make_uniform_delay_sampler(omega_max: int, rng=None):
    """
    Generates delay completion times as iid uniform process.
    """
    rng = rng or random.Random()

    # iid so pair and t are not used.
    def sampler(pair: Tuple[int, int], t: int) -> int:
        return rng.randint(1, omega_max)
    return sampler


def _validate_calendar_delay_params(
    *,
    omega_max: int,
    frequency: int,
) -> None:
    if omega_max < 1:
        raise ValueError("omega_max must be >= 1.")
    if frequency < 1:
        raise ValueError("frequency must be >= 1.")


def _calendar_feasible_delays(t: int, *, omega_max: int, frequency: int) -> Sequence[int]:
    _validate_calendar_delay_params(
        omega_max=omega_max,
        frequency=frequency,
    )
    if t < 1:
        raise ValueError("t must be >= 1.")

    first_delay = frequency - (t % frequency)
    if first_delay == 0:
        first_delay = frequency

    return list(range(first_delay, omega_max + 1, frequency))


def _require_calendar_feasible_delays(
    t: int,
    *,
    omega_max: int,
    frequency: int,
) -> Sequence[int]:
    feasible_delays = _calendar_feasible_delays(
        t,
        omega_max=omega_max,
        frequency=frequency,
    )
    if not feasible_delays:
        raise ValueError(
            "No feasible calendar-aligned completion times exist within omega_max "
            f"for t={t}, frequency={frequency}, omega_max={omega_max}."
        )
    return feasible_delays


def make_calendar_delay_sampler(
    omega_max: int,
    *,
    frequency: int,
    distribution: str = "geom",
    geom_p: float = 0.5,
    rng: Optional[random.Random] = None,
):
    """
    Sample delays whose completion times land on a fixed calendar.

    Example: if periods are hours and ``frequency=8``, then a replacement started
    at time ``t`` may only complete at times ``t + omega`` that are multiples of 8.
    Feasible delays are therefore the positive values in ``[1, omega_max]`` such
    that ``(t + omega) % frequency == 0``.

    Parameters
    ----------
    omega_max:
        Maximum allowed delay.
    frequency:
        Calendar spacing between feasible completion times.
    distribution:
        One of ``"geom"``/``"geometric"`` or ``"unif"``/``"uniform"``.
    geom_p:
        Success probability for the truncated geometric distribution over the
        ordered feasible completion periods. Ignored for uniform sampling.
    """
    _validate_calendar_delay_params(
        omega_max=omega_max,
        frequency=frequency,
    )
    if not (0.0 < geom_p <= 1.0):
        raise ValueError("geom_p must satisfy 0 < geom_p <= 1.")

    if rng is None:
        rng = random.Random()

    normalized_distribution = distribution.lower().strip()
    if normalized_distribution not in {"geom", "geometric", "unif", "uniform"}:
        raise ValueError("distribution must be one of: 'geom', 'geometric', 'unif', 'uniform'.")

    def sampler(pair: Tuple[int, int], t: int) -> int:
        del pair

        feasible_delays = _require_calendar_feasible_delays(
            t,
            omega_max=omega_max,
            frequency=frequency,
        )

        if normalized_distribution in {"unif", "uniform"}:
            return int(rng.choice(feasible_delays))

        weights = [geom_p * ((1.0 - geom_p) ** idx) for idx in range(len(feasible_delays))]
        total_weight = sum(weights)
        draw = rng.random() * total_weight
        cumulative = 0.0
        for delay, weight in zip(feasible_delays, weights):
            cumulative += weight
            if draw <= cumulative:
                return int(delay)

        return int(feasible_delays[-1])

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


def make_calendar_adversarial_delay(
    means: Sequence[float],
    omega_max: int,
    *,
    frequency: int,
):
    """
    Adversarial delay process restricted to calendar-aligned completion times.
    """
    _validate_calendar_delay_params(
        omega_max=omega_max,
        frequency=frequency,
    )

    def sampler(pair: Tuple[int, int], t: int) -> int:
        i, j = pair
        feasible_delays = _require_calendar_feasible_delays(
            t,
            omega_max=omega_max,
            frequency=frequency,
        )

        if means[i - 1] >= means[j - 1]:
            return int(feasible_delays[0])
        return int(feasible_delays[-1])

    return sampler
