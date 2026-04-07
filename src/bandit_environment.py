from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union
import random


RewardSampler = Callable[[], float]
DelaySampler = Callable[[Tuple[int, int], int], int]  # (i, j), current time -> omega


@dataclass
class PendingReplacement:
    """A committed one-for-one replacement that will complete at a future period."""
    i: int
    j: int
    start_time: int
    completion_time: int


@dataclass(frozen=True)
class EnvObservation:
    """Observable environment state at the start of a period."""
    t: int
    active_set: FrozenSet[int]
    pending_replacements: Tuple[Tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class StepFeedback:
    """Semi-bandit feedback and transition details from one environment step."""
    individual_rewards: Dict[int, float]
    active_set: FrozenSet[int]
    completed_this_period: Tuple[Tuple[int, int], ...]
    pending_count: int


class TemporaryHiringBanditEnv:
    """
    Environment for the sequential hiring model with:
      - k workers (arms)
      - active workforce size m
      - stochastic execution delays for replacements
      - per-replacement switching cost c

    Period t proceeds in three stages:
      1) Agent initiates replacements R_t, paying c * |R_t|.
      2) Previously initiated replacements that complete at t are realized,
         updating the active roster A_t.
      3) Active workers generate individual rewards; the period reward is their sum.
    """

    def __init__(
        self,
        k: int,
        m: int,
        reward_samplers: Union[Sequence[RewardSampler], Dict[int, RewardSampler]],
        delay_sampler: DelaySampler,
        *,
        c: float = 0.0,
        omega_max: int = 1,
        initial_workforce: Optional[Sequence[int]] = None,
        rng: Optional[random.Random] = None,
        true_means: Optional[Sequence[float]] = None,
    ):
        if k < 2:
            raise ValueError("k must be >= 2.")
        if not (1 <= m < k):
            raise ValueError("m must satisfy 1 <= m < k.")
        if omega_max < 1:
            raise ValueError("omega_max must be >= 1.")
        if c < 0:
            raise ValueError("c must be nonnegative.")

        self.k = int(k)
        self.m = int(m)
        self.c = float(c)
        self.omega_max = int(omega_max)
        self.rng = rng or random.Random()

        # Reward samplers keyed by worker id in {1,...,k}
        if isinstance(reward_samplers, dict):
            self.reward_samplers = {int(i): reward_samplers[i] for i in reward_samplers}
        else:
            if len(reward_samplers) != k:
                raise ValueError("reward_samplers must have length k.")
            self.reward_samplers = {i + 1: reward_samplers[i] for i in range(k)}

        missing = [i for i in range(1, k + 1) if i not in self.reward_samplers]
        if missing:
            raise ValueError(f"Missing reward samplers for workers: {missing}")

        # Optional: store true means for regret/oracle evaluation in simulation
        self.true_means = list(true_means) if true_means is not None else None
        if self.true_means is not None and len(self.true_means) != k:
            raise ValueError("true_means must have length k if provided.")

        self.delay_sampler = delay_sampler

        self.t: int = 0
        self.active_set: Set[int] = set()
        self.pending: List[PendingReplacement] = []
        self.pending_workers: Set[int] = set()

        if initial_workforce is None:
            initial_workforce = list(range(1, m + 1))
        self.reset(initial_workforce)

    def reset(self, initial_workforce: Optional[Sequence[int]] = None) -> EnvObservation:
        """Reset environment to period 1 with a specified initial active workforce."""
        if initial_workforce is None:
            initial_workforce = list(range(1, self.m + 1))
        init = list(initial_workforce)

        if len(init) != self.m or len(set(init)) != self.m:
            raise ValueError("initial_workforce must contain exactly m distinct workers.")
        if any(w < 1 or w > self.k for w in init):
            raise ValueError("initial_workforce contains invalid worker id.")

        self.t = 1
        self.active_set = set(init)
        self.pending = []
        self.pending_workers = set()
        return self._get_obs()

    def _get_obs(self) -> EnvObservation:
        """Return a simple observable state for simulation/debugging."""
        return EnvObservation(
            t=self.t,
            active_set=frozenset(self.active_set),
            pending_replacements=tuple(
                (pr.i, pr.j, pr.start_time, pr.completion_time) for pr in self.pending
            ),
        )

    def _workers_in_pending(self) -> Set[int]:
        return set(self.pending_workers)

    def _validate_single_replacement(
        self,
        i: int,
        j: int,
        *,
        pending_workers: Set[int],
        seen_remove: Set[int],
        seen_add: Set[int],
    ) -> None:
        if not (1 <= i <= self.k and 1 <= j <= self.k):
            raise ValueError(f"Invalid worker ids in replacement ({i}, {j}).")
        if i == j:
            raise ValueError("Replacement must be between distinct workers.")
        if i not in self.active_set:
            raise ValueError(f"Cannot replace {i}: not currently active.")
        if j in self.active_set:
            raise ValueError(f"Cannot add {j}: already active.")
        if i in pending_workers or j in pending_workers:
            raise ValueError(
                f"Cannot use {i} or {j}: worker appears in a pending replacement."
            )
        if i in seen_remove:
            raise ValueError(f"Worker {i} appears twice as a removed worker in R_t.")
        if j in seen_add:
            raise ValueError(f"Worker {j} appears twice as an added worker in R_t.")

    def can_append_replacement(
        self,
        accepted: Sequence[Tuple[int, int]],
        pair: Tuple[int, int],
    ) -> bool:
        pending_workers = self.pending_workers
        seen_remove = {remove_id for remove_id, _ in accepted}
        seen_add = {add_id for _, add_id in accepted}
        try:
            self._validate_single_replacement(
                int(pair[0]),
                int(pair[1]),
                pending_workers=pending_workers,
                seen_remove=seen_remove,
                seen_add=seen_add,
            )
        except ValueError:
            return False
        return True

    def validate_replacements(self, replacements: Sequence[Tuple[int, int]]) -> None:
        """
        Feasibility checks aligned with the paper's modeling constraints:
          - one-for-one replacements
          - no worker can appear in more than one pending replacement at a time
          - remove workers must currently be active
          - add workers must not currently be active
        """
        pending_workers = self.pending_workers
        seen_remove: Set[int] = set()
        seen_add: Set[int] = set()

        for (i, j) in replacements:
            self._validate_single_replacement(
                int(i),
                int(j),
                pending_workers=pending_workers,
                seen_remove=seen_remove,
                seen_add=seen_add,
            )
            seen_remove.add(i)
            seen_add.add(j)

    def step(
        self,
        replacements: Optional[Sequence[Tuple[int, int]]] = None
    ) -> Tuple[EnvObservation, float, float, StepFeedback]:
        """
        Advance one period.

        Args:
            replacements: sequence of (i, j) pairs initiated at current time t.

        Returns:
            obs: next-period observation dict
            reward: sum of individual rewards for active workers this period
            cost: switching cost incurred this period
            info: debugging details (individual rewards, completions, etc.)
        """
        if replacements is None:
            replacements = []
        replacements = list(replacements)

        # 1) Initiate replacements and pay cost.
        if replacements:
            self.validate_replacements(replacements)

        cost = self.c * len(replacements)
        for (i, j) in replacements:
            omega = int(self.delay_sampler((i, j), self.t))
            if not (1 <= omega <= self.omega_max):
                raise ValueError("delay_sampler returned omega outside [1, omega_max].")
            self.pending.append(
                PendingReplacement(
                    i=i,
                    j=j,
                    start_time=self.t,
                    completion_time=self.t + omega
                )
            )
            self.pending_workers.add(i)
            self.pending_workers.add(j)

        # 2) Realize previously initiated replacements that complete at t.
        completing = [pr for pr in self.pending if pr.completion_time == self.t]
        if completing:
            for pr in completing:
                self.active_set.discard(pr.i)
            for pr in completing:
                self.active_set.add(pr.j)
            self.pending = [pr for pr in self.pending if pr.completion_time != self.t]
            for pr in completing:
                self.pending_workers.discard(pr.i)
                self.pending_workers.discard(pr.j)

        if len(self.active_set) != self.m:
            raise RuntimeError(
                "Active set size violated. Check feasibility rules / delay model."
            )

        # 3) Generate rewards for active workers.
        individual: Dict[int, float] = {}
        total = 0.0
        for i in sorted(self.active_set):
            r = float(self.reward_samplers[i]())
            individual[i] = r
            total += r

        feedback = StepFeedback(
            individual_rewards=dict(individual),
            active_set=frozenset(self.active_set),
            completed_this_period=tuple((pr.i, pr.j) for pr in completing),
            pending_count=len(self.pending),
        )

        # Advance time
        self.t += 1
        return self._get_obs(), total, cost, feedback

    # --- Convenience helpers ---

    def optimal_team(self) -> Optional[Set[int]]:
        """
        Return the oracle best-m set by true means (if provided).
        """
        means = self.true_means
        if means is None:
            return None

        order = sorted(
            range(1, self.k + 1),
            key=lambda i: means[i - 1],
            reverse=True
        )
        return set(order[:self.m])

    def optimal_expected_reward(self) -> Optional[float]:
        """Expected per-period reward of the oracle best-m team (if means provided)."""
        means = self.true_means
        if means is None:
            return None

        team = self.optimal_team()
        if team is None:
            return None

        return sum(means[i - 1] for i in team)
