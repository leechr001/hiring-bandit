from __future__ import annotations

from bandit_environment import StepFeedback
from bijections import make_bijection

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

class DelayedActionPolicy(ABC):
    """
    Base class for policies that (a) maintain internal statistics via semi-bandit feedback,
    (b) choose a target team of size m from k workers, and (c) propose feasible replacements
    to an environment with delayed/validated action constraints.

    Subclasses must implement:
      - reset()
      - update(feedback)
      - compute_target()
    """

    def __init__(
        self,
        k: int,
        m: int,
        *,
        c: float = 0.0,
        omega_max: int = 0,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
    ) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not isinstance(m, int) or not (1 <= m < k):
            raise ValueError("Require 1 <= m < k.")

        self.k = k
        self.m = m

        # Optional parameters used by some delayed-action policies.
        self.c = float(c)
        self.omega_max = int(omega_max)

        # Set the type of bijection to be used
        self._bijection_fn = make_bijection(bijection_name)

        # Default seeded RNG for reproducibility unless caller supplies one.
        self.rng = rng or random.Random(123)

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state to an initial configuration."""
        raise NotImplementedError

    @abstractmethod
    def update(self, feedback: StepFeedback) -> None:
        """
        Update internal statistics with one environment feedback object.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_target(self) -> List[int]:
        """
        Return a target team (list of worker IDs) of size m.
        The environment will attempt to move from current active_set to this target.
        """
        raise NotImplementedError

    def _normalize_target(self, target: Sequence[int]) -> List[int]:
        target_set = sorted(set(int(x) for x in target))
        if len(target_set) != self.m:
            raise ValueError(
                f"compute_target() must return exactly {self.m} distinct worker IDs."
            )
        return target_set

    def build_proposed_replacements(
        self,
        active: Sequence[int],
        target: Sequence[int],
        env,
    ) -> List[Tuple[int, int]]:
        return self._bijection_fn(active, target, rng=self.rng)

    def _propose_replacements_from_target(
        self,
        env,
        target: Sequence[int],
    ) -> List[Tuple[int, int]]:
        target_set = self._normalize_target(target)
        active = sorted(set(env.active_set))
        proposed = self.build_proposed_replacements(active, target_set, env)

        feasible: List[Tuple[int, int]] = []
        for pair in proposed:
            if hasattr(env, "can_append_replacement"):
                if env.can_append_replacement(feasible, pair):
                    feasible.append(pair)
                continue

            candidate = feasible + [pair]
            try:
                env.validate_replacements(candidate)
            except ValueError:
                continue
            else:
                feasible.append(pair)

        return feasible

    def act(self, env) -> List[Tuple[int, int]]:
        """
        Propose a set of feasible replacements for the environment.

        Workflow:
          1) compute_target() -> desired worker IDs
          2) propose a bijection between env.active_set and target (via self.bijection)
          3) let the environment reject proposals that conflict with pending replacements.
        """
        target = self.compute_target()
        if not target:
            return []
        return self._propose_replacements_from_target(env, target)


class EmpiricalDelayedActionPolicy(DelayedActionPolicy):
    """Shared empirical-statistics machinery for delayed-action policies."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counts = np.zeros(self.k, dtype=np.int64)
        self.sums = np.zeros(self.k, dtype=np.float64)
        self.t = 0

    def reset(self) -> None:
        self.counts.fill(0)
        self.sums.fill(0.0)
        self.t = 0
        self.reset_policy_state()

    def reset_policy_state(self) -> None:
        """Reset subclass-specific control state."""

    def update(self, feedback: StepFeedback) -> None:
        self.observe_rewards(feedback.individual_rewards)
        self.t += 1
        self.after_feedback(feedback)

    def after_feedback(self, feedback: StepFeedback) -> None:
        """Handle control-flow updates after observing rewards."""

    def observe_rewards(self, individual_rewards: Dict[int, float]) -> None:
        for worker_id, reward in individual_rewards.items():
            idx = int(worker_id) - 1
            if 0 <= idx < self.k:
                self.counts[idx] += 1
                self.sums[idx] += float(reward)

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.k, dtype=np.float64)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means


class StatefulDelayedActionPolicy(EmpiricalDelayedActionPolicy):
    """Shared phase bookkeeping for delayed-action policies with scheduled decisions."""

    def reset_policy_state(self) -> None:
        self.phase = "init"  # init -> ready -> transition -> hold
        self.current_target: Set[int] = set()
        self.reset_control_state()

    def reset_control_state(self) -> None:
        """Reset subclass-specific transition/hold state."""

    @abstractmethod
    def initialize_control(self, active_now: Sequence[int], env) -> None:
        """Initialize control state from the environment's starting workforce."""
        raise NotImplementedError

    @abstractmethod
    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        """Advance control state until a target is ready, or return None to wait."""
        raise NotImplementedError

    @abstractmethod
    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        """Advance hold-phase bookkeeping after a reward observation."""
        raise NotImplementedError

    def after_feedback(self, feedback: StepFeedback) -> None:
        if self.phase == "transition":
            self.phase = "hold"

        if self.phase == "hold":
            self.on_hold_feedback(feedback)

    def act(self, env) -> List[Tuple[int, int]]:
        active_now = sorted(env.active_set)

        if self.phase == "init":
            self.initialize_control(active_now, env)

        target = self.plan_next_target(active_now, env)
        if target is None:
            return []

        normalized_target = self._normalize_target(target)
        self.current_target = set(normalized_target)
        self.phase = "transition"
        return self._propose_replacements_from_target(env, normalized_target)


class EpsilonGreedyHiringPolicy(EmpiricalDelayedActionPolicy):
    """
    Epsilon-greedy for m-of-k selection with decaying epsilon.

    With prob epsilon_t: pick random target team of size m.
    Else: pick top-m by empirical means.

    Parameters
    ----------
    k : int
    m : int
    epsilon : float
        Initial epsilon (epsilon_0).
    schedule : str
        "inverse_sqrt" or "exponential".
    epsilon_min : float
        Lower bound on epsilon_t.
    decay : float
        Exponential decay factor (used only if schedule="exponential").
    rng : random.Random, optional
    """

    def __init__(
        self,
        k: int,
        m: int,
        epsilon: float = 0.25,
        bijection_name: str = 'random',
        *,
        schedule: str = "inverse_sqrt",
        epsilon_min: float = 0,
        decay: float = 0.999,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(k, m, bijection_name=bijection_name, rng=rng)

        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0,1].")
        if not (0.0 <= epsilon_min <= 1.0):
            raise ValueError("epsilon_min must be in [0,1].")
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0,1).")

        self.epsilon0 = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.schedule = schedule.lower().strip()
        self.decay = float(decay)
        self.decision_round = 0
        self.reset()

    def reset_policy_state(self) -> None:
        self.decision_round = 0

    def current_epsilon(self) -> float:
        t = max(1, self.decision_round)

        if self.schedule in {"inverse_sqrt", "inv_sqrt", "sqrt"}:
            eps = self.epsilon0 / math.sqrt(t)
        elif self.schedule in {"exponential", "exp"}:
            eps = self.epsilon0 * (self.decay ** (t - 1))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return float(max(self.epsilon_min, min(1.0, eps)))

    def compute_target(self) -> List[int]:
        self.decision_round += 1
        eps_t = self.current_epsilon()

        if self.rng.random() < eps_t:
            return self.rng.sample(list(range(1, self.k + 1)), self.m)

        means = self.empirical_means()
        means[self.counts == 0] = float("inf")

        # Random tie-breaking via permutation
        perm = list(range(self.k))
        self.rng.shuffle(perm)
        perm_means = [(means[i], i) for i in perm]
        perm_means.sort(key=lambda x: x[0], reverse=True)

        top_idx = [i for _, i in perm_means[: self.m]]
        return [i + 1 for i in top_idx]


class OMM(EmpiricalDelayedActionPolicy):
    """
    OMM (Optimistic Matroid Maximization) specialized to m-of-k semi-bandits.

    Following Algorithm 2 of Kveton et al. (2014), this baseline:
      - Computes U_i(t) = mean_i + sqrt(alpha * log(t) / n_i)
      - If n_i == 0, set UCB_i = +inf (forces initial exploration)
      - Chooses the top-m workers greedily by those optimistic scores.

    Parameters
    ----------
    k : int
    m : int
    alpha : float
        Exploration strength. The paper uses alpha=2, yielding sqrt(2 log t / n_i).
    rng : random.Random, optional
        Used only for tie-breaking.
    """

    def __init__(
        self,
        k: int,
        m: int,
        alpha: float = 2.0,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(k, m, bijection_name=bijection_name, rng=rng)

        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)
        self.decision_round = 0
        self.reset()

    def reset_policy_state(self) -> None:
        self.decision_round = 0

    def ucb_scores(self) -> np.ndarray:
        means = self.empirical_means()
        scores = np.empty(self.k, dtype=np.float64)

        logt = math.log(max(self.decision_round, 2))
        for i in range(self.k):
            n = self.counts[i]
            if n == 0:
                scores[i] = float("inf")
            else:
                scores[i] = means[i] + math.sqrt(self.alpha * logt / n)

        return scores

    def compute_target(self) -> List[int]:
        self.decision_round += 1
        scores = self.ucb_scores()

        # Random tie-breaking: sort by score then by a shuffled permutation
        perm = list(range(self.k))
        self.rng.shuffle(perm)
        perm_scores = [(scores[i], i) for i in perm]
        perm_scores.sort(key=lambda x: x[0], reverse=True)

        top_idx = [i for _, i in perm_scores[: self.m]]
        return [i + 1 for i in top_idx]


class Threshold(EmpiricalDelayedActionPolicy):
    """
    Threshold policy for delayed hiring.

    The policy keeps the current workforce until one or more active workers with
    observations have empirical mean below a fixed threshold. Those workers are
    replaced by the highest-scoring available workers, where untried workers are
    treated as having empirical mean +inf.
    """

    def __init__(
        self,
        k: int,
        m: int,
        threshold: float = 1,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)
        self.threshold = float(threshold)
        self._active_snapshot: List[int] = []
        self.reset()

    def reset_policy_state(self) -> None:
        self._active_snapshot = []

    def act(self, env) -> List[Tuple[int, int]]:
        self._active_snapshot = sorted(int(worker_id) for worker_id in env.active_set)
        return super().act(env)

    def _available_worker_scores(self, active: Sequence[int]) -> List[int]:
        means = self.empirical_means()
        scores = means.copy()
        scores[self.counts == 0] = float("inf")

        perm = list(range(self.k))
        self.rng.shuffle(perm)
        perm_means = [
            (scores[i], i)
            for i in perm
            if (i + 1) not in active
        ]
        perm_means.sort(key=lambda x: x[0], reverse=True)
        top_idx = [i for _, i in perm_means]
        return [i + 1 for i in top_idx]

    def compute_target(self) -> List[int]:
        active = list(self._active_snapshot)
        if len(active) != self.m:
            raise RuntimeError("Threshold.act() must be called with the current environment.")

        means = self.empirical_means()
        below_threshold = [
            worker_id
            for worker_id in active
            if self.counts[worker_id - 1] > 0 and means[worker_id - 1] < self.threshold
        ]
        if not below_threshold:
            return active

        best_available = self._available_worker_scores(active)
        target = set(active)
        for remove_id, add_id in zip(below_threshold, best_available):
            target.discard(remove_id)
            target.add(add_id)

        return sorted(target)


class SemiAnnualReview(StatefulDelayedActionPolicy):
    """
    Periodic review policy that reselects the workforce on a fixed review cadence.

    The policy holds the current workforce between reviews. At each review time it
    selects the top-m workers by empirical mean, treating untried workers as having
    score +inf so they are explored when available.
    """

    def __init__(
        self,
        k: int,
        m: int,
        review_interval: int = 6 * 30 * 24,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)
        if review_interval <= 0:
            raise ValueError("review_interval must be positive.")
        self.review_interval = int(review_interval)
        self.next_review_time = self.review_interval
        self.reset()

    def reset_control_state(self) -> None:
        self.next_review_time = self.review_interval

    def initialize_control(self, active_now: Sequence[int], env) -> None:
        self.current_target = set(active_now)
        self.phase = "hold"

    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        if self.phase != "ready":
            return None
        return self.compute_target()

    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        if self.t >= self.next_review_time:
            self.next_review_time += self.review_interval
            self.phase = "ready"

    def compute_target(self) -> List[int]:
        means = self.empirical_means()
        scores = means.copy()
        scores[self.counts == 0] = float("inf")

        perm = list(range(self.k))
        self.rng.shuffle(perm)
        ranked = [(scores[i], i) for i in perm]
        ranked.sort(key=lambda x: x[0], reverse=True)
        top_idx = [i for _, i in ranked[: self.m]]
        return [i + 1 for i in top_idx]


class WorkTrial(StatefulDelayedActionPolicy):
    """
    Work-trial policy with an initial short trial and two fixed review blocks.

    Stage 1:
      - Give every worker a short trial of ``trial_periods`` active periods.
    Stage 2:
      - Select the top-2m workers by empirical mean.
      - Hold the first m for ``rotation_periods`` periods.
      - Then hold the second m for ``rotation_periods`` periods.
    Stage 3:
      - Permanently retain the empirically best m workers.
    """

    def __init__(
        self,
        k: int,
        m: int,
        trial_periods: int = 1,
        rotation_periods: int = 90 * 24,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)
        if trial_periods <= 0:
            raise ValueError("trial_periods must be positive.")
        if rotation_periods <= 0:
            raise ValueError("rotation_periods must be positive.")
        self.trial_periods = int(trial_periods)
        self.rotation_periods = int(rotation_periods)
        self.stage = "trial"
        self.required_trial_workers: Set[int] = set()
        self.trial_baselines: Dict[int, int] = {}
        self.shortlist: List[int] = []
        self.block_elapsed = 0
        self._active_snapshot: List[int] = []
        self.reset()

    def reset_control_state(self) -> None:
        self.stage = "trial"
        self.required_trial_workers = set()
        self.trial_baselines = {}
        self.shortlist = []
        self.block_elapsed = 0
        self._active_snapshot = []

    def initialize_control(self, active_now: Sequence[int], env) -> None:
        self._active_snapshot = list(active_now)
        self.current_target = set(active_now)
        self.required_trial_workers = set(active_now)
        self.trial_baselines = {
            worker_id: int(self.counts[worker_id - 1])
            for worker_id in active_now
        }
        self.stage = "trial"
        self.phase = "hold"

    def _rank_workers(self, *, treat_untried_as_infinite: bool = False) -> List[int]:
        means = self.empirical_means()
        scores = means.copy()
        if treat_untried_as_infinite:
            scores[self.counts == 0] = float("inf")

        perm = list(range(self.k))
        self.rng.shuffle(perm)
        ranked = [(scores[i], i) for i in perm]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [i + 1 for _, i in ranked]

    def _next_trial_target(self, active_now: Sequence[int]) -> List[int]:
        remaining_untried = [
            worker_id
            for worker_id in range(1, self.k + 1)
            if self.counts[worker_id - 1] == 0 and worker_id not in active_now
        ]

        if len(remaining_untried) >= self.m:
            target = remaining_untried[: self.m]
        else:
            fillers = [worker_id for worker_id in active_now if worker_id not in remaining_untried]
            target = remaining_untried + fillers[: self.m - len(remaining_untried)]

        self.required_trial_workers = {
            worker_id
            for worker_id in target
            if self.counts[worker_id - 1] == 0
        }
        self.trial_baselines = {
            worker_id: int(self.counts[worker_id - 1])
            for worker_id in target
        }
        return sorted(target)

    def _set_shortlist(self) -> None:
        ranked = self._rank_workers()
        shortlist_size = min(self.k, 2 * self.m)
        self.shortlist = ranked[:shortlist_size]

    def _first_block_target(self) -> List[int]:
        return list(self.shortlist[: self.m])

    def _second_block_target(self) -> List[int]:
        second = list(self.shortlist[self.m : self.m + self.m])
        if len(second) < self.m:
            for worker_id in self.shortlist:
                if worker_id not in second:
                    second.append(worker_id)
                if len(second) == self.m:
                    break
        return second

    def _final_target(self) -> List[int]:
        return self._rank_workers()[: self.m]

    def compute_target(self) -> List[int]:
        if self.stage == "trial":
            return self._next_trial_target(self._active_snapshot)
        if self.stage == "first_block":
            self.block_elapsed = 0
            return self._first_block_target()
        if self.stage == "second_block":
            self.block_elapsed = 0
            return self._second_block_target()
        if self.stage == "final":
            return self._final_target()
        return list(self._active_snapshot)

    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        if self.phase != "ready":
            return None
        self._active_snapshot = list(active_now)
        return self.compute_target()

    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        if self.stage == "trial":
            if all(
                int(self.counts[worker_id - 1]) - self.trial_baselines.get(worker_id, 0) >= self.trial_periods
                for worker_id in self.required_trial_workers
            ):
                remaining_untried = any(self.counts[i] == 0 for i in range(self.k))
                if remaining_untried:
                    self.phase = "ready"
                else:
                    self._set_shortlist()
                    self.stage = "first_block"
                    self.phase = "ready"
            return

        if self.stage in {"first_block", "second_block"}:
            self.block_elapsed += 1
            if self.block_elapsed >= self.rotation_periods:
                if self.stage == "first_block":
                    self.stage = "second_block"
                    self.phase = "ready"
                else:
                    self.stage = "final"
                    self.phase = "ready"


@dataclass(frozen=True)
class AHTConfig:
    """
    Configuration for the block-allocation algorithm of Agrawal, Hegde, and Teneketzis (1990).
    """
    k: int
    m: int
    delta: float
    ucb_coef: float = 2.0


class AgrawalHegdeTeneketzisPolicy(StatefulDelayedActionPolicy):
    """
    Block-allocation policy for multiple plays with switching costs (Agrawal, Hegde, Teneketzis, 1990).

    Notes on inheritance:
    - Inherits validation, empirical statistics, and the transition/hold lifecycle from
      StatefulDelayedActionPolicy.
    - Supplies the AHT-specific schedule and block-length logic through hooks.
    """

    def __init__(
        self,
        *,
        k: int,
        m: int,
        bijection_name: str = 'random',
        rng: Optional[random.Random] = None,
        delta: Optional[float] = None,
        ucb_coef: float = 2.0,
    ) -> None:
        super().__init__(k, m, bijection_name=bijection_name, rng=rng)

        if ucb_coef <= 0:
            raise ValueError("ucb_coef must be > 0.")

        if delta is None:
            delta = 1.0 / (2.0 * m * k)
        if not (0.0 < delta < 1.0 / k):
            raise ValueError("delta must satisfy 0 < delta < 1/k.")

        self.cfg = AHTConfig(
            k=self.k,
            m=self.m,
            delta=float(delta),
            ucb_coef=float(ucb_coef),
        )

        # Block scheduling (frames f and blocks i)
        self.frame_f: int = 0
        self.block_i: int = 0  # 0-based index within the current frame
        self.blocks_in_frame: int = 0
        self.block_len: int = 0
        self.block_remaining: int = 0

        self.reset()

    # ----------------------------
    # StatefulDelayedActionPolicy hooks
    # ----------------------------

    def reset_control_state(self) -> None:
        self.frame_f = 0
        self.block_i = 0
        self.blocks_in_frame = 0
        self.block_len = 0
        self.block_remaining = 0

    def compute_target(self) -> List[int]:
        """
        For AHT, the target is the set we are currently trying to realize/hold.
        This is mainly for introspection/testing; act() is the real control logic.
        """
        return sorted(self.current_target)

    def initialize_control(self, active_now: Sequence[int], env) -> None:
        self.current_target = set(active_now)
        self._initialize_schedule()
        self.phase = "ready"

    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        if self.phase == "ready":
            desired = self._choose_set_at_comparison_instant()
            self.block_remaining = self.block_len
            return sorted(desired)

        return None

    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        self.block_remaining -= 1
        if self.block_remaining <= 0:
            self._advance_block()
            self.phase = "ready"

    # ----------------------------
    # Estimation primitives
    # ----------------------------

    def _empirical_means(self) -> np.ndarray:
        means = np.zeros(self.k, dtype=np.float64)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def _ucb_values(self) -> np.ndarray:
        """
        Hoeffding-style UCB:
            U_j(t) = mean_j(t) + sqrt(ucb_coef * log(max(2, t)) / T_j(t)),
            with U_j(t) = +inf when T_j(t) = 0.
        """
        k, m = self.k, self.m
        means = self._empirical_means()
        ucb = np.empty(k, dtype=np.float64)

        t_for_log = max(1, self.t)
        log_term = math.log(max(2, t_for_log))

        for i in range(k):
            n = int(self.counts[i])
            if n == 0:
                ucb[i] = float("inf")
            else:
                rad = math.sqrt(self.cfg.ucb_coef * log_term / float(n))
                ucb[i] = means[i] + rad

        return ucb

    # ----------------------------
    # Scheduling (frames/blocks)
    # ----------------------------

    def _initialize_schedule(self) -> None:
        self.frame_f = 0
        self.block_i = 0
        self._set_frame_params(self.frame_f)

    def _set_frame_params(self, f: int) -> None:
        p = self.k
        if f == 0:
            self.block_len = self.m
            self.blocks_in_frame = p
        else:
            self.block_len = f
            num = (1 << (f * f)) - (1 << ((f - 1) * (f - 1)))
            q = int(math.ceil(num / float(f)))
            self.blocks_in_frame = q * p

    def _advance_block(self) -> None:
        self.block_i += 1
        if self.block_i >= self.blocks_in_frame:
            self.frame_f += 1
            self.block_i = 0
            self._set_frame_params(self.frame_f)

    # ----------------------------
    # AHT comparison-instant rule
    # ----------------------------

    def _choose_set_at_comparison_instant(self) -> Set[int]:
        p = self.k
        j = (self.block_i % p) + 1

        means = self._empirical_means()
        ucb = self._ucb_values()

        leaders = self._compute_leaders(means)
        leaders_list = sorted(leaders, key=lambda a: float(means[a - 1]), reverse=True)

        if len(leaders_list) < self.m:
            leaders_list = self._top_by_value(means, self.m)
            leaders = set(leaders_list)

        if j in leaders:
            return set(leaders_list[: self.m])

        if all(ucb[j - 1] < means[k - 1] for k in leaders):
            return set(leaders_list[: self.m])

        keep = leaders_list[: max(0, self.m - 1)]
        return set(keep + [j])

    def _compute_leaders(self, means: np.ndarray) -> Set[int]:
        t = max(1, self.t)
        thresh = self.cfg.delta * float(t)

        well = [i for i in range(self.k) if float(self.counts[i]) >= thresh]
        if not well:
            return set()

        self.rng.shuffle(well)
        well.sort(key=lambda i: float(means[i]), reverse=True)
        top = well[: self.m]
        return {i + 1 for i in top}

    def _top_by_value(self, values: np.ndarray, n: int) -> List[int]:
        idxs = list(range(self.k))
        self.rng.shuffle(idxs)
        idxs.sort(key=lambda i: float(values[i]), reverse=True)
        return [i + 1 for i in idxs[:n]]
