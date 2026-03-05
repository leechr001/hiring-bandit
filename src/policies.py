from __future__ import annotations

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
      - update(individual_rewards)
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
        self.bijection = make_bijection(bijection_name)

        # Default seeded RNG for reproducibility unless caller supplies one.
        self.rng = rng or random.Random(123)

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state to an initial configuration."""
        raise NotImplementedError

    @abstractmethod
    def update(self, individual_rewards: Dict[int, float]) -> None:
        """
        Update internal statistics with semi-bandit feedback:
        individual_rewards maps worker_id (1..k) -> realized reward.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_target(self) -> List[int]:
        """
        Return a target team (list of worker IDs) of size m.
        The environment will attempt to move from current active_set to this target.
        """
        raise NotImplementedError

    def act(self, env) -> List[Tuple[int, int]]:
        """
        Propose a set of feasible replacements for the environment.

        Workflow:
          1) compute_target() -> desired worker IDs
          2) propose a bijection between env.active_set and target (via self.bijection)
          3) greedily keep a maximal feasible subset under env.validate_replacements
        """
        target = self.compute_target()
        if not target:
            return []

        # Normalize target: unique, sorted, valid IDs.
        # (If a subclass returns duplicates, we clean it up here.)
        target_set = sorted(set(int(x) for x in target))
        if len(target_set) == 0:
            return []

        active = sorted(env.active_set)

        # Initial proposal: may fail if sizes mismatch or other constraints exist.
        try:
            proposed: List[Tuple[int, int]] = self.bijection(active, target_set)
        except (ValueError, TypeError):
            return []

        # Greedily keep only the subset that remains valid.
        feasible: List[Tuple[int, int]] = []
        for pair in proposed:
            candidate = feasible + [pair]
            try:
                env.validate_replacements(candidate)
            except Exception:
                # If you have a specific exception type for invalid replacements,
                # replace `Exception` with that to avoid masking unrelated bugs.
                continue
            feasible.append(pair)

        return feasible


class EpsilonGreedyHiringPolicy(DelayedActionPolicy):
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
        epsilon_min: float = 0.01,
        decay: float = 0.999,
        rng: Optional[random.Random] = None,
        # keep compatibility with DelayedActionPolicy signature if needed:
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)

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

        self.counts = np.zeros(self.k, dtype=np.int64)
        self.sums = np.zeros(self.k, dtype=np.float64)
        self.t = 0  # number of decision rounds taken

    def reset(self) -> None:
        self.counts.fill(0)
        self.sums.fill(0.0)
        self.t = 0

    def update(self, individual_rewards: Dict[int, float]) -> None:
        for worker_id, r in individual_rewards.items():
            idx = int(worker_id) - 1
            if 0 <= idx < self.k:
                self.counts[idx] += 1
                self.sums[idx] += float(r)

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.k, dtype=np.float64)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def current_epsilon(self) -> float:
        t = max(1, self.t)

        if self.schedule in {"inverse_sqrt", "inv_sqrt", "sqrt"}:
            eps = self.epsilon0 / math.sqrt(t)
        elif self.schedule in {"exponential", "exp"}:
            eps = self.epsilon0 * (self.decay ** (t - 1))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return float(max(self.epsilon_min, min(1.0, eps)))

    def compute_target(self) -> List[int]:
        self.t += 1
        eps_t = self.current_epsilon()

        if self.rng.random() < eps_t:
            return self.rng.sample(list(range(1, self.k + 1)), self.m)

        means = self.empirical_means()

        # Random tie-breaking via permutation
        perm = list(range(self.k))
        self.rng.shuffle(perm)
        perm_means = [(means[i], i) for i in perm]
        perm_means.sort(key=lambda x: x[0], reverse=True)

        top_idx = [i for _, i in perm_means[: self.m]]
        return [i + 1 for i in top_idx]


class VanillaUCBHiringPolicy(DelayedActionPolicy):
    """
    Vanilla UCB for m-of-k semi-bandits.

    Each period:
      - Compute UCB_i(t) = mean_i + sqrt(alpha * log(t) / n_i)
      - If n_i == 0, set UCB_i = +inf (forces initial exploration)
      - Choose top-m by UCB.

    Parameters
    ----------
    k : int
    m : int
    alpha : float
        Exploration strength. alpha=2 is a common safe default.
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
        # keep compatibility with DelayedActionPolicy signature if needed:
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)

        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        self.alpha = float(alpha)

        self.counts = np.zeros(self.k, dtype=np.int64)
        self.sums = np.zeros(self.k, dtype=np.float64)
        self.t = 0  # number of decision rounds taken

    def reset(self) -> None:
        self.counts.fill(0)
        self.sums.fill(0.0)
        self.t = 0

    def update(self, individual_rewards: Dict[int, float]) -> None:
        for worker_id, r in individual_rewards.items():
            idx = int(worker_id) - 1
            if 0 <= idx < self.k:
                self.counts[idx] += 1
                self.sums[idx] += float(r)

    def empirical_means(self) -> np.ndarray:
        means = np.zeros(self.k, dtype=np.float64)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means

    def ucb_scores(self) -> np.ndarray:
        means = self.empirical_means()
        scores = np.empty(self.k, dtype=np.float64)

        logt = math.log(max(self.t, 2))
        for i in range(self.k):
            n = self.counts[i]
            if n == 0:
                scores[i] = float("inf")
            else:
                scores[i] = means[i] + math.sqrt(self.alpha * logt / n)

        return scores

    def compute_target(self) -> List[int]:
        self.t += 1
        scores = self.ucb_scores()

        # Random tie-breaking: sort by score then by a shuffled permutation
        perm = list(range(self.k))
        self.rng.shuffle(perm)
        perm_scores = [(scores[i], i) for i in perm]
        perm_scores.sort(key=lambda x: x[0], reverse=True)

        top_idx = [i for _, i in perm_scores[: self.m]]
        return [i + 1 for i in top_idx]


@dataclass(frozen=True)
class AHTConfig:
    """
    Configuration for the block-allocation algorithm of Agrawal, Hegde, and Teneketzis (1990).
    """
    k: int
    m: int
    delta: float
    ucb_coef: float = 2.0
    init_pulls: int = 0


class AgrawalHegdeTeneketzisPolicy(DelayedActionPolicy):
    """
    Block-allocation policy for multiple plays with switching costs (Agrawal, Hegde, Teneketzis, 1990).

    Notes on inheritance:
    - Inherits k/m/rng validation and storage from DelayedActionPolicy.
    - Overrides act() because AHT restricts switching to block boundaries and has a transition phase.
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
        init_pulls: Optional[int] = None,
        # keep DelayedActionPolicy signature compatibility (unused here):
        c: float = 0.0,
        omega_max: int = 0,
    ) -> None:
        super().__init__(k, m, c=c, bijection_name=bijection_name, omega_max=omega_max, rng=rng)

        if ucb_coef <= 0:
            raise ValueError("ucb_coef must be > 0.")

        if delta is None:
            delta = 1.0 / (2.0 * k)
        if not (0.0 < delta < 1.0 / k):
            raise ValueError("delta must satisfy 0 < delta < 1/k.")

        if init_pulls is None:
            init_pulls = m
        if init_pulls < 0:
            raise ValueError("init_pulls must be >= 0.")

        self.cfg = AHTConfig(
            k=self.k,
            m=self.m,
            delta=float(delta),
            ucb_coef=float(ucb_coef),
            init_pulls=int(init_pulls),
        )

        # Empirical stats (semi-bandit feedback)
        self.counts = np.zeros(self.k, dtype=np.int64)
        self.sums = np.zeros(self.k, dtype=np.float64)

        # Period counter (stages in the paper)
        self.t = 0

        # Control state
        self.phase = "init"  # init -> warmup -> decide -> transition -> block
        self.current_target: Set[int] = set()

        # Block scheduling (frames f and blocks i)
        self.frame_f: int = 0
        self.block_i: int = 0  # 0-based index within the current frame
        self.blocks_in_frame: int = 0
        self.block_len: int = 0
        self.block_remaining: int = 0

        # Warmup scheduling
        self._warmup_queue: List[int] = []

    # ----------------------------
    # Required DelayedActionPolicy API
    # ----------------------------

    def reset(self) -> None:
        self.counts.fill(0)
        self.sums.fill(0.0)
        self.t = 0

        self.phase = "init"
        self.current_target = set()

        self.frame_f = 0
        self.block_i = 0
        self.blocks_in_frame = 0
        self.block_len = 0
        self.block_remaining = 0

        self._warmup_queue = []

    def update(self, individual_rewards: Dict[int, float]) -> None:
        for worker_id, r in individual_rewards.items():
            idx = int(worker_id) - 1
            if 0 <= idx < self.k:
                self.counts[idx] += 1
                self.sums[idx] += float(r)

        self.t += 1

        if self.phase == "block":
            self.block_remaining -= 1
            if self.block_remaining <= 0:
                self._advance_block()
                self.phase = "decide"

        # transition/warmup accounting is driven by act() + environment state; no-op here.

    def compute_target(self) -> List[int]:
        """
        For AHT, the target is the set we are currently trying to realize/hold.
        This is mainly for introspection/testing; act() is the real control logic.
        """
        return sorted(self.current_target)

    # ----------------------------
    # AHT-specific act() (must override base)
    # ----------------------------

    def act(self, env) -> List[Tuple[int, int]]:
        active_now = sorted(list(env.active_set))

        if self.phase == "init":
            self.current_target = set(active_now)
            self._initialize_schedule()
            self._initialize_warmup()
            self.phase = "warmup" if self.cfg.init_pulls > 0 else "decide"

        if self.phase == "warmup":
            desired = self._warmup_target()
            if desired is None:
                self.phase = "decide"
            else:
                reps = self.bijection(active_now, sorted(desired))
                self.current_target = set(desired)
                self.phase = "transition"
                self.block_remaining = 1  # warmup plays exactly one period once active
                return reps

        if self.phase == "decide":
            desired = self._choose_set_at_comparison_instant()
            reps = self.bijection(active_now, sorted(desired))
            self.current_target = set(desired)

            self.phase = "transition"
            self.block_remaining = self.block_len
            return reps

        if self.phase == "transition":
            # Wait until the chosen set is fully active, then enter the block phase.
            if set(active_now) == self.current_target:
                self.phase = "block"
            return []

        if self.phase == "block":
            # Hold workforce fixed for the remainder of the current block.
            return []

        return []

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
            U_j(t) = mean_j(t) + sqrt(ucb_coef * log(max(2, k*m*t)) / T_j(t)),
            with U_j(t) = +inf when T_j(t) = 0.
        """
        k, m = self.k, self.m
        means = self._empirical_means()
        ucb = np.empty(k, dtype=np.float64)

        t_for_log = max(1, self.t)
        log_term = math.log(max(2, k * m * t_for_log))

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
            self.block_len = 1
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
    # Warm start
    # ----------------------------

    def _initialize_warmup(self) -> None:
        self._warmup_queue = list(range(1, self.k + 1))
        self.rng.shuffle(self._warmup_queue)

    def _warmup_target(self) -> Optional[Set[int]]:
        if self.cfg.init_pulls <= 0:
            return None
        if int(np.min(self.counts)) >= self.cfg.init_pulls:
            return None

        # Choose m arms with the smallest counts (random tie-breaking).
        idxs = list(range(self.k))
        self.rng.shuffle(idxs)
        idxs.sort(key=lambda i: int(self.counts[i]))
        chosen = idxs[: self.m]
        return {i + 1 for i in chosen}

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
