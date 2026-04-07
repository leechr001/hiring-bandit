from bandit_environment import StepFeedback
from policies import StatefulDelayedActionPolicy

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

@dataclass
class OptimisticHireConfig:
    k: int
    m: int
    gamma: float
    horizon: Optional[int] = None
    ucb_coef: float = 2.0


class OptimisticHire(StatefulDelayedActionPolicy):
    """
    Implements Algorithm 1 (Optimistic-Hire).

    Assumptions about env:
    - env.active_set: current active set as an iterable of 1-indexed worker IDs.
    - env.step(replacements) returns (obs, total_reward, cost, feedback)
    - feedback is a StepFeedback object with observed rewards and the realized active set.

    Iteration structure:
    - At the start of each iteration, compute a target workforce U_ell (top-m by UCB, respecting cost constraint),
      compute a buffer threshold N(ell), and identify i_min in U_ell with minimal count
      at iteration start c_ell.
    - Initiate one-for-one replacements to move toward U_ell.
    - Transition phase: do not initiate further replacements until env.active_set == U_ell.
    - Hold phase: keep the workforce fixed until
          T_{i_min}(t) - T_{i_min}(c_ell) >= N(ell),
      where T_i is the cumulative number of observed outcomes for worker i.

    The policy is anytime (does not require horizon T).
    """

    def __init__(
        self,
        *,
        k: int,
        m: int,
        gamma: float,
        horizon: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(k=k, m=m, rng=rng)
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be > 0 when provided.")

        self.cfg = OptimisticHireConfig(k=k, m=m, gamma=gamma, horizon=horizon)
        self.prev_target: Set[int] = set()
        self.iterations: int = 0

        # Buffer control state (count-based stopping rule)
        self.i_min: Optional[int] = None                 # 1-indexed worker ID
        self.i_min_count_at_c: Optional[int] = None      # T_{i_min}(c_ell)
        self.buffer_threshold_N: int = 0                  # N(ell)
        self.reset()

    # ----------------------------
    # StatefulDelayedActionPolicy hooks
    # ----------------------------

    def reset_control_state(self) -> None:
        self.prev_target = set()
        self.iterations = 0
        self.i_min = None
        self.i_min_count_at_c = None
        self.buffer_threshold_N = 0

    def initialize_control(self, active_now: Sequence[int], env) -> None:
        self.prev_target = set(active_now)
        self.current_target = set(active_now)
        self.phase = "ready"

    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        if self.phase != "ready":
            return None

        self.iterations += 1

        new_target = self.compute_target()
        self.buffer_threshold_N = self.compute_buffer_length(new_target)
        self.i_min, self.i_min_count_at_c = self._compute_i_min_and_baseline(new_target)
        return sorted(new_target)

    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        if self.i_min is None or self.i_min_count_at_c is None:
            self.prev_target = set(self.current_target)
            self.phase = "ready"
            return

        cur = int(self.counts[self.i_min - 1])
        if cur - int(self.i_min_count_at_c) >= int(self.buffer_threshold_N):
            self.prev_target = set(self.current_target)
            self.phase = "ready"
    
    def _confidence_rad(self, i:int):
        """Return the confidence radius for arm i"""
        k, m = self.cfg.k, self.cfg.m

        t_for_log = max(1, self.t)
        log_term = math.log(max(2, k * m * t_for_log))

        return math.sqrt(self.cfg.ucb_coef * log_term / self.counts[i])

    def ucb_values(self) -> np.ndarray:
        """
        Compute UCB indices.

            UCB_i(t) = mean_i(t) + sqrt( ucb_coef * log(k*m*t) / T_i(t) ),
            with UCB_i(t) = +inf when T_i(t) = 0.
        """
        k = self.cfg.k
        means = self.empirical_means()
        ucb = np.empty(k, dtype=np.float64)

        for i in range(k):
            if self.counts[i] == 0:
                ucb[i] = float("inf")
            else:
                rad = self._confidence_rad(i)
                ucb[i] = means[i] + rad

        return ucb
    
    def lcb_values(self) -> np.ndarray:
        """
        Compute LCB indices.

            LCB_i(t) = mean_i(t) - sqrt( ucb_coef * log(k*m*t) / T_i(t) ),
            with LCB_i(t) = +inf when T_i(t) = 0.
        """

        k = self.cfg.k
        means = self.empirical_means()
        lcb = np.empty(k, dtype=np.float64)

        for i in range(k):
            if self.counts[i] == 0:
                lcb[i] = -float("inf")
            else:
                rad = self._confidence_rad(i)
                lcb[i] = means[i] - rad

        return lcb

    # ----------------------------
    # Core algorithm pieces
    # ----------------------------

    def compute_target(self) -> Set[int]:
        """
        Compute the optimistic target workforce U_ell.

        Since the objective is additive, the optimistic solution is the top-m workers by UCB.
        Ties are broken randomly.
        """
        ucb = self.ucb_values()

        idxs = list(range(self.cfg.k))
        self.rng.shuffle(idxs)
        idxs.sort(key=lambda i: ucb[i], reverse=True)

        top = idxs[: self.cfg.m]
        return {i + 1 for i in top}  # 1-indexed worker IDs

    def compute_buffer_length(self, target: Set[int]) -> int:
        """
        Compute N(ell) based on counts at the start of the iteration (c_ell):

            N(ell) := min_{j in U_ell} ceil( 2 * sqrt(gamma * T_j(c_ell)) + gamma ).

        Uses current counts as T_j(c_ell).
        """
        gamma = self.cfg.gamma
        vals: List[float] = []
        for j in target:
            Tj = int(self.counts[j - 1])
            vals.append(2.0 * math.sqrt(gamma * Tj) + gamma)

        if not vals:
            return 1

        return max(1, int(math.ceil(min(vals))))

    def bijection(
        self,
        current: Sequence[int],
        target: Sequence[int],
        *,
        current_period: Optional[int] = None,
        switching_cost: float = 0.0,
    ) -> List[Tuple[int, int]]:
        """
        Construct a rank-matching bijection based on LCB values.

        If a finite horizon is configured, exclude any pair (i, j) for which
        UCB(j) - LCB(i) < c / (T - t), where T is the horizon, t is the current
        period, and c is the switching cost.

        Parameters
        ----------
        current:
            Current active set as 1-indexed worker IDs.
        target:
            Target set as 1-indexed worker IDs.

        Returns
        -------
        List of (remove_id, add_id) replacement pairs.
        """
        cur_set = set(current)
        tar_set = set(target)

        remove = list(cur_set - tar_set)
        add = list(tar_set - cur_set)

        if not remove or not add:
            return []

        # Ensure equal length (defensive; should match in intended use).
        n = min(len(remove), len(add))
        remove = remove[:n]
        add = add[:n]

        lcb = self.lcb_values()  # lcb[i] corresponds to worker (i+1)
        ucb = self.ucb_values()  # ucb[i] corresponds to worker (i+1)

        def lcb_key(worker_id: int) -> Tuple[float, int]:
            # Sort by LCB descending, then by worker_id ascending for stable tie-break.
            return (float(lcb[worker_id - 1]), -worker_id)

        # Descending by LCB; tie-break deterministically.
        remove.sort(key=lcb_key, reverse=True)
        add.sort(key=lcb_key, reverse=True)


        pairs = list(zip(remove, add))

        if self.cfg.horizon is None or current_period is None:
            return pairs

        remaining_periods = int(self.cfg.horizon) - int(current_period)
        if remaining_periods <= 0:
            return []

        threshold = float(switching_cost) / float(remaining_periods)
        admissible: List[Tuple[int, int]] = []
        for i, j in pairs:
            if float(ucb[j - 1]) - float(lcb[i - 1]) >= threshold:
                admissible.append((i, j))

        return admissible

    def _compute_i_min_and_baseline(self, target: Set[int]) -> Tuple[int, int]:
        """
        Identify i_min in the target set and its baseline count at iteration start.

        i_min is any minimizer of T_i(c_ell) over i in U_ell, with random tie-breaking.
        Returns (i_min, T_{i_min}(c_ell)), where i_min is 1-indexed.
        """
        if not target:
            return 1, int(self.counts[0])

        target_list = list(target)
        self.rng.shuffle(target_list)

        i_min = target_list[0]
        min_count = int(self.counts[i_min - 1])

        for j in target_list[1:]:
            cj = int(self.counts[j - 1])
            if cj < min_count:
                i_min = j
                min_count = cj

        return i_min, min_count

    def build_proposed_replacements(
        self,
        active: Sequence[int],
        target: Sequence[int],
        env,
    ) -> List[Tuple[int, int]]:
        return self.bijection(
            active,
            target,
            current_period=env.t,
            switching_cost=env.c,
        )
