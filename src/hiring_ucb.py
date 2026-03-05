from policies import DelayedActionPolicy

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

@dataclass
class HiringUCBConfig:
    k: int
    m: int
    gamma: float
    horizon: Optional[int] = None
    # UCB confidence term uses log(k*m*t) per paper text
    # and UCB_i(0) = +inf initialization.
    # We keep it as bounded-reward Hoeffding-style index.
    # You can adjust this constant if your paper version differs.
    ucb_coef: float = 2.0


class HiringUCBPolicy(DelayedActionPolicy):
    """
    Implements Algorithm 1 (Hiring-UCB).

    Assumptions about env:
    - env.active_set: current active set as an iterable of 1-indexed worker IDs.
    - env.step(replacements) returns (obs, total_reward, cost, info)
    - info includes:
        - "individual_rewards": Dict[int, float]
        - "active_set": iterable of 1-indexed worker IDs

    Iteration structure:
    - At the start of each iteration, compute a target workforce U_ell (top-m by UCB),
      compute a buffer threshold N(ell), and identify i_min in U_ell with minimal count
      at iteration start c_ell.
    - Initiate one-for-one replacements to move toward U_ell.
    - Transition phase: do not initiate further replacements until env.active_set == U_ell.
    - Buffer phase: hold the workforce fixed until
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
        if not (1 <= m < k):
            raise ValueError("Require 1 <= m < k.")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be > 0 when provided.")

        self.cfg = HiringUCBConfig(k=k, m=m, gamma=gamma, horizon=horizon)
        self.rng = rng or random.Random()

        # Empirical stats from observed semi-bandit feedback used to construct index
        self.counts = np.zeros(k, dtype=np.int64)
        self.sums = np.zeros(k, dtype=np.float64)

        # Period counter for log term in UCB
        self.t = 0

        # Iteration state
        self.phase = "init"  # init -> ready -> transition -> buffer
        self.current_target: Set[int] = set()
        self.prev_target: Set[int] = set()
        self.iterations: int = 0

        # Buffer control state (count-based stopping rule)
        self.i_min: Optional[int] = None                 # 1-indexed worker ID
        self.i_min_count_at_c: Optional[int] = None      # T_{i_min}(c_ell)
        self.buffer_threshold_N: int = 0                  # N(ell)

    # ----------------------------
    # Define Control API Methods
    # ----------------------------

    def reset(self):
        """Reset internal state and empirical statistics."""
        self.counts.fill(0)
        self.sums.fill(0.0)
        self.t = 0
        self.phase = "init"
        self.current_target = set()
        self.prev_target = set()
        self.iterations = 0

        self.i_min = None
        self.i_min_count_at_c = None
        self.buffer_threshold_N = 0

    def update(self, individual_rewards: Dict[int, float]):
        """
        Update empirical statistics after a single environment step.

        Parameters
        ----------
        individual_rewards:
            Mapping from 1-indexed worker ID to observed reward for workers that were active
            and produced feedback this period.
        """
        for worker_id, r in individual_rewards.items():
            idx = worker_id - 1
            self.counts[idx] += 1
            self.sums[idx] += float(r)

        self.t += 1

        if self.phase == "buffer":
            # End the buffer phase once i_min has accrued N(ell) additional observations since c_ell.
            if self.i_min is None or self.i_min_count_at_c is None:
                # Defensive: avoid deadlock if state is inconsistent.
                self.prev_target = set(self.current_target)
                self.phase = "ready"
                return

            cur = int(self.counts[self.i_min - 1])
            # T_{i_min}(t) - T_{i_min}(c_ell) >= N(ell)
            if cur - int(self.i_min_count_at_c) >= int(self.buffer_threshold_N):
                self.prev_target = set(self.current_target)
                self.phase = "ready"

    def empirical_means(self) -> np.ndarray:
        """Return the vector of empirical means (0 for unseen workers)."""
        means = np.zeros(self.cfg.k, dtype=np.float64)
        mask = self.counts > 0
        means[mask] = self.sums[mask] / self.counts[mask]
        return means
    
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
        UCB(j) - LCB(i) < c * (T - t), where T is the horizon, t is the current
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
        ucb = self.ucb_values()

        def lcb_key(worker_id: int) -> Tuple[float, int]:
            # Sort by LCB descending, then by worker_id ascending for stable tie-break.
            return (float(lcb[worker_id - 1]), -worker_id)

        # Descending by LCB; tie-break deterministically.
        remove.sort(key=lcb_key, reverse=True)
        add.sort(key=lcb_key, reverse=True)


        pairs = list(zip(remove, add))

        if self.cfg.horizon is None or current_period is None:
            return pairs

        threshold = float(switching_cost) / max(0, self.cfg.horizon - int(current_period))
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

    # ----------------------------
    # Public control API
    # ----------------------------

    def act(self, env) -> List[Tuple[int, int]]:
        """
        Decide which replacements to initiate at the start of the current period.

        Policy rule:
        - Initiate replacements only when starting a new iteration (phase == "ready").
        - During transition and buffer phases, return [].

        Returns
        -------
        A list of (remove_id, add_id) replacement pairs to initiate this period.
        """
        active_now = sorted(list(env.active_set))

        if self.phase == "init":
            # Treat the initial active workforce as U_0.
            self.prev_target = set(active_now)
            self.current_target = set(active_now)
            self.phase = "ready"

        if self.phase == "ready":
            self.iterations += 1

            new_target = self.compute_target()
            N = self.compute_buffer_length(new_target)

            i_min, baseline = self._compute_i_min_and_baseline(new_target)
            self.i_min = i_min
            self.i_min_count_at_c = baseline
            self.buffer_threshold_N = N

            reps = self.bijection(
                active_now,
                sorted(new_target),
                current_period=env.t,
                switching_cost=env.c,
            )

            self.current_target = set(new_target)
            self.phase = "transition"
            return reps

        if self.phase == "transition":
            # Wait until target workforce is fully active.
            if set(active_now) == self.current_target:
                self.phase = "buffer"
            return []

        if self.phase == "buffer":
            # Hold workforce fixed until the buffer stopping rule is met in update().
            return []

        return []
