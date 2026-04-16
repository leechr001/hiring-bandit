from bandit_environment import StepFeedback
from bijections import (
    optimistic_hire_rank_matching_bijection,
    optimistic_hire_switching_threshold,
    random_bijection,
)
from choose_target import ChooseTargetFrontierSizeRecord, choose_target
from policies import StatefulDelayedActionPolicy

import math
import random
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

@dataclass
class OptimisticHireConfig:
    k: int
    m: int
    gamma: float
    horizon: Optional[int] = None
    ucb_coef: float = 1.0
    log_frontier_sizes: bool = False
    switching_mode: str = "adaptive-count"
    pairing_rule: str = "rank-matching"


class OptimisticHire(StatefulDelayedActionPolicy):
    """
    Implements Algorithm 1 (Optimistic-Hire).

    Assumptions about env:
    - env.active_set: current active set as an iterable of 1-indexed worker IDs.
    - env.step(replacements) returns (obs, total_reward, cost, feedback)
    - feedback is a StepFeedback object with observed rewards and the realized active set.

    Iteration structure:
    - At the start of each iteration, compute a target workforce U_ell (top-m by UCB, respecting cost constraint),
      compute a block threshold N(ell), and identify i_min in U_ell with minimal count
      at iteration start c_ell.
    - Initiate one-for-one replacements to move toward U_ell.
    - Block phase: keep the workforce fixed until the realized active workforce
      matches the target and
          T_{i_min}(t) - T_{i_min}(c_ell) >= N(ell),
      where T_i is the cumulative number of observed outcomes for worker i.

    The main paper studies the finite-horizon version with horizon-aware
    screening. The implementation also supports ablation variants with no
    screening, a fully preset calendar-time switching rule, or alternative
    pairing rules.
    """

    def __init__(
        self,
        *,
        k: int,
        m: int,
        gamma: float,
        horizon: Optional[int] = None,
        rng: Optional[random.Random] = None,
        log_frontier_sizes: bool = False,
        switching_mode: str = "adaptive-count",
        pairing_rule: str = "rank-matching",
    ):
        super().__init__(k=k, m=m, rng=rng)
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if horizon is not None and horizon <= 0:
            raise ValueError("horizon must be > 0 when provided.")
        switching_mode_normalized = switching_mode.strip().lower()
        if switching_mode_normalized not in {"adaptive-count", "fixed-calendar"}:
            raise ValueError(
                "switching_mode must be 'adaptive-count' or 'fixed-calendar'."
            )
        if switching_mode_normalized == "fixed-calendar" and horizon is None:
            raise ValueError(
                "fixed-calendar mode requires a finite horizon so the switching "
                "calendar can be precomputed before the run starts."
            )
        pairing_rule_normalized = pairing_rule.strip().lower()
        if pairing_rule_normalized not in {"rank-matching", "random"}:
            raise ValueError(
                "pairing_rule must be 'rank-matching' or 'random'."
            )

        self.cfg = OptimisticHireConfig(
            k=k,
            m=m,
            gamma=gamma,
            horizon=horizon,
            log_frontier_sizes=log_frontier_sizes,
            switching_mode=switching_mode_normalized,
            pairing_rule=pairing_rule_normalized,
        )
        self.prev_target: Set[int] = set()
        self.iterations: int = 0
        self.frontier_size_log: List[ChooseTargetFrontierSizeRecord] = []
        self.last_frontier_size_log: List[ChooseTargetFrontierSizeRecord] = []
        self.selection_runtime_log: List[float] = []
        self.last_selection_runtime: float = 0.0

        # block control state (count-based stopping rule)
        self.i_min: Optional[int] = None                 # 1-indexed worker ID
        self.i_min_count_at_c: Optional[int] = None      # T_{i_min}(c_ell)
        self.block_threshold_N: int = 0                  # N(ell)
        self.switch_time_baseline: Optional[int] = None  # calendar baseline for ablations
        self.fixed_calendar_switch_times: List[int] = []
        self.next_fixed_calendar_idx: int = 0
        self.reset()

    # ----------------------------
    # StatefulDelayedActionPolicy hooks
    # ----------------------------

    def reset_control_state(self) -> None:
        self.prev_target = set()
        self.iterations = 0
        self.i_min = None
        self.i_min_count_at_c = None
        self.block_threshold_N = 0
        self.switch_time_baseline = None
        self.fixed_calendar_switch_times.clear()
        self.next_fixed_calendar_idx = 0
        self.frontier_size_log.clear()
        self.last_frontier_size_log.clear()
        self.selection_runtime_log.clear()
        self.last_selection_runtime = 0.0

    def initialize_control(self, active_now: Sequence[int], env) -> None:
        self.prev_target = set(active_now)
        self.current_target = set(active_now)
        if self.cfg.switching_mode == "fixed-calendar":
            self._initialize_fixed_calendar(start_time=int(env.t))
        self.phase = "ready"

    def plan_next_target(self, active_now: Sequence[int], env) -> Optional[Sequence[int]]:
        if self.phase != "ready":
            return None

        if self.cfg.switching_mode == "fixed-calendar":
            next_switch_time = self._next_fixed_calendar_switch_time()
            if next_switch_time is None or int(env.t) < next_switch_time:
                return None
            self._advance_fixed_calendar()
            self.iterations += 1
            self.i_min = None
            self.i_min_count_at_c = None
            self.block_threshold_N = 0
            self.switch_time_baseline = None
            new_target = self.compute_target(
                active=active_now,
                current_period=env.t,
                switching_cost=env.c,
            )
            return sorted(new_target)

        self.iterations += 1

        new_target = self.compute_target(
            active=active_now,
            current_period=env.t,
            switching_cost=env.c,
        )
        self.block_threshold_N = self.compute_block_length(new_target)
        self.i_min, self.i_min_count_at_c = self._compute_i_min_and_baseline(new_target)
        self.switch_time_baseline = int(self.t)
        return sorted(new_target)

    def on_hold_feedback(self, feedback: StepFeedback) -> None:
        if self.cfg.switching_mode == "fixed-calendar":
            # The non-adaptive ablation follows a review calendar computed
            # before the episode starts. Once a period's feedback arrives, the
            # policy is immediately eligible to check whether the next preset
            # decision time has been reached.
            self.phase = "ready"
            return

        if self.i_min is None or self.i_min_count_at_c is None:
            self.prev_target = set(self.current_target)
            self.phase = "ready"
            return

        cur = int(self.counts[self.i_min - 1])
        if self.cfg.switching_mode == "adaptive-count":
            ready_to_switch = (
                set(feedback.active_set) == set(self.current_target)
                and cur - int(self.i_min_count_at_c) >= int(self.block_threshold_N)
            )
        if ready_to_switch:
            self.prev_target = set(self.current_target)
            self.phase = "ready"
    
    def _confidence_rad(self, i:int):
        """Return the confidence radius for arm i"""

        t_for_log = max(1, self.t)
        log_term = math.log(t_for_log)

        return math.sqrt(self.cfg.ucb_coef * log_term / self.counts[i])

    def ucb_values(self) -> np.ndarray:
        """
        Compute UCB indices.

            UCB_i(t) = mean_i(t) + sqrt( ucb_coef * log(t) / T_i(t) ),
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

            LCB_i(t) = mean_i(t) - sqrt( ucb_coef * log(t) / T_i(t) ),
            with LCB_i(t) = -inf when T_i(t) = 0.
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

    def compute_target(
        self,
        active: Optional[Sequence[int]] = None,
        *,
        current_period: Optional[int] = None,
        switching_cost: float = 0.0,
    ) -> Set[int]:
        """
        Compute the optimistic target workforce U_ell.

        This solves the horizon-aware ChooseTarget subproblem from the paper:
        maximize the sum of UCB scores over the target workforce while only
        allowing the rank-matched replacements to recover the switching cost in
        aggregate over the remaining horizon.
        """
        if active is None:
            active = sorted(self.current_target)

        frontier_size_log: List[ChooseTargetFrontierSizeRecord] | None = None
        if self.cfg.log_frontier_sizes:
            frontier_size_log = []

        start_time = time.perf_counter()
        result = choose_target(
            active_set=active,
            counts=self.counts.tolist(),
            empirical_means=self.empirical_means().tolist(),
            current_period=current_period,
            horizon=self.cfg.horizon,
            switching_cost=switching_cost,
            ucb_coef=self.cfg.ucb_coef,
            time_index=self.t,
            frontier_size_log=frontier_size_log,
        )
        self.last_selection_runtime = time.perf_counter() - start_time
        self.selection_runtime_log.append(self.last_selection_runtime)

        if frontier_size_log is not None:
            annotated_records = [
                replace(record, decision_iteration=self.iterations)
                for record in frontier_size_log
            ]
            self.last_frontier_size_log = annotated_records
            self.frontier_size_log.extend(annotated_records)

        return set(result.target)

    def compute_block_length(self, target: Set[int]) -> int:
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

    def _fixed_calendar_block_length(self, baseline_count: int) -> int:
        """
        Deterministic block length used by the fixed-calendar ablation.

        The calendar is computed before the run starts, so it cannot depend on
        realized counts, realized active sets, or replacement completions. We
        therefore use the same functional form as the adaptive rule but evolve
        it on a deterministic proxy sequence in which the least-observed target
        worker is assumed to accrue one observation per period between review
        epochs.
        """
        gamma = self.cfg.gamma
        return max(1, int(math.ceil(2.0 * math.sqrt(gamma * baseline_count) + gamma)))

    def _initialize_fixed_calendar(self, *, start_time: int) -> None:
        if self.cfg.horizon is None:
            raise RuntimeError(
                "fixed-calendar mode requires a finite horizon to precompute the "
                "switching calendar."
            )

        episode_end_time = int(start_time) + int(self.cfg.horizon) - 1
        switch_times = [int(start_time)]
        baseline_count = 0
        current_time = int(start_time)

        while True:
            block_length = self._fixed_calendar_block_length(baseline_count)
            baseline_count += block_length
            current_time = int(start_time) + baseline_count
            if current_time > episode_end_time:
                break
            switch_times.append(current_time)

        self.fixed_calendar_switch_times = switch_times
        self.next_fixed_calendar_idx = 0

    def _next_fixed_calendar_switch_time(self) -> Optional[int]:
        if self.next_fixed_calendar_idx >= len(self.fixed_calendar_switch_times):
            return None
        return int(self.fixed_calendar_switch_times[self.next_fixed_calendar_idx])

    def _advance_fixed_calendar(self) -> None:
        if self.next_fixed_calendar_idx < len(self.fixed_calendar_switch_times):
            self.next_fixed_calendar_idx += 1

    def construct_bijection(
        self,
        current: Sequence[int],
        target: Sequence[int],
        *,
        current_period: Optional[int] = None,
        switching_cost: float = 0.0,
    ) -> List[Tuple[int, int]]:
        """
        Construct the configured replacement bijection.

        If a finite horizon is configured, return the full replacement set only
        when

            sum_{(i, j)} (UCB(j) - LCB(i)) >= |pi| * c / (T - t),

        where T is the horizon, t is the current period, and c is the switching
        cost. Otherwise return no replacements.

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
        lcb = self.lcb_values()  # lcb[i] corresponds to worker (i+1)
        ucb = self.ucb_values()  # ucb[i] corresponds to worker (i+1)
        if self.cfg.pairing_rule == "rank-matching":
            pairs = optimistic_hire_rank_matching_bijection(
                current,
                target,
                ucb_values=ucb.tolist(),
            )
        else:
            pairs = random_bijection(
                current,
                target,
                rng=self.rng,
            )

        threshold = optimistic_hire_switching_threshold(
            current_period=current_period,
            horizon=self.cfg.horizon,
            switching_cost=switching_cost,
        )
        if threshold is None:
            return pairs
        if math.isinf(threshold):
            return []

        aggregate_slack = 0.0
        for remove_id, add_id in pairs:
            add_ucb = float(ucb[add_id - 1])
            remove_lcb = float(lcb[remove_id - 1])
            if math.isinf(add_ucb) and add_ucb > 0.0:
                return pairs
            if math.isinf(remove_lcb) and remove_lcb < 0.0:
                return pairs
            aggregate_slack += add_ucb - remove_lcb - threshold

        if aggregate_slack >= -1e-12:
            return pairs
        return []

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
        return self.construct_bijection(
            active,
            target,
            current_period=env.t,
            switching_cost=env.c,
        )
