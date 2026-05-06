import os
import json
import random
import sys
import tempfile
import unittest
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

mpl_dir = Path(tempfile.gettempdir()) / "hiringbandit-mpl"
mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

from bijections import (
    delayed_replace_ucb_rank_matching_bijection,
    delayed_replace_ucb_switching_threshold,
)
from bandit_environment import PendingReplacement, StepFeedback, TemporaryHiringBanditEnv
from choose_target import choose_target
from delayed_replace_ucb import DelayedReplaceUCB
from experiments.helpers import _json_safe_metadata
from policies import (
    AdaptedAHTPolicy,
    FixedScheduleGreedy,
    PreScreen,
    StatefulDelayedActionPolicy,
    Threshold,
    WorkTrial,
)
from samplers import (
    make_bernoulli_samplers,
    make_calendar_adversarial_delay,
    make_calendar_delay_sampler,
    make_conditional_samplers,
    make_geometric_delay_sampler,
    make_truncated_normal_samplers,
    make_uniform_delay_sampler,
)
from simulation import (
    _average_regret_results,
    build_planning_horizon_regret_table,
    ExperimentSeries,
    make_delay_sampler_factory,
    make_reward_sampler_factory,
    compute_delayed_replace_ucb_auto_gamma,
    make_policy,
    delayed_replace_ucb_regret_bound,
    simulate,
)


class RegressionTests(unittest.TestCase):
    class _StaticEnv:
        def __init__(self, active_set):
            self.active_set = set(active_set)

        def validate_replacements(self, replacements) -> None:
            seen_remove = set()
            seen_add = set()
            for remove_id, add_id in replacements:
                if remove_id not in self.active_set:
                    raise ValueError("remove worker must be active")
                if add_id in self.active_set:
                    raise ValueError("add worker must be inactive")
                if remove_id in seen_remove or add_id in seen_add:
                    raise ValueError("duplicate worker in replacements")
                seen_remove.add(remove_id)
                seen_add.add(add_id)

    class _PolicyEnv:
        def __init__(self, active_set, *, t: int = 1, c: float = 0.0):
            self.active_set = set(active_set)
            self.t = t
            self.c = c

        def validate_replacements(self, replacements) -> None:
            return None

    class _PendingPolicyEnv(_PolicyEnv):
        def __init__(self, active_set, *, pending_pairs, t: int = 1, c: float = 0.0):
            super().__init__(active_set, t=t, c=c)
            self.pending = [
                PendingReplacement(
                    i=int(remove_id),
                    j=int(add_id),
                    start_time=t,
                    completion_time=t + 2,
                )
                for remove_id, add_id in pending_pairs
            ]
            self.pending_workers = {
                worker_id
                for pending_replacement in self.pending
                for worker_id in (pending_replacement.i, pending_replacement.j)
            }

        def can_append_replacement(self, accepted, pair) -> bool:
            remove_id, add_id = (int(pair[0]), int(pair[1]))
            if remove_id not in self.active_set:
                return False
            if add_id in self.active_set:
                return False
            if remove_id in self.pending_workers or add_id in self.pending_workers:
                return False
            if remove_id in {existing_remove for existing_remove, _ in accepted}:
                return False
            if add_id in {existing_add for _, existing_add in accepted}:
                return False
            return True

    @staticmethod
    def _bruteforce_choose_target_zero_ucb_bonus(
        *,
        active_set,
        empirical_means,
        current_period,
        horizon,
        switching_cost,
    ):
        active = tuple(sorted(active_set))
        ucb_values = [float(x) for x in empirical_means]
        lcb_values = [float(x) for x in empirical_means]
        threshold = delayed_replace_ucb_switching_threshold(
            horizon=horizon,
            current_period=current_period,
            switching_cost=switching_cost,
        )
        if threshold is None:
            threshold = 0.0
        if math.isinf(threshold):
            objective = sum(ucb_values[worker_id - 1] for worker_id in active)
            return set(active), tuple(), objective

        best_target = set(active)
        best_pairs = tuple()
        best_objective = sum(ucb_values[worker_id - 1] for worker_id in active)

        for target in combinations(range(1, len(empirical_means) + 1), len(active)):
            pairs = delayed_replace_ucb_rank_matching_bijection(
                active,
                target,
                ucb_values=ucb_values,
            )
            total_slack = sum(
                ucb_values[add_id - 1] - lcb_values[remove_id - 1] - threshold
                for remove_id, add_id in pairs
            )
            if total_slack < -1e-12:
                continue

            objective = sum(ucb_values[worker_id - 1] for worker_id in target)
            if objective > best_objective + 1e-12:
                best_target = set(target)
                best_pairs = tuple(pairs)
                best_objective = objective

        return best_target, best_pairs, best_objective

    @staticmethod
    def _myopic_greedy_choose_target_zero_ucb_bonus(
        *,
        active_set,
        empirical_means,
        current_period,
        horizon,
        switching_cost,
    ):
        active = set(active_set)
        k = len(empirical_means)
        threshold = delayed_replace_ucb_switching_threshold(
            horizon=horizon,
            current_period=current_period,
            switching_cost=switching_cost,
        )
        if threshold is None:
            threshold = 0.0
        if math.isinf(threshold):
            objective = sum(empirical_means[worker_id - 1] for worker_id in active)
            return active, objective

        while True:
            best_target = None
            best_gain = 0.0

            for remove_id in sorted(active):
                for add_id in range(1, k + 1):
                    if add_id in active:
                        continue

                    candidate_target = sorted((active - {remove_id}) | {add_id})
                    pairs = delayed_replace_ucb_rank_matching_bijection(
                        sorted(active),
                        candidate_target,
                        ucb_values=empirical_means,
                    )
                    if len(pairs) != 1:
                        continue

                    gain = empirical_means[add_id - 1] - empirical_means[remove_id - 1]
                    if gain + 1e-12 < threshold:
                        continue
                    if gain > best_gain + 1e-12:
                        best_target = set(candidate_target)
                        best_gain = gain

            if best_target is None:
                break

            active = best_target

        objective = sum(empirical_means[worker_id - 1] for worker_id in active)
        return active, objective

    def test_simulate_results_do_not_depend_on_policy_order(self) -> None:
        kwargs = {
            "k": 5,
            "m": 2,
            "T": 12,
            "means": [0.9, 0.8, 0.5, 0.3, 0.1],
            "delay_upper": 2,
            "c": 1.0,
            "n_runs": 3,
            "seed0": 7,
        }

        _, results_ab = simulate(
            policies=["a-omm", "epsilon-greedy"],
            **kwargs,
        )
        _, results_ba = simulate(
            policies=["epsilon-greedy", "a-omm"],
            **kwargs,
        )

        np.testing.assert_allclose(results_ab["a-omm"][0], results_ba["a-omm"][0])
        np.testing.assert_allclose(results_ab["a-omm"][1], results_ba["a-omm"][1])
        np.testing.assert_allclose(
            results_ab["epsilon-greedy"][0],
            results_ba["epsilon-greedy"][0],
        )
        np.testing.assert_allclose(
            results_ab["epsilon-greedy"][1],
            results_ba["epsilon-greedy"][1],
        )

    def test_simulate_parallel_matches_sequential_for_builtin_processes(self) -> None:
        kwargs = {
            "policies": ["a-omm", "delayed-replace-ucb"],
            "k": 5,
            "m": 2,
            "T": 20,
            "means": [0.9, 0.8, 0.5, 0.3, 0.1],
            "delay_upper": 2,
            "c": 1.0,
            "n_runs": 2,
            "seed0": 11,
        }

        _, sequential = simulate(n_jobs=1, **kwargs)
        _, parallel = simulate(n_jobs=2, **kwargs)

        for policy_name in kwargs["policies"]:
            np.testing.assert_allclose(sequential[policy_name][0], parallel[policy_name][0])
            np.testing.assert_allclose(sequential[policy_name][1], parallel[policy_name][1])

    def test_adapted_aht_counts_first_target_active_period_toward_block(self) -> None:
        policy = AdaptedAHTPolicy(
            k=3,
            m=1,
            rng=random.Random(0),
        )
        policy.reset()
        policy._initialize_schedule()
        policy.phase = "transition"
        policy.current_target = {1}
        policy.block_remaining = 1

        policy.update(
            StepFeedback(
                individual_rewards={1: 1.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )

        self.assertEqual(policy.phase, "ready")

    def test_stateful_policy_base_does_not_block_replacements_during_hold(self) -> None:
        class _NonBlockingScheduledPolicy(StatefulDelayedActionPolicy):
            def reset_control_state(self) -> None:
                return None

            def initialize_control(self, active_now, env) -> None:
                self.current_target = set(active_now)
                self.phase = "hold"

            def plan_next_target(self, active_now, env):
                return [1, 3]

            def on_hold_feedback(self, feedback: StepFeedback) -> None:
                return None

            def compute_target(self):
                return [1, 3]

        policy = _NonBlockingScheduledPolicy(k=3, m=2, rng=random.Random(0))
        policy.reset()
        env = self._PolicyEnv({1, 2})

        replacements = policy.act(env)

        self.assertEqual(replacements, [(2, 3)])
        self.assertEqual(policy.current_target, {1, 3})
        self.assertEqual(policy.phase, "transition")

    def test_policy_does_not_cancel_conflicting_pending_replacements(self) -> None:
        policy = Threshold(k=4, m=2, threshold=0.0, rng=random.Random(0))
        env = self._PendingPolicyEnv({1, 2}, pending_pairs=[(1, 3)])

        replacements = policy._propose_replacements_from_target(env, [1, 4])

        self.assertEqual(replacements, [(2, 4)])
        self.assertEqual([(pending.i, pending.j) for pending in env.pending], [(1, 3)])
        self.assertEqual(env.pending_workers, {1, 3})

    def test_policy_keeps_aligned_pending_replacements_when_target_matches(self) -> None:
        policy = Threshold(k=4, m=2, threshold=0.0, rng=random.Random(0))
        env = self._PendingPolicyEnv({1, 2}, pending_pairs=[(1, 3)])

        replacements = policy._propose_replacements_from_target(env, [2, 3])

        self.assertEqual(replacements, [])
        self.assertEqual([(pending.i, pending.j) for pending in env.pending], [(1, 3)])
        self.assertEqual(env.pending_workers, {1, 3})

    def test_environment_rejects_replacements_that_conflict_with_pending(self) -> None:
        env = TemporaryHiringBanditEnv(
            k=4,
            m=2,
            reward_samplers=[lambda: 0.0 for _ in range(4)],
            delay_sampler=lambda pair, t: 2,
            c=1.0,
            omega_mean=2.0,
            initial_workforce=[1, 2],
            rng=random.Random(0),
        )

        env.step([(1, 3)])
        _, _, cost, _ = env.step([(1, 4), (2, 4)])

        self.assertEqual(cost, 1.0)
        self.assertEqual(
            sorted((pending.i, pending.j) for pending in env.pending),
            [(1, 3), (2, 4)],
        )
        self.assertEqual(env.pending_workers, {1, 2, 3, 4})

    def test_delayed_replace_ucb_counts_first_target_active_period_toward_buffer(self) -> None:
        policy = DelayedReplaceUCB(
            k=3,
            m=1,
            gamma=1.0,
            horizon=10,
            rng=random.Random(0),
        )
        policy.phase = "transition"
        policy.current_target = {1}
        policy.i_min = 1
        policy.i_min_count_at_c = 0
        policy.block_threshold_N = 1

        policy.update(
            StepFeedback(
                individual_rewards={1: 1.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )

        self.assertEqual(policy.phase, "ready")

    def test_delayed_replace_ucb_waits_for_target_realization_before_becoming_ready(self) -> None:
        policy = DelayedReplaceUCB(
            k=3,
            m=1,
            gamma=1.0,
            horizon=10,
            rng=random.Random(0),
        )
        policy.phase = "transition"
        policy.current_target = {1}
        policy.i_min = 1
        policy.i_min_count_at_c = 0
        policy.block_threshold_N = 1

        policy.update(
            StepFeedback(
                individual_rewards={1: 1.0},
                active_set=frozenset({2}),
                completed_this_period=(),
                pending_count=0,
            )
        )

        self.assertEqual(policy.phase, "hold")

    def test_choose_target_uses_horizon_aware_aggregate_dp(self) -> None:
        result = choose_target(
            active_set=[1, 2],
            counts=np.ones(4, dtype=np.int64),
            empirical_means=np.array([0.5, 0.1, 0.9, 0.55], dtype=np.float64),
            current_period=8,
            horizon=10,
            switching_cost=1.0,
            ucb_coef=0.0,
            time_index=8,
        )

        self.assertEqual(set(result.target), {1, 3})
        self.assertEqual(result.matched_pairs, ((2, 3),))

    def test_choose_target_can_record_frontier_sizes(self) -> None:
        frontier_size_log = []

        choose_target(
            active_set=[1, 2],
            counts=[1, 1, 1, 1],
            empirical_means=[0.5, 0.1, 0.9, 0.55],
            current_period=8,
            horizon=10,
            switching_cost=1.0,
            ucb_coef=0.0,
            time_index=8,
            frontier_size_log=frontier_size_log,
        )

        self.assertEqual(len(frontier_size_log), 6)
        self.assertEqual(frontier_size_log[0].active_prefix_size, 0)
        self.assertEqual(frontier_size_log[0].replacement_count, 0)
        self.assertEqual(frontier_size_log[0].candidate_count, 1)
        self.assertEqual(frontier_size_log[0].frontier_size, 1)
        self.assertTrue(
            any(
                record.active_prefix_size == 2
                and record.replacement_count == 1
                and record.frontier_size > 0
                for record in frontier_size_log
            )
        )
        self.assertTrue(all(record.time_index == 8 for record in frontier_size_log))

    def test_delayed_replace_ucb_compute_target_respects_switching_constraint(self) -> None:
        policy = DelayedReplaceUCB(
            k=4,
            m=2,
            gamma=1.0,
            horizon=10,
            rng=random.Random(0),
        )
        policy.current_target = {1, 2}
        policy.counts[:] = 1
        policy.sums[:] = [0.5, 0.1, 0.9, 0.55]
        policy.t = 8
        policy.cfg.ucb_coef = 0.0

        target = policy.compute_target(
            active=[1, 2],
            current_period=8,
            switching_cost=1.0,
        )

        self.assertEqual(target, {1, 3})

    def test_simulate_can_collect_delayed_replace_ucb_frontier_logs(self) -> None:
        frontier_size_log = []

        simulate(
            policies=["delayed-replace-ucb"],
            k=4,
            m=2,
            T=5,
            means=[0.9, 0.7, 0.4, 0.2],
            delay_upper=1,
            c=0.5,
            n_runs=1,
            seed0=7,
            gamma=1.0,
            frontier_size_log=frontier_size_log,
        )

        self.assertTrue(frontier_size_log)
        self.assertTrue(
            all(record.policy_name == "delayed-replace-ucb" for record in frontier_size_log)
        )
        self.assertTrue(all(record.episode_seed == 7 for record in frontier_size_log))
        self.assertTrue(
            all(record.decision_iteration is not None for record in frontier_size_log)
        )

    def test_delayed_replace_ucb_uses_aggregate_switching_rule(self) -> None:
        policy = DelayedReplaceUCB(
            k=5,
            m=3,
            gamma=1.0,
            horizon=10,
            rng=random.Random(0),
        )
        policy.current_target = {1, 2, 3}
        policy.counts[:] = 1
        policy.sums[:] = [0.9, 0.8, 0.1, 1.0, 0.95]
        policy.t = 8
        policy.cfg.ucb_coef = 0.0

        target = policy.compute_target(
            active=[1, 2, 3],
            current_period=8,
            switching_cost=0.9,
        )
        pairs = policy.construct_bijection(
            current=[1, 2, 3],
            target=sorted(target),
            current_period=8,
            switching_cost=0.9,
        )

        self.assertEqual(target, {1, 4, 5})
        self.assertEqual(pairs, [(2, 4), (3, 5)])

    def test_choose_target_matches_bruteforce_enumeration_on_small_instance(self) -> None:
        active_set = [1, 2, 3]
        empirical_means = [0.75, 0.4, 0.2, 0.9, 0.7]
        current_period = 8
        horizon = 10
        switching_cost = 0.8

        expected_target, expected_pairs, expected_objective = (
            self._bruteforce_choose_target_zero_ucb_bonus(
                active_set=active_set,
                empirical_means=empirical_means,
                current_period=current_period,
                horizon=horizon,
                switching_cost=switching_cost,
            )
        )

        result = choose_target(
            active_set=active_set,
            counts=[1, 1, 1, 1, 1],
            empirical_means=empirical_means,
            current_period=current_period,
            horizon=horizon,
            switching_cost=switching_cost,
            ucb_coef=0.0,
            time_index=current_period,
        )
        result_objective = sum(empirical_means[worker_id - 1] for worker_id in result.target)

        self.assertEqual(expected_target, {1, 4, 5})
        self.assertEqual(expected_pairs, ((2, 4), (3, 5)))
        self.assertAlmostEqual(expected_objective, 2.35)
        self.assertEqual(set(result.target), expected_target)
        self.assertEqual(result.matched_pairs, expected_pairs)
        self.assertAlmostEqual(result_objective, expected_objective)

    def test_choose_target_beats_myopic_greedy_on_small_instance(self) -> None:
        active_set = [1, 2, 3]
        empirical_means = [0.9, 0.6, 0.1, 0.95, 0.8, 0.7]
        current_period = 8
        horizon = 12
        switching_cost = 1.0

        expected_target, expected_pairs, expected_objective = (
            self._bruteforce_choose_target_zero_ucb_bonus(
                active_set=active_set,
                empirical_means=empirical_means,
                current_period=current_period,
                horizon=horizon,
                switching_cost=switching_cost,
            )
        )
        greedy_target, greedy_objective = self._myopic_greedy_choose_target_zero_ucb_bonus(
            active_set=active_set,
            empirical_means=empirical_means,
            current_period=current_period,
            horizon=horizon,
            switching_cost=switching_cost,
        )

        result = choose_target(
            active_set=active_set,
            counts=[1, 1, 1, 1, 1, 1],
            empirical_means=empirical_means,
            current_period=current_period,
            horizon=horizon,
            switching_cost=switching_cost,
            ucb_coef=0.0,
            time_index=current_period,
        )
        result_objective = sum(empirical_means[worker_id - 1] for worker_id in result.target)

        self.assertEqual(greedy_target, {1, 2, 4})
        self.assertAlmostEqual(greedy_objective, 2.45)
        self.assertEqual(expected_target, {1, 4, 5})
        self.assertEqual(expected_pairs, ((2, 4), (3, 5)))
        self.assertAlmostEqual(expected_objective, 2.65)
        self.assertEqual(set(result.target), expected_target)
        self.assertEqual(result.matched_pairs, expected_pairs)
        self.assertAlmostEqual(result_objective, expected_objective)
        self.assertGreater(result_objective, greedy_objective)

    def test_choose_target_matches_bruteforce_on_aggregate_instance(
        self,
    ) -> None:
        active_set = [1, 2, 3]
        empirical_means = [0.9, 0.8, 0.1, 1.0, 0.95]
        current_period = 8
        horizon = 10
        switching_cost = 0.9

        expected_target, expected_pairs, expected_objective = (
            self._bruteforce_choose_target_zero_ucb_bonus(
                active_set=active_set,
                empirical_means=empirical_means,
                current_period=current_period,
                horizon=horizon,
                switching_cost=switching_cost,
            )
        )

        result = choose_target(
            active_set=active_set,
            counts=[1, 1, 1, 1, 1],
            empirical_means=empirical_means,
            current_period=current_period,
            horizon=horizon,
            switching_cost=switching_cost,
            ucb_coef=0.0,
            time_index=current_period,
        )
        result_objective = sum(
            empirical_means[worker_id - 1]
            for worker_id in result.target
        )

        self.assertEqual(expected_target, {1, 4, 5})
        self.assertEqual(expected_pairs, ((2, 4), (3, 5)))
        self.assertAlmostEqual(expected_objective, 2.85)
        self.assertEqual(set(result.target), expected_target)
        self.assertEqual(result.matched_pairs, expected_pairs)
        self.assertAlmostEqual(result_objective, expected_objective)

    def test_make_policy_resolves_auto_gamma_for_delayed_replace_ucb(self) -> None:
        kwargs = {
            "k": 10,
            "m": 3,
            "T": 2000,
            "c": 5.0,
            "omega_mean": 4,
        }
        expected_gamma = compute_delayed_replace_ucb_auto_gamma(**kwargs)

        policy = make_policy(
            "delayed-replace-ucb",
            rng=random.Random(0),
            gamma="auto",
            **kwargs,
        )
        self.assertIsInstance(policy, DelayedReplaceUCB)
        assert isinstance(policy, DelayedReplaceUCB)

        self.assertGreater(policy.cfg.gamma, 0.0)
        self.assertAlmostEqual(policy.cfg.gamma, expected_gamma)

        base_bound = delayed_replace_ucb_regret_bound(policy.cfg.gamma, **kwargs)
        lower_bound = delayed_replace_ucb_regret_bound(policy.cfg.gamma * 0.8, **kwargs)
        upper_bound = delayed_replace_ucb_regret_bound(policy.cfg.gamma * 1.2, **kwargs)

        self.assertLessEqual(base_bound, lower_bound)
        self.assertLessEqual(base_bound, upper_bound)

    def test_make_policy_parses_auto_gamma_omega_override(self) -> None:
        kwargs = {
            "k": 10,
            "m": 3,
            "T": 2000,
            "c": 5.0,
            "omega_mean": 4,
        }

        policy_zero = make_policy(
            "delayed-replace-ucb",
            rng=random.Random(0),
            gamma="auto0",
            **kwargs,
        )
        expected_zero_gamma = compute_delayed_replace_ucb_auto_gamma(
            **{**kwargs, "omega_mean": 0.0}
        )
        self.assertAlmostEqual(policy_zero.cfg.gamma, expected_zero_gamma)

        policy_custom = make_policy(
            "delayed-replace-ucb",
            rng=random.Random(0),
            gamma="auto-2.5",
            **kwargs,
        )
        expected_custom_gamma = compute_delayed_replace_ucb_auto_gamma(
            **{**kwargs, "omega_mean": 2.5}
        )
        self.assertAlmostEqual(policy_custom.cfg.gamma, expected_custom_gamma)

        with self.assertRaises(ValueError):
            make_policy(
                "delayed-replace-ucb",
                rng=random.Random(0),
                gamma="auto-n",
                **kwargs,
            )

    def test_uniform_delay_sampler_can_use_positive_lower_bound(self) -> None:
        sampler = make_uniform_delay_sampler(
            6,
            random.Random(0),
            delay_lower=3,
        )
        draws = [sampler((1, 2), 1) for _ in range(200)]

        self.assertGreaterEqual(min(draws), 3)
        self.assertLessEqual(max(draws), 6)

    def test_geometric_delay_sampler_respects_bounds_and_p(self) -> None:
        sampler = make_geometric_delay_sampler(
            p=0.4,
            rng=random.Random(0),
            delay_lower=2,
            delay_upper=6,
        )
        draws = [sampler((1, 2), 1) for _ in range(1000)]

        self.assertGreaterEqual(min(draws), 2)
        self.assertLessEqual(max(draws), 6)
        self.assertGreater(Counter(draws)[2], Counter(draws)[5])

    def test_geometric_delay_factory_builds_sampler(self) -> None:
        factory = make_delay_sampler_factory(
            "geometric",
            means=[0.8, 0.2],
            delay_upper=5,
            delay_lower=1,
            delay_geom_p=1.0,
        )

        sampler = factory(7)
        self.assertEqual(sampler((1, 2), 3), 1)

    def test_geometric_delay_factory_can_be_unbounded_from_omega_mean(self) -> None:
        factory = make_delay_sampler_factory(
            "geometric",
            means=[0.8, 0.2],
            omega_mean=100.0,
        )

        sampler = factory(0)
        draws = [sampler((1, 2), 3) for _ in range(200)]
        self.assertGreater(max(draws), 3)

    def test_environment_allows_unbounded_delay_without_delay_upper(self) -> None:
        env = TemporaryHiringBanditEnv(
            k=2,
            m=1,
            reward_samplers=[lambda: 0.0, lambda: 0.0],
            delay_sampler=lambda pair, t: 10,
            initial_workforce=[1],
            omega_mean=10.0,
        )

        _, _, _, feedback = env.step([(1, 2)])
        self.assertEqual(feedback.accepted_delays, (10,))
        self.assertEqual(feedback.pending_count, 1)

    def test_make_policy_auto_gamma_uses_omega_mean(self) -> None:
        kwargs = {
            "k": 10,
            "m": 3,
            "T": 2000,
            "c": 5.0,
            "omega_mean": 2.5,
        }
        policy = make_policy(
            "delayed-replace-ucb",
            rng=random.Random(0),
            gamma="auto",
            **kwargs,
        )
        expected_gamma = compute_delayed_replace_ucb_auto_gamma(
            k=kwargs["k"],
            m=kwargs["m"],
            T=kwargs["T"],
            c=kwargs["c"],
            omega_mean=kwargs["omega_mean"],
        )

        self.assertAlmostEqual(policy.cfg.gamma, expected_gamma)

    def test_simulate_geometric_delay_can_use_omega_mean_without_delay_upper(self) -> None:
        _, results = simulate(
            policies=["delayed-replace-ucb"],
            k=3,
            m=1,
            T=3,
            means=[0.9, 0.4, 0.2],
            c=0.0,
            delay_process_name="geometric",
            omega_mean=2.0,
            gamma="auto",
            n_runs=1,
            seed0=0,
        )

        self.assertEqual(results["delayed-replace-ucb"][0].shape, (3,))

    def test_calendar_delay_sampler_geom_prefers_earlier_feasible_periods(self) -> None:
        sampler = make_calendar_delay_sampler(
            20,
            frequency=8,
            distribution="geom",
            geom_p=0.5,
            rng=random.Random(0),
        )

        counts = Counter(sampler((1, 2), 5) for _ in range(2000))

        self.assertGreater(counts[3], counts[11])
        self.assertGreater(counts[11], counts[19])

    def test_calendar_adversarial_delay_uses_calendar_extremes(self) -> None:
        sampler = make_calendar_adversarial_delay(
            means=[0.8, 0.2, 0.95],
            delay_upper=20,
            frequency=8,
        )

        self.assertEqual(sampler((1, 2), 5), 3)
        self.assertEqual(sampler((2, 3), 5), 19)
        self.assertEqual(sampler((2, 3), 8), 16)

    def test_calendar_adversarial_delay_factory_builds_sampler(self) -> None:
        factory = make_delay_sampler_factory(
            "calendar-adversarial",
            means=[0.8, 0.2, 0.95],
            delay_upper=20,
            calendar_frequency=8,
        )

        sampler = factory(11)
        self.assertEqual(sampler((1, 2), 5), 3)
        self.assertEqual(sampler((2, 3), 5), 19)

    def test_average_regret_results_divides_by_time_and_oracle_reward(self) -> None:
        averaged = _average_regret_results(
            [0.9, 0.8, 0.1],
            2,
            {
                "A-OMM": (
                    np.asarray([2.0, 6.0, 12.0], dtype=np.float64),
                    np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                )
            }
        )

        mean_curve, std_curve = averaged["A-OMM"]
        np.testing.assert_allclose(
            mean_curve,
            np.asarray([2.0, 3.0, 4.0]) / 1.7,
        )
        np.testing.assert_allclose(
            std_curve,
            np.asarray([1.0, 1.5, 2.0]) / 1.7,
        )

    def test_threshold_policy_replaces_below_threshold_worker_with_untried_worker(self) -> None:
        policy = Threshold(k=4, m=2, threshold=0.5, rng=random.Random(0))
        policy.counts[:] = np.asarray([5, 5, 0, 4], dtype=np.int64)
        policy.sums[:] = np.asarray([2.0, 4.0, 0.0, 3.2], dtype=np.float64)

        replacements = policy.act(self._StaticEnv({1, 2}))

        self.assertEqual(replacements, [(1, 3)])

    def test_threshold_policy_uses_best_available_worker_after_all_workers_tried(self) -> None:
        policy = make_policy(
            "threshold",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_mean=1.0,
            rng=random.Random(0),
            threshold=0.5,
        )
        policy.counts[:] = np.asarray([5, 5, 4, 4], dtype=np.int64)
        policy.sums[:] = np.asarray([2.0, 4.0, 3.6, 2.4], dtype=np.float64)

        replacements = policy.act(self._StaticEnv({1, 2}))

        self.assertEqual(replacements, [(1, 3)])

    def test_make_policy_rejects_removed_threshold_alias(self) -> None:
        with self.assertRaises(ValueError):
            make_policy(
                "threshold-0.75",
                k=4,
                m=2,
                T=10,
                c=0.0,
                omega_mean=1.0,
                rng=random.Random(0),
            )

    def test_make_policy_builds_fixed_schedule_greedy_with_custom_interval(self) -> None:
        policy = make_policy(
            "FixedScheduleGreedy",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_mean=1.0,
            rng=random.Random(0),
            review_interval=12,
        )

        self.assertIsInstance(policy, FixedScheduleGreedy)
        assert isinstance(policy, FixedScheduleGreedy)
        self.assertEqual(policy.review_interval, 12)

    def test_make_policy_builds_pre_screen_with_correlated_estimates(self) -> None:
        true_means = [0.1, 0.8, 0.4, 0.7]
        policy = make_policy(
            "pre-screen",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_mean=0.0,
            rng=random.Random(0),
            true_means=true_means,
            rho=0.5,
            cost=3.5,
        )

        self.assertIsInstance(policy, PreScreen)
        assert isinstance(policy, PreScreen)
        self.assertEqual(policy.initial_regret, 3.5)
        self.assertAlmostEqual(
            float(np.corrcoef(policy.estimates, true_means)[0, 1]),
            0.5,
        )

        fixed_target = policy.compute_target()
        self.assertEqual(policy.compute_target(), fixed_target)
        policy.update(
            StepFeedback(
                individual_rewards={worker_id: -100.0 for worker_id in fixed_target},
                active_set=frozenset(fixed_target),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.compute_target(), fixed_target)

    def test_pre_screen_initial_cost_is_charged_before_first_period(self) -> None:
        _, results = simulate(
            policies=["pre-screen"],
            k=3,
            m=1,
            T=1,
            means=[0.1, 0.9, 0.2],
            c=0.0,
            delay_upper=0,
            rho=1.0,
            cost=7.5,
            n_runs=1,
            seed0=0,
        )

        mean_curve, std_curve = results["pre-screen"]
        self.assertAlmostEqual(float(mean_curve[0]), 7.5)
        self.assertAlmostEqual(float(std_curve[0]), 0.0)

    def test_fixed_schedule_greedy_reselects_team_on_review_schedule(self) -> None:
        policy = FixedScheduleGreedy(k=3, m=1, review_interval=2, rng=random.Random(0))

        self.assertEqual(policy.act(self._StaticEnv({1})), [])
        self.assertEqual(policy.phase, "hold")

        policy.counts[:] = np.asarray([5, 5, 5], dtype=np.int64)
        policy.sums[:] = np.asarray([2.0, 4.5, 4.8], dtype=np.float64)

        policy.update(
            StepFeedback(
                individual_rewards={1: 0.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "hold")

        policy.update(
            StepFeedback(
                individual_rewards={1: 0.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "ready")

        replacements = policy.act(self._StaticEnv({1}))
        self.assertEqual(replacements, [(1, 3)])

    def test_make_policy_builds_work_trial_with_custom_schedule(self) -> None:
        true_means = [0.1, 0.8, 0.4, 0.7]
        policy = make_policy(
            "WorkTrial",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_mean=1.0,
            rng=random.Random(0),
            true_means=true_means,
            rho=0.5,
            cost=4.0,
            work_trial_rotation_periods=7,
        )

        self.assertIsInstance(policy, WorkTrial)
        assert isinstance(policy, WorkTrial)
        self.assertAlmostEqual(
            float(np.corrcoef(policy.prescreen_estimates, true_means)[0, 1]),
            0.5,
        )
        self.assertEqual(policy.initial_regret, 4.0)
        self.assertEqual(policy.rotation_periods, 7)

    def test_work_trial_uses_prescreen_before_rotation_blocks(self) -> None:
        policy = WorkTrial(
            k=4,
            m=1,
            true_means=[0.1, 0.4, 0.9, 0.8],
            rho=1.0,
            cost=2.0,
            rotation_periods=2,
            rng=random.Random(0),
        )

        self.assertEqual(policy.initial_workforce, [3])
        self.assertEqual(policy.shortlist, [3, 4])
        self.assertEqual(policy.initial_regret, 2.0)
        self.assertEqual(policy.act(self._StaticEnv({3})), [])
        self.assertEqual(policy.stage, "first_block")

        policy.update(
            StepFeedback(
                individual_rewards={3: 0.5},
                active_set=frozenset({3}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "hold")

        policy.update(
            StepFeedback(
                individual_rewards={3: 0.6},
                active_set=frozenset({3}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "ready")
        self.assertEqual(policy.stage, "second_block")

        replacements = policy.act(self._StaticEnv({3}))
        self.assertEqual(replacements, [(3, 4)])

    def test_build_planning_horizon_regret_table_uses_precomputed_results(self) -> None:
        rows = build_planning_horizon_regret_table(
            series=[ExperimentSeries(policy_name="a-omm", label="A-OMM")],
            means=[0.9, 0.8, 0.1],
            m=2,
            results={
                "A-OMM": (
                    np.asarray([2.0, 6.0, 12.0], dtype=np.float64),
                    np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                )
            },
            horizons=[("Month 1", 2), ("Year 1", 4)],
        )

        self.assertEqual(rows[0]["horizon"], "Month 1")
        self.assertAlmostEqual(float(rows[0]["A-OMM cumulative"]), 6.0)
        self.assertAlmostEqual(float(rows[0]["A-OMM normalized"]), 3.0 / 1.7)
        self.assertIsNone(rows[1]["A-OMM cumulative"])
        self.assertIsNone(rows[1]["A-OMM normalized"])

    def test_benchmark_metadata_sanitizes_custom_sampler_functions(self) -> None:
        def reward_sampler() -> float:
            return 1.0

        metadata = _json_safe_metadata(
            {
                "simulate_kwargs": {
                    "reward_samplers": [reward_sampler],
                    "means": np.asarray([0.2, 0.8], dtype=np.float64),
                }
            }
        )

        json.dumps(metadata)
        self.assertIn("<callable", metadata["simulate_kwargs"]["reward_samplers"][0])
        self.assertEqual(metadata["simulate_kwargs"]["means"], [0.2, 0.8])

    def test_truncated_normal_samplers_are_bounded_and_match_target_mean(self) -> None:
        sampler = make_truncated_normal_samplers(
            [0.75],
            random.Random(0),
            stddev=0.1,
            lower=0.0,
            upper=1.0,
        )[0]

        draws = np.asarray([sampler() for _ in range(4000)], dtype=np.float64)

        self.assertTrue(np.all(draws >= 0.0))
        self.assertTrue(np.all(draws <= 1.0))
        self.assertAlmostEqual(float(draws.mean()), 0.75, delta=0.03)

    def test_reward_sampler_factory_supports_truncated_normal(self) -> None:
        factory = make_reward_sampler_factory(
            "truncated-normal",
            means=[0.25],
            reward_stddev=0.08,
            reward_lower=0.0,
            reward_upper=1.0,
        )
        sampler = factory(3)[0]

        draws = np.asarray([sampler() for _ in range(3000)], dtype=np.float64)
        self.assertTrue(np.all(draws >= 0.0))
        self.assertTrue(np.all(draws <= 1.0))
        self.assertAlmostEqual(float(draws.mean()), 0.25, delta=0.03)

    def test_conditional_sampler_resamples_acceptance_each_draw(self) -> None:
        acceptance_draws = iter([False, True, False, True])
        reward_draws = iter([0.4, 0.8])
        sampler = make_conditional_samplers(
            accept=lambda: next(acceptance_draws),
            reward=lambda: next(reward_draws),
        )

        self.assertEqual(
            [sampler(), sampler(), sampler(), sampler()],
            [0.0, 0.4, 0.0, 0.8],
        )

    def test_conditional_sampler_composes_sampler_sequences(self) -> None:
        samplers = make_conditional_samplers(
            accept=[lambda: False, lambda: True],
            reward=[lambda: 0.4, lambda: 0.8],
        )

        self.assertEqual([sampler() for sampler in samplers], [0.0, 0.8])

    def test_reward_samplers_accept_numpy_generator(self) -> None:
        rng = np.random.default_rng(0)
        bernoulli_sampler = make_bernoulli_samplers([1.0], rng)[0]
        normal_sampler = make_truncated_normal_samplers([0.5], rng)[0]

        self.assertEqual(bernoulli_sampler(), 1.0)
        self.assertGreaterEqual(normal_sampler(), 0.0)
        self.assertLessEqual(normal_sampler(), 1.0)


if __name__ == "__main__":
    unittest.main()
