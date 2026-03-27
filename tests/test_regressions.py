import os
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

mpl_dir = Path(tempfile.gettempdir()) / "hiringbandit-mpl"
mpl_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

from bandit_environment import StepFeedback
from hiring_ucb import HiringUCBPolicy
from policies import AgrawalHegdeTeneketzisPolicy, SemiAnnualReview, Threshold, WorkTrial
from samplers import (
    make_calendar_adversarial_delay,
    make_calendar_delay_sampler,
    make_truncated_normal_samplers,
    make_uniform_delay_sampler,
)
from simulation import (
    _average_regret_results,
    build_planning_horizon_regret_table,
    ExperimentSeries,
    make_delay_sampler_factory,
    make_reward_sampler_factory,
    compute_optimistic_hire_auto_gamma,
    make_policy,
    optimistic_hire_regret_bound,
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

    def test_simulate_results_do_not_depend_on_policy_order(self) -> None:
        kwargs = {
            "k": 5,
            "m": 2,
            "T": 12,
            "means": [0.9, 0.8, 0.5, 0.3, 0.1],
            "omega_max": 2,
            "c": 1.0,
            "n_runs": 3,
            "seed0": 7,
        }

        _, results_ab = simulate(
            policies=["omm", "epsilon-greedy"],
            **kwargs,
        )
        _, results_ba = simulate(
            policies=["epsilon-greedy", "omm"],
            **kwargs,
        )

        np.testing.assert_allclose(results_ab["omm"][0], results_ba["omm"][0])
        np.testing.assert_allclose(results_ab["omm"][1], results_ba["omm"][1])
        np.testing.assert_allclose(
            results_ab["epsilon-greedy"][0],
            results_ba["epsilon-greedy"][0],
        )
        np.testing.assert_allclose(
            results_ab["epsilon-greedy"][1],
            results_ba["epsilon-greedy"][1],
        )

    def test_aht_counts_first_target_active_period_toward_block(self) -> None:
        policy = AgrawalHegdeTeneketzisPolicy(
            k=3,
            m=1,
            init_pulls=1,
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

    def test_hiring_ucb_counts_first_target_active_period_toward_buffer(self) -> None:
        policy = HiringUCBPolicy(
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
        policy.buffer_threshold_N = 1

        policy.update(
            StepFeedback(
                individual_rewards={1: 1.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )

        self.assertEqual(policy.phase, "ready")

    def test_hiring_ucb_bijection_returns_no_switches_at_horizon(self) -> None:
        policy = HiringUCBPolicy(
            k=2,
            m=1,
            gamma=1.0,
            horizon=5,
            rng=random.Random(0),
        )
        policy.counts[:] = 1
        policy.sums[:] = 0.5
        policy.t = 5

        pairs = policy.bijection(
            current=[1],
            target=[2],
            current_period=5,
            switching_cost=1.0,
        )

        self.assertEqual(pairs, [])

    def test_make_policy_resolves_auto_gamma_for_optimistic_hire(self) -> None:
        kwargs = {
            "k": 10,
            "m": 3,
            "T": 2000,
            "c": 5.0,
            "omega_max": 4,
        }
        expected_gamma = compute_optimistic_hire_auto_gamma(**kwargs)

        policy = make_policy(
            "optimistic-hire",
            rng=random.Random(0),
            gamma="auto",
            **kwargs,
        )
        self.assertIsInstance(policy, HiringUCBPolicy)
        assert isinstance(policy, HiringUCBPolicy)

        self.assertGreater(policy.cfg.gamma, 0.0)
        self.assertAlmostEqual(policy.cfg.gamma, expected_gamma)

        base_bound = optimistic_hire_regret_bound(policy.cfg.gamma, **kwargs)
        lower_bound = optimistic_hire_regret_bound(policy.cfg.gamma * 0.8, **kwargs)
        upper_bound = optimistic_hire_regret_bound(policy.cfg.gamma * 1.2, **kwargs)

        self.assertLessEqual(base_bound, lower_bound)
        self.assertLessEqual(base_bound, upper_bound)

    def test_calendar_delay_sampler_only_returns_calendar_aligned_delays(self) -> None:
        sampler = make_calendar_delay_sampler(
            20,
            frequency=8,
            distribution="unif",
            rng=random.Random(0),
        )

        observed = {sampler((1, 2), 5) for _ in range(200)}
        self.assertEqual(observed, {3, 11, 19})

        factory = make_delay_sampler_factory(
            "calendar",
            means=[0.8, 0.2],
            omega_max=20,
            calendar_frequency=8,
            calendar_distribution="unif",
        )
        factory_sampler = factory(7)
        for _ in range(50):
            delay = factory_sampler((1, 2), 8)
            self.assertIn(delay, {8, 16})

    def test_uniform_delay_sampler_can_use_positive_lower_bound(self) -> None:
        sampler = make_uniform_delay_sampler(
            6,
            random.Random(0),
            delay_lower=3,
        )
        draws = [sampler((1, 2), 1) for _ in range(200)]

        self.assertGreaterEqual(min(draws), 3)
        self.assertLessEqual(max(draws), 6)

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
            omega_max=20,
            frequency=8,
        )

        self.assertEqual(sampler((1, 2), 5), 3)
        self.assertEqual(sampler((2, 3), 5), 19)
        self.assertEqual(sampler((2, 3), 8), 16)

    def test_calendar_adversarial_delay_factory_builds_sampler(self) -> None:
        factory = make_delay_sampler_factory(
            "calendar-adversarial",
            means=[0.8, 0.2, 0.95],
            omega_max=20,
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
                "OMM": (
                    np.asarray([2.0, 6.0, 12.0], dtype=np.float64),
                    np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                )
            }
        )

        mean_curve, std_curve = averaged["OMM"]
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
            omega_max=1,
            rng=random.Random(0),
            threshold=0.5,
        )
        policy.counts[:] = np.asarray([5, 5, 4, 4], dtype=np.int64)
        policy.sums[:] = np.asarray([2.0, 4.0, 3.6, 2.4], dtype=np.float64)

        replacements = policy.act(self._StaticEnv({1, 2}))

        self.assertEqual(replacements, [(1, 3)])

    def test_threshold_policy_name_can_encode_threshold_value(self) -> None:
        policy = make_policy(
            "threshold-0.75",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_max=1,
            rng=random.Random(0),
        )

        self.assertIsInstance(policy, Threshold)
        assert isinstance(policy, Threshold)
        self.assertAlmostEqual(policy.threshold, 0.75)

    def test_make_policy_builds_semiannual_review_with_custom_interval(self) -> None:
        policy = make_policy(
            "SemiAnnualReview",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_max=1,
            rng=random.Random(0),
            review_interval=12,
        )

        self.assertIsInstance(policy, SemiAnnualReview)
        assert isinstance(policy, SemiAnnualReview)
        self.assertEqual(policy.review_interval, 12)

    def test_semiannual_review_reselects_team_on_review_schedule(self) -> None:
        policy = SemiAnnualReview(k=3, m=1, review_interval=2, rng=random.Random(0))

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
        policy = make_policy(
            "WorkTrial",
            k=4,
            m=2,
            T=10,
            c=0.0,
            omega_max=1,
            rng=random.Random(0),
            work_trial_periods=2,
            work_trial_rotation_periods=7,
        )

        self.assertIsInstance(policy, WorkTrial)
        assert isinstance(policy, WorkTrial)
        self.assertEqual(policy.trial_periods, 2)
        self.assertEqual(policy.rotation_periods, 7)

    def test_work_trial_moves_from_trial_stage_to_first_block(self) -> None:
        policy = WorkTrial(k=4, m=1, trial_periods=1, rotation_periods=2, rng=random.Random(0))

        self.assertEqual(policy.act(self._StaticEnv({1})), [])
        self.assertEqual(policy.stage, "trial")

        policy.update(
            StepFeedback(
                individual_rewards={1: 1.0},
                active_set=frozenset({1}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "ready")

        replacements = policy.act(self._StaticEnv({1}))
        self.assertEqual(replacements, [(1, 2)])

        policy.current_target = {2}
        policy.phase = "transition"
        policy.update(
            StepFeedback(
                individual_rewards={2: 1.0},
                active_set=frozenset({2}),
                completed_this_period=(),
                pending_count=0,
            )
        )
        self.assertEqual(policy.phase, "ready")

        replacements = policy.act(self._StaticEnv({2}))
        self.assertEqual(replacements, [(2, 3)])

    def test_build_planning_horizon_regret_table_uses_precomputed_results(self) -> None:
        rows = build_planning_horizon_regret_table(
            series=[ExperimentSeries(policy_name="omm", label="OMM")],
            means=[0.9, 0.8, 0.1],
            m=2,
            results={
                "OMM": (
                    np.asarray([2.0, 6.0, 12.0], dtype=np.float64),
                    np.asarray([1.0, 3.0, 6.0], dtype=np.float64),
                )
            },
            horizons=[("Month 1", 2), ("Year 1", 4)],
        )

        self.assertEqual(rows[0]["horizon"], "Month 1")
        self.assertAlmostEqual(float(rows[0]["OMM cumulative"]), 6.0)
        self.assertAlmostEqual(float(rows[0]["OMM normalized"]), 3.0 / 1.7)
        self.assertIsNone(rows[1]["OMM cumulative"])
        self.assertIsNone(rows[1]["OMM normalized"])

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


if __name__ == "__main__":
    unittest.main()
