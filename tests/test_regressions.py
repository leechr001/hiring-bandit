import os
import random
import sys
import tempfile
import unittest
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
from policies import AgrawalHegdeTeneketzisPolicy
from simulation import (
    compute_optimistic_hire_auto_gamma,
    make_policy,
    optimistic_hire_regret_bound,
    simulate,
)


class RegressionTests(unittest.TestCase):
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
            policies=["ucb", "epsilon-greedy"],
            **kwargs,
        )
        _, results_ba = simulate(
            policies=["epsilon-greedy", "ucb"],
            **kwargs,
        )

        np.testing.assert_allclose(results_ab["ucb"][0], results_ba["ucb"][0])
        np.testing.assert_allclose(results_ab["ucb"][1], results_ba["ucb"][1])
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

        self.assertGreater(policy.cfg.gamma, 0.0)
        self.assertAlmostEqual(policy.cfg.gamma, expected_gamma)

        base_bound = optimistic_hire_regret_bound(policy.cfg.gamma, **kwargs)
        lower_bound = optimistic_hire_regret_bound(policy.cfg.gamma * 0.8, **kwargs)
        upper_bound = optimistic_hire_regret_bound(policy.cfg.gamma * 1.2, **kwargs)

        self.assertLessEqual(base_bound, lower_bound)
        self.assertLessEqual(base_bound, upper_bound)


if __name__ == "__main__":
    unittest.main()
