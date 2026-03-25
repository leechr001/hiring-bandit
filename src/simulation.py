from bandit_environment import TemporaryHiringBanditEnv
from policies import (
    EpsilonGreedyHiringPolicy,
    VanillaUCBHiringPolicy,
    AgrawalHegdeTeneketzisPolicy
)

from hiring_ucb import HiringUCBPolicy
from samplers import (
    make_bernoulli_samplers, 
    make_uniform_delay_sampler, 
    make_adversarial_delay
)

import random
from typing import Sequence, Dict, Tuple, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Simulation and regret
# ----------------------------

def _seeded_rng(seed: int, stream: str) -> random.Random:
    """Create an independent deterministic RNG stream for one episode component."""
    return random.Random(f"{seed}:{stream}")

def make_policy(
    policy_name: str,
    *,
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
    rng: random.Random,
    epsilon: float = 0.1,
    ucb_coef: float = 2.0,
    gamma: float = 0.5,
):
    """
    Factory to build a policy object consistent with the simulation interface.

    Supported names (case- and spacing-insensitive after lowercasing/stripping):
      - "epsilon-greedy", "eps", "epsilon", "egreedy"
      - "ucb", "vanilla-ucb", "vanilla ucb"
      - "hiring-ucb", "hiring ucb", "paper", "algorithm-1"
      - "agrawalhegdeteneketzis", "classic", "rarely-switch", "round-robin", "aht"
    """
    name = policy_name.lower().strip()

    # Naive baselines
    if name in {"eps", "epsilon", "epsilon-greedy", "egreedy"}:
        return EpsilonGreedyHiringPolicy(k=k, m=m, epsilon=epsilon, rng=rng)

    elif name in {"ucb", "vanilla-ucb", "vanilla ucb"}:
        return VanillaUCBHiringPolicy(k=k, m=m, alpha=ucb_coef, rng=rng)
    
    # Construct UCB with different bijections for experiements
    elif name in {"ucb-rm"}:
        return VanillaUCBHiringPolicy(k=k, m=m, alpha=ucb_coef, bijection_name='oracle-match', rng=rng)
    
    elif name in {"ucb-rmm"}:
        return VanillaUCBHiringPolicy(k=k, m=m, alpha=ucb_coef, bijection_name='oracle-mismatch', rng=rng)

    # Adaptive batching algorithm from the paper
    elif name in {"hiring-ucb", "hiring ucb", "paper", "algorithm-1"}:
        return HiringUCBPolicy(k=k, m=m, gamma=gamma, horizon=T, rng=rng)
    
    elif name in {"hiring-ucb-gamma-1"}:
        return HiringUCBPolicy(k=k, m=m, gamma=(c+omega_max)**2 * m, horizon=T, rng=rng)
    
    elif name in {"hiring-ucb-gamma-2"}:
        return HiringUCBPolicy(k=k, m=m, gamma=(c+omega_max) * m, horizon=T, rng=rng)
    
    elif name in {"hiring-ucb-gamma-3"}:
        return HiringUCBPolicy(k=k, m=m, gamma=c * m, horizon=T, rng=rng)
    
    # Paper by Agrawal, Hedge, and Teneketzis 
    elif name in {"agrawalhegdeteneketzis", "classic", "rarely-switch", "round-robin", "aht"}:
        return AgrawalHegdeTeneketzisPolicy(k=k, m=m, rng=rng)

    else:
        raise ValueError(f"Unknown policy_name: {policy_name}")

def run_episode(
    *,
    policy_name: str,
    k: int,
    m: int,
    means: Sequence[float],
    reward_samplers: Sequence[Callable],
    delay_sampler: Callable,
    T: int,
    epsilon: float,
    gamma: float,
    c: float,
    omega_max: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)

    initial = rng.sample(list(range(1, k + 1)), m)

    env = TemporaryHiringBanditEnv(
        k=k,
        m=m,
        reward_samplers=reward_samplers,
        delay_sampler=delay_sampler,
        c=c,
        omega_max=omega_max,
        rng=rng,
        true_means=means,
        initial_workforce=initial,
    )

    policy = make_policy(
        policy_name,
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=omega_max,
        rng=rng,
        epsilon=epsilon,
        ucb_coef=2.0,  # adjust if you want different default exploration strength
        gamma=gamma
    )

    oracle_per_period = env.optimal_expected_reward()
    if oracle_per_period is None:
        raise ValueError("True means required for regret simulation.")

    regret_increments = np.zeros(T, dtype=np.float64)

    for _ in range(T):
        replacements = policy.act(env)
        env.validate_replacements(replacements)

        obs, total_reward, cost, info = env.step(replacements)

        policy.update(info["individual_rewards"])

        active_now = info["active_set"]
        active_expected = sum(means[i - 1] for i in active_now)

        # Pseudo-regret increment: gap to oracle + switching cost
        regret_increments[env.t - 2] = (oracle_per_period - active_expected) + cost

    return np.cumsum(regret_increments)

def simulate(
    *,
    policies: Sequence[str],
    k: int = 10,
    m: int = 3,
    T: int = 2000,
    means: Optional[Sequence[float]] = None,
    reward_samplers: Optional[Sequence[Callable]] = None,
    delay_sampler: Optional[Callable] = None,
    reward_sampler_factory: Optional[Callable[[int], Sequence[Callable]]] = None,
    delay_sampler_factory: Optional[Callable[[int], Callable]] = None,
    epsilon: float = 0.1,
    gamma: float = 0.5,
    c: float = 1,
    omega_max: int = 3,
    n_runs: int = 50,
    seed0: int = 0,
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    # Simple mean profile with a clear top-m
    rng = random.Random(999)

    if reward_samplers is not None and reward_sampler_factory is not None:
        raise ValueError("Pass either reward_samplers or reward_sampler_factory, not both.")
    if delay_sampler is not None and delay_sampler_factory is not None:
        raise ValueError("Pass either delay_sampler or delay_sampler_factory, not both.")

    if means is None:
        means = sorted([rng.uniform(0.1, 0.9) for _ in range(k)], reverse=True)
    
    if reward_samplers is None and reward_sampler_factory is None:
        reward_sampler_factory = lambda seed: make_bernoulli_samplers(
            means,
            _seeded_rng(seed, "reward"),
        )
    
    if delay_sampler is None and delay_sampler_factory is None:
        delay_sampler_factory = lambda seed: make_uniform_delay_sampler(
            omega_max,
            _seeded_rng(seed, "delay"),
        )

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pname in policies:
        curves = []
        for r in range(n_runs):
            episode_seed = seed0 + r
            episode_reward_samplers = reward_samplers
            if episode_reward_samplers is None:
                if reward_sampler_factory is None:
                    raise ValueError("reward_sampler_factory must be set when reward_samplers is None.")
                episode_reward_samplers = reward_sampler_factory(episode_seed)

            episode_delay_sampler = delay_sampler
            if episode_delay_sampler is None:
                if delay_sampler_factory is None:
                    raise ValueError("delay_sampler_factory must be set when delay_sampler is None.")
                episode_delay_sampler = delay_sampler_factory(episode_seed)

            curves.append(
                run_episode(
                    policy_name=pname,
                    k=k,
                    m=m,
                    means=means,
                    reward_samplers=episode_reward_samplers,
                    delay_sampler=episode_delay_sampler,
                    T=T,
                    epsilon=epsilon,
                    gamma=gamma,
                    c=c,
                    omega_max=omega_max,
                    seed=episode_seed,
                )
            )
        curves = np.stack(curves, axis=0)
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        results[pname] = (mean_curve, std_curve)

    return means, results


"""
Compare policies
"""
def run_policy_comparisons(
    *,
    policies: Sequence[str],
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
    means: Sequence[float],
    delay_process_name: str = 'uniform',
    labels: Optional[Sequence[str]] = None,
    n_runs: int = 20,
    base_seed: int = 12345,  
    title: str = 'Regret by Policy'
) -> None:
    
    if labels is None:
        labels = policies

    if delay_process_name in {'uniform', 'random', 'stochastic', 'iid'}:
        delay_sampler_factory = lambda seed: make_uniform_delay_sampler(
            omega_max,
            _seeded_rng(seed, "delay"),
        )
    elif delay_process_name in {'adversarial', 'wc'}:
        delay_sampler_factory = lambda seed: make_adversarial_delay(
            means=means,
            omega_max=omega_max,
        )
    else:
        raise ValueError(
            "delay_process_name must be one of: 'uniform', 'random', "
            "'stochastic', 'iid', 'adversarial', or 'wc'."
        )

    # --- Run simulation ---
    means, results = simulate(
        policies=policies,
        k=k,
        m=m,
        T=T,
        means=means,
        delay_sampler_factory=delay_sampler_factory,
        c=c,
        omega_max=omega_max,
        n_runs=n_runs,
        seed0=base_seed
    )


    print("True means (sorted high to low):")
    print(np.round(means, 3))

    plt.figure()
    x = np.arange(T)

    for plabel, pname in zip(labels, policies):
        mean_curve, std_curve = results[pname]
        plt.plot(x, mean_curve, label=plabel)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
        )

    plt.xlabel("Time t")
    plt.ylabel("Cumulative pseudo-regret")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


"""
Define functions to sweep parameters
"""

def run_gamma_sweep(
    *,
    k: int,
    m: int,
    T: int,
    means: Sequence[float],
    gammas: Sequence[float],
    c: float,
    omega_max: int,
    n_runs: int = 20,
    base_seed: int = 12345,
) -> None:
    """
    Run average regret trajectories for a multiple values of gamma.
    Can only be used with hiring-ucb for obvious reaons
    """

    def label_fun(gamma: float, c: float, omega: int) -> str:
        return rf"$\gamma={gamma:.2f}$, $c={c:.2f}$, $\omega_\max = {omega}$"

    plt.figure(figsize=(8, 5))
    time_axis = np.arange(1, T + 1)

    for _, val in enumerate(gammas):
        
        gamma = float(val)

        def label_fun(gamma: float, c: float, omega: int) -> str:
            return rf"$\gamma={gamma:.2f}$"

        _, results = simulate(
                policies=['hiring-ucb'],
                k=k,
                m=m,
                T=T,
                gamma=gamma,
                c=c,
                omega_max=omega_max,
                means=means,
                seed0=base_seed,
                n_runs=20
            )
        
        mean_curve, std_curve = results["hiring-ucb"]

        label = label_fun(gamma=gamma, c=c, omega=omega_max)

        plt.plot(time_axis, mean_curve, linewidth=2, label=label)
        plt.fill_between(
            time_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
            linewidth=0,
        )

    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.ylim(0,2500)
    plt.title(rf"Average regret over {n_runs} runs with $c = {c}$ and $\omega_\max = {omega_max}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_c_sweep(
    *,
    k: int,
    m: int,
    T: int,
    policy_name: str = 'hiring-ucb',
    means: Sequence[float],
    c_values: Sequence[float],
    omega_max: int,
    n_runs: int = 20,
    base_seed: int = 12345,
) -> None:
    """
    Run average regret trajectories for a multiple values of gamma.
    Can only be used with hiring-ucb for obvious reaons
    """

    def label_fun(c: float, omega: int) -> str:
        return rf"$c={c:.2f}$, $\omega_\max = {omega}$"

    plt.figure(figsize=(8, 5))
    time_axis = np.arange(1, T + 1)

    for _, val in enumerate(c_values):
        
        c = float(val)

        def label_fun(c: float, omega: int) -> str:
            return rf"$c={c:.2f}$"

        _, results = simulate(
                policies=[policy_name],
                k=k,
                m=m,
                T=T,
                c=val,
                omega_max=omega_max,
                means=means,
                seed0=base_seed,
                n_runs=20
            )
        
        mean_curve, std_curve = results[policy_name]

        label = label_fun(c=val, omega=omega_max)

        plt.plot(time_axis, mean_curve, linewidth=2, label=label)
        plt.fill_between(
            time_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
            linewidth=0,
        )

    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.ylim(0,3500)
    plt.title(rf"Average regret over {n_runs} runs with $\omega_\max = {omega_max}$")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_omega_sweep(
    *,
    k: int,
    m: int,
    T: int,
    policy_name: str = 'hiring-ucb',
    means: Sequence[float],
    c: float,
    omega_max_values: Sequence[float],
    omega_process: str = 'stochastic',
    n_runs: int = 20,
    base_seed: int = 12345,
    y_up_lim: int = 2500
) -> None:
    """
    Run average regret trajectories for a multiple values of gamma.
    Can only be used with hiring-ucb for obvious reaons

    pick between delay processes: 'stochastic', 'adversarial'
    """

    rng = random.Random(base_seed)

    def label_fun(c: float, omega: int) -> str:
        return rf"$c={c:.2f}$, $\omega_\max = {omega}$"

    plt.figure(figsize=(8, 5))
    time_axis = np.arange(1, T + 1)

    for _, val in enumerate(omega_max_values):
        
        omega = int(val)

        if omega_process == 'stochastic':
            delay_sampler = make_uniform_delay_sampler(omega, rng)
        elif omega_process == 'adversarial':
            delay_sampler = make_adversarial_delay(means, omega)
        else:
            raise ValueError("delays process not one of: 'stochastic', 'adversarial'")

        def label_fun(c: float, omega: int) -> str:
            return rf"$\omega_\max={omega:.2f}$"

        _, results = simulate(
                policies=[policy_name],
                k=k,
                m=m,
                T=T,
                c=c,
                omega_max=omega,
                delay_sampler=delay_sampler,
                means=means,
                seed0=base_seed,
                n_runs=20
            )
        
        mean_curve, std_curve = results[policy_name]

        label = label_fun(c=c, omega=omega)

        plt.plot(time_axis, mean_curve, linewidth=2, label=label)
        plt.fill_between(
            time_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
            linewidth=0,
        )

    plt.xlabel("t")
    plt.ylabel("Cumulative regret")
    plt.ylim(0,y_up_lim)
    plt.title(rf"Average regret over {n_runs} runs with $c = {c}$ and {omega_process} delay process.")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
