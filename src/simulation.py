from bandit_environment import TemporaryHiringBanditEnv
from policies import (
    EpsilonGreedyHiringPolicy,
    OMM,
    AgrawalHegdeTeneketzisPolicy
)

from hiring_ucb import HiringUCBPolicy
from samplers import (
    make_bernoulli_samplers, 
    make_calendar_adversarial_delay,
    make_calendar_delay_sampler,
    make_uniform_delay_sampler, 
    make_adversarial_delay
)

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Simulation and regret
# ----------------------------


@dataclass(frozen=True)
class ExperimentSeries:
    policy_name: str
    label: Optional[str] = None
    sim_kwargs: Mapping[str, Any] = field(default_factory=dict)
    plot_kwargs: Mapping[str, Any] = field(default_factory=dict)
    band_alpha: float = 0.15


@dataclass(frozen=True)
class HighlightPoint:
    value: float
    label: str
    plot_kwargs: Mapping[str, Any] = field(default_factory=dict)

def _seeded_rng(seed: int, stream: str) -> random.Random:
    """Create an independent deterministic RNG stream for one episode component."""
    return random.Random(f"{seed}:{stream}")


def _merge_sim_kwargs(
    base_kwargs: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(base_kwargs)
    if overrides:
        merged.update(dict(overrides))
    return merged


def make_delay_sampler_factory(
    delay_process_name: str,
    *,
    means: Sequence[float],
    omega_max: int,
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
) -> Callable[[int], Callable]:
    if delay_process_name in {"uniform", "random", "stochastic", "iid"}:
        return lambda seed: make_uniform_delay_sampler(
            omega_max,
            _seeded_rng(seed, "delay"),
        )
    if delay_process_name in {"adversarial", "wc"}:
        return lambda seed: make_adversarial_delay(
            means=means,
            omega_max=omega_max,
        )
    if delay_process_name in {"calendar-adversarial", "calendar-wc"}:
        if calendar_frequency is None:
            raise ValueError("calendar_frequency is required for calendar delays.")

        return lambda seed: make_calendar_adversarial_delay(
            means=means,
            omega_max=omega_max,
            frequency=calendar_frequency,
        )
    if delay_process_name in {"calendar", "calendar-unif", "calendar-geom"}:
        if calendar_frequency is None:
            raise ValueError("calendar_frequency is required for calendar delays.")

        distribution = calendar_distribution
        if delay_process_name == "calendar-unif":
            distribution = "unif"
        elif delay_process_name == "calendar-geom":
            distribution = "geom"

        return lambda seed: make_calendar_delay_sampler(
            omega_max,
            frequency=calendar_frequency,
            distribution=distribution,
            geom_p=calendar_geom_p,
            rng=_seeded_rng(seed, "delay"),
        )

    raise ValueError(
        "delay_process_name must be one of: 'uniform', 'random', "
        "'stochastic', 'iid', 'adversarial', 'wc', 'calendar', "
        "'calendar-unif', 'calendar-geom', 'calendar-adversarial', "
        "or 'calendar-wc'."
    )


def run_series_simulations(
    *,
    series: Sequence[ExperimentSeries],
    simulate_kwargs: Mapping[str, Any],
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    means_out: Optional[Sequence[float]] = None
    collected: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for spec in series:
        current_kwargs = _merge_sim_kwargs(simulate_kwargs, spec.sim_kwargs)
        means_out, results = simulate(
            policies=[spec.policy_name],
            **current_kwargs,
        )
        collected[spec.label or spec.policy_name] = results[spec.policy_name]

    if means_out is None:
        raise ValueError("At least one series is required.")

    return means_out, collected


def plot_regret_series(
    *,
    series: Sequence[ExperimentSeries],
    simulate_kwargs: Mapping[str, Any],
    title: str = "Regret by Policy",
    xlabel: str = "Time t",
    ylabel: str = "Cumulative pseudo-regret",
    figure_kwargs: Optional[Mapping[str, Any]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
    precomputed: Optional[Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    if precomputed is None:
        means, results = run_series_simulations(series=series, simulate_kwargs=simulate_kwargs)
    else:
        means, results = precomputed

    T = int(simulate_kwargs["T"])
    x = np.arange(T)

    plt.figure(**dict(figure_kwargs or {}))

    for spec in series:
        label = spec.label or spec.policy_name
        mean_curve, std_curve = results[label]
        line_kwargs = dict(spec.plot_kwargs)
        plt.plot(x, mean_curve, label=label, **line_kwargs)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=spec.band_alpha,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    merged_grid_kwargs: Dict[str, Any] = {}
    if grid_kwargs:
        merged_grid_kwargs.update(dict(grid_kwargs))
    plt.grid(True, **merged_grid_kwargs)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return means, results


def evaluate_final_regret_sweep(
    *,
    policy_name: str,
    parameter_name: str,
    values: Sequence[float],
    simulate_kwargs: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(values, dtype=float)
    mean_finals = []
    std_finals = []

    for value in values:
        current_kwargs = _merge_sim_kwargs(simulate_kwargs, {parameter_name: value})
        _, results = simulate(
            policies=[policy_name],
            **current_kwargs,
        )
        mean_curve, std_curve = results[policy_name]
        mean_finals.append(float(mean_curve[-1]))
        std_finals.append(float(std_curve[-1]))

    return (
        x_values,
        np.asarray(mean_finals, dtype=float),
        np.asarray(std_finals, dtype=float),
    )


def plot_final_regret_sweep(
    *,
    policy_name: str,
    parameter_name: str,
    values: Sequence[float],
    simulate_kwargs: Mapping[str, Any],
    title: str,
    xlabel: str,
    ylabel: str,
    highlight_points: Optional[Sequence[HighlightPoint]] = None,
    figure_kwargs: Optional[Mapping[str, Any]] = None,
    errorbar_kwargs: Optional[Mapping[str, Any]] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
    xscale: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values, mean_finals, std_finals = evaluate_final_regret_sweep(
        policy_name=policy_name,
        parameter_name=parameter_name,
        values=values,
        simulate_kwargs=simulate_kwargs,
    )

    order = np.argsort(x_values)
    x_sorted = x_values[order]
    mean_sorted = mean_finals[order]
    std_sorted = std_finals[order]

    plt.figure(**dict(figure_kwargs or {}))
    base_errorbar_kwargs = {
        "fmt": "o-",
        "linewidth": 1,
        "capsize": 2,
        "label": "Average final cumulative regret",
    }
    if errorbar_kwargs:
        base_errorbar_kwargs.update(dict(errorbar_kwargs))
    plt.errorbar(
        x_sorted,
        mean_sorted,
        yerr=std_sorted,
        **base_errorbar_kwargs,
    )

    for point in highlight_points or []:
        matches = np.where(np.isclose(x_sorted, point.value))[0]
        if len(matches) == 0:
            continue

        idx = int(matches[0])
        point_kwargs = {
            "s": 120,
            "edgecolor": "black",
            "linewidth": 1.5,
            "label": point.label,
        }
        point_kwargs.update(dict(point.plot_kwargs))
        plt.scatter(
            x_sorted[idx],
            mean_sorted[idx],
            **point_kwargs,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xscale is not None:
        plt.xscale(xscale)

    merged_grid_kwargs = {"visible": True, "linestyle": "--", "alpha": 0.5}
    if grid_kwargs:
        merged_grid_kwargs.update(dict(grid_kwargs))
    plt.grid(**merged_grid_kwargs)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return x_sorted, mean_sorted, std_sorted


def _validate_optimistic_hire_auto_gamma_inputs(
    *,
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")
    if m <= 0 or m >= k:
        raise ValueError("auto gamma requires 1 <= m < k.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if c < 0:
        raise ValueError("c must be non-negative.")
    if omega_max < 0:
        raise ValueError("omega_max must be non-negative.")


def optimistic_hire_regret_bound(
    gamma: float,
    *,
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
) -> float:
    """Upper bound used to select an automatic gamma for Optimistic-Hire."""
    _validate_optimistic_hire_auto_gamma_inputs(
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=omega_max,
    )
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    log_kmt = math.log(k * m * T)
    log_ratio = math.log((m * T) / ((k - m) * log_kmt))

    leading_term = 5.0 * math.sqrt(m * (k - m) * T * log_kmt)
    switching_term = c * (k - m) * math.log(T) + c * (k - m)
    reciprocal_gamma_term = (
        (8.0 * math.sqrt(k) / (m * gamma))
        + (4.0 * k / (m * gamma))
        + (21.0 / (m * gamma * k))
    )
    sqrt_gamma_term = (
        2.0
        * (k - m)
        * math.sqrt(gamma * 4.0 * log_kmt)
        * (1.0 + 0.5 * log_ratio)
    )
    inverse_sqrt_gamma_term = (
        omega_max * m * math.sqrt((k * T) / gamma)
        + c * m * math.sqrt((k * T) / gamma)
    )

    return (
        leading_term
        + switching_term
        + k * (gamma + 1.0)
        + reciprocal_gamma_term
        + sqrt_gamma_term
        + inverse_sqrt_gamma_term
        + omega_max * m * k
        + c * m * k
    )


def compute_optimistic_hire_auto_gamma(
    *,
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
) -> float:
    """Numerically minimize the Optimistic-Hire gamma bound over gamma > 0."""
    _validate_optimistic_hire_auto_gamma_inputs(
        k=k,
        m=m,
        T=T,
        c=c,
        omega_max=omega_max,
    )

    log_kmt = math.log(k * m * T)
    log_ratio = math.log((m * T) / ((k - m) * log_kmt))

    reciprocal_gamma_coeff = (
        (8.0 * math.sqrt(k) / m)
        + (4.0 * k / m)
        + (21.0 / (m * k))
    )
    sqrt_gamma_coeff = (
        2.0
        * (k - m)
        * math.sqrt(4.0 * log_kmt)
        * (1.0 + 0.5 * log_ratio)
    )
    inverse_sqrt_gamma_coeff = (omega_max + c) * m * math.sqrt(k * T)

    # With x = sqrt(gamma), the derivative is:
    #   2k x + sqrt_gamma_coeff - inverse_sqrt_gamma_coeff / x^2
    #   - 2 * reciprocal_gamma_coeff / x^3 = 0
    # which becomes the quartic
    #   2k x^4 + sqrt_gamma_coeff x^3 - inverse_sqrt_gamma_coeff x - 2B = 0.
    roots = np.roots(
        [
            2.0 * k,
            sqrt_gamma_coeff,
            0.0,
            -inverse_sqrt_gamma_coeff,
            -2.0 * reciprocal_gamma_coeff,
        ]
    )

    candidate_gammas = [
        float(root.real) ** 2
        for root in roots
        if abs(root.imag) < 1e-9 and root.real > 0
    ]

    if not candidate_gammas:
        search_grid = np.logspace(-8, 8, num=400)
        candidate_gammas = [float(gamma) for gamma in search_grid]

    return min(
        candidate_gammas,
        key=lambda gamma: optimistic_hire_regret_bound(
            gamma,
            k=k,
            m=m,
            T=T,
            c=c,
            omega_max=omega_max,
        ),
    )

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
    gamma: float | str = 0.5,
):
    """
    Factory to build a policy object consistent with the simulation interface.

    Supported names (case- and spacing-insensitive after lowercasing/stripping):
      - "epsilon-greedy", "eps", "epsilon", "egreedy"
      - "omm", "optimistic-matroid-maximization", "optimistic matroid maximization"
      - "optimistic-hire", "optimistic hire", "optimistic-hire-auto", "paper", "algorithm-1"
      - "agrawalhegdeteneketzis", "classic", "rarely-switch", "round-robin", "aht"
    """
    name = policy_name.lower().strip()

    # Naive baselines
    if name in {"eps", "epsilon", "epsilon-greedy", "egreedy"}:
        return EpsilonGreedyHiringPolicy(k=k, m=m, epsilon=epsilon, rng=rng)

    elif name in {
        "omm",
        "optimistic-matroid-maximization",
        "optimistic matroid maximization",
    }:
        return OMM(k=k, m=m, alpha=ucb_coef, rng=rng)
    
    # Construct OMM with different bijections for experiments.
    elif name in {"omm-rm"}:
        return OMM(k=k, m=m, alpha=ucb_coef, bijection_name='oracle-match', rng=rng)
    
    elif name in {"omm-rmm"}:
        return OMM(k=k, m=m, alpha=ucb_coef, bijection_name='oracle-mismatch', rng=rng)

    # Adaptive batching algorithm from the paper
    elif name in {"optimistic-hire", "optimistic hire", "optimistic-hire-auto", "paper", "algorithm-1"}:
        resolved_gamma: float
        if name == "optimistic-hire-auto" or gamma == "auto":
            resolved_gamma = compute_optimistic_hire_auto_gamma(
                k=k,
                m=m,
                T=T,
                c=c,
                omega_max=omega_max,
            )
        elif isinstance(gamma, str):
            raise ValueError("gamma must be a positive float or 'auto'.")
        else:
            resolved_gamma = float(gamma)
        return HiringUCBPolicy(k=k, m=m, gamma=resolved_gamma, horizon=T, rng=rng)
    
    elif name in {"optimistic-hire-gamma-1"}:
        return HiringUCBPolicy(k=k, m=m, gamma=(c+omega_max)**2 * m, horizon=T, rng=rng)
    
    elif name in {"optimistic-hire-gamma-2"}:
        return HiringUCBPolicy(k=k, m=m, gamma=(c+omega_max) * m, horizon=T, rng=rng)
    
    elif name in {"optimistic-hire-gamma-3"}:
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
    gamma: float | str,
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

        obs, total_reward, cost, feedback = env.step(replacements)

        policy.update(feedback)

        active_now = feedback.active_set
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
    gamma: float | str = 0.5,
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
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
    labels: Optional[Sequence[str]] = None,
    n_runs: int = 20,
    base_seed: int = 12345,  
    title: str = 'Regret by Policy'
) -> None:
    if labels is None:
        labels = policies

    delay_sampler_factory = make_delay_sampler_factory(
        delay_process_name,
        means=means,
        omega_max=omega_max,
        calendar_frequency=calendar_frequency,
        calendar_distribution=calendar_distribution,
        calendar_geom_p=calendar_geom_p,
    )

    series = [
        ExperimentSeries(policy_name=pname, label=plabel)
        for plabel, pname in zip(labels, policies)
    ]
    simulate_kwargs = {
        "k": k,
        "m": m,
        "T": T,
        "means": means,
        "delay_sampler_factory": delay_sampler_factory,
        "c": c,
        "omega_max": omega_max,
        "n_runs": n_runs,
        "seed0": base_seed,
    }

    means, results = run_series_simulations(
        series=series,
        simulate_kwargs=simulate_kwargs,
    )

    print("True means (sorted high to low):")
    print(np.round(means, 3))
    plot_regret_series(
        series=series,
        simulate_kwargs=simulate_kwargs,
        title=title,
        precomputed=(means, results),
    )


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
    series = [
        ExperimentSeries(
            policy_name="optimistic-hire",
            label=rf"$\gamma={float(gamma):.2f}$",
            sim_kwargs={"gamma": float(gamma)},
            plot_kwargs={"linewidth": 2},
        )
        for gamma in gammas
    ]
    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "c": c,
            "omega_max": omega_max,
            "n_runs": n_runs,
            "seed0": base_seed,
        },
        title=rf"Average regret over {n_runs} runs with $c = {c}$ and $\omega_\max = {omega_max}$",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, 2500),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
    )

def run_c_sweep(
    *,
    k: int,
    m: int,
    T: int,
    policy_name: str = 'optimistic-hire',
    means: Sequence[float],
    c_values: Sequence[float],
    omega_max: int,
    n_runs: int = 20,
    base_seed: int = 12345,
) -> None:
    series = [
        ExperimentSeries(
            policy_name=policy_name,
            label=rf"$c={float(value):.2f}$",
            sim_kwargs={"c": float(value)},
            plot_kwargs={"linewidth": 2},
        )
        for value in c_values
    ]
    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "omega_max": omega_max,
            "n_runs": n_runs,
            "seed0": base_seed,
        },
        title=rf"Average regret over {n_runs} runs with $\omega_\max = {omega_max}$",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, 3500),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
    )

def run_omega_sweep(
    *,
    k: int,
    m: int,
    T: int,
    policy_name: str = 'optimistic-hire',
    means: Sequence[float],
    c: float,
    omega_max_values: Sequence[float],
    omega_process: str = 'stochastic',
    n_runs: int = 20,
    base_seed: int = 12345,
    y_up_lim: int = 2500
) -> None:
    if omega_process not in {"stochastic", "adversarial"}:
        raise ValueError("delays process not one of: 'stochastic', 'adversarial'")

    delay_name = "stochastic" if omega_process == "stochastic" else "adversarial"
    series = [
        ExperimentSeries(
            policy_name=policy_name,
            label=rf"$\omega_\max={int(value):.2f}$",
            sim_kwargs={
                "omega_max": int(value),
                "delay_sampler_factory": make_delay_sampler_factory(
                    delay_name,
                    means=means,
                    omega_max=int(value),
                ),
            },
            plot_kwargs={"linewidth": 2},
        )
        for value in omega_max_values
    ]
    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "c": c,
            "n_runs": n_runs,
            "seed0": base_seed,
        },
        title=rf"Average regret over {n_runs} runs with $c = {c}$ and {omega_process} delay process.",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, y_up_lim),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
    )
