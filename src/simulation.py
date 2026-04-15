from bandit_environment import TemporaryHiringBanditEnv
from policies import (
    EpsilonGreedyHiringPolicy,
    OMM,
    SemiAnnualReview,
    Threshold,
    WorkTrial,
    AgrawalHegdeTeneketzisPolicy
)

from choose_target import ChooseTargetFrontierSizeRecord
from optimistic_hire import OptimisticHire
from samplers import (
    make_bernoulli_samplers, 
    make_calendar_adversarial_delay,
    make_calendar_delay_sampler,
    make_truncated_normal_samplers,
    make_uniform_delay_sampler, 
    make_adversarial_delay
)

from dataclasses import dataclass, field, replace
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import random
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


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
    delay_lower: int = 1,
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
) -> Callable[[int], Callable]:
    if delay_process_name in {"uniform", "random", "stochastic", "iid"}:
        return lambda seed: make_uniform_delay_sampler(
            omega_max=omega_max,
            rng=_seeded_rng(seed, "delay"),
            delay_lower=delay_lower,
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


def make_reward_sampler_factory(
    reward_process_name: str,
    *,
    means: Sequence[float],
    reward_stddev: float = 0.1,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
) -> Callable[[int], Sequence[Callable]]:
    if reward_process_name in {"bernoulli", "binary"}:
        return lambda seed: make_bernoulli_samplers(
            means,
            _seeded_rng(seed, "reward"),
        )
    if reward_process_name in {"truncated-normal", "truncnorm", "tn"}:
        return lambda seed: make_truncated_normal_samplers(
            means,
            _seeded_rng(seed, "reward"),
            stddev=reward_stddev,
            lower=reward_lower,
            upper=reward_upper,
        )

    raise ValueError(
        "reward_process_name must be one of: 'bernoulli', 'binary', "
        "'truncated-normal', 'truncnorm', or 'tn'."
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


def _average_regret_results(
    means: Sequence[float],
    m: int,
    results: Mapping[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    oracle_reward = float(sum(sorted((float(mu) for mu in means), reverse=True)[:m]))
    if oracle_reward <= 0.0:
        raise ValueError("mu(A*) must be positive to normalize regret by T * mu(A*).")

    averaged: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, (mean_curve, std_curve) in results.items():
        periods = np.arange(1, len(mean_curve) + 1, dtype=np.float64)
        denom = periods * oracle_reward
        averaged[label] = (
            np.asarray(mean_curve, dtype=np.float64) / denom,
            np.asarray(std_curve, dtype=np.float64) / denom,
        )
    return averaged


def _default_series_style(index: int) -> Dict[str, Any]:
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "marker": markers[index % len(markers)],
        "markevery": 0.08,
        "linewidth": 2,
        "markersize": 5,
    }


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
    y_axis_percent: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    precomputed: Optional[Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    if precomputed is None:
        means, results = run_series_simulations(series=series, simulate_kwargs=simulate_kwargs)
    else:
        means, results = precomputed

    T = int(simulate_kwargs["T"])
    x = np.arange(T)

    plt.figure(**dict(figure_kwargs or {}))

    for idx, spec in enumerate(series):
        label = spec.label or spec.policy_name
        mean_curve, std_curve = results[label]
        line_kwargs = _default_series_style(idx)
        line_kwargs.update(dict(spec.plot_kwargs))
        plt.plot(x, mean_curve, label=label, **line_kwargs)
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=spec.band_alpha,
            color=line_kwargs.get("color"),
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if y_axis_percent:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    merged_grid_kwargs: Dict[str, Any] = {}
    if grid_kwargs:
        merged_grid_kwargs.update(dict(grid_kwargs))
    plt.grid(True, **merged_grid_kwargs)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    return means, results


def plot_average_regret_series(
    *,
    series: Sequence[ExperimentSeries],
    simulate_kwargs: Mapping[str, Any],
    title: str = "Normalized Loss by Policy",
    xlabel: str = "Time t",
    ylabel: str = r"Normalized loss",
    figure_kwargs: Optional[Mapping[str, Any]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    grid_kwargs: Optional[Mapping[str, Any]] = None,
    y_axis_percent: bool = False,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    precomputed: Optional[Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    if precomputed is None:
        means, results = run_series_simulations(series=series, simulate_kwargs=simulate_kwargs)
    else:
        means, results = precomputed

    averaged_results = _average_regret_results(
        means,
        int(simulate_kwargs["m"]),
        results,
    )
    plot_regret_series(
        series=series,
        simulate_kwargs=simulate_kwargs,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figure_kwargs=figure_kwargs,
        ylim=ylim,
        grid_kwargs=grid_kwargs,
        y_axis_percent=y_axis_percent,
        save_path=save_path,
        show_plot=show_plot,
        precomputed=(means, averaged_results),
    )
    return means, averaged_results


def build_planning_horizon_regret_table(
    *,
    series: Sequence[ExperimentSeries],
    means: Sequence[float],
    m: int,
    results: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    horizons: Sequence[Tuple[str, int]],
) -> Sequence[Dict[str, Any]]:
    normalized_results = _average_regret_results(means, m, results)
    rows: list[Dict[str, Any]] = []

    for horizon_label, horizon_period in horizons:
        row: Dict[str, Any] = {
            "horizon": horizon_label,
            "period": horizon_period,
        }
        for spec in series:
            label = spec.label or spec.policy_name
            mean_curve, _ = results[label]
            norm_curve, _ = normalized_results[label]

            if horizon_period < 1 or horizon_period > len(mean_curve):
                row[f"{label} cumulative"] = None
                row[f"{label} normalized"] = None
                continue

            idx = horizon_period - 1
            row[f"{label} cumulative"] = float(mean_curve[idx])
            row[f"{label} normalized"] = float(norm_curve[idx])
        rows.append(row)

    return rows


def print_planning_horizon_regret_table(
    *,
    series: Sequence[ExperimentSeries],
    means: Sequence[float],
    m: int,
    results: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    horizons: Sequence[Tuple[str, int]],
) -> Sequence[Dict[str, Any]]:
    rows = build_planning_horizon_regret_table(
        series=series,
        means=means,
        m=m,
        results=results,
        horizons=horizons,
    )

    header = ["Horizon"]
    for spec in series:
        label = spec.label or spec.policy_name
        header.extend([f"{label} Regret", f"{label} Loss %"])

    table: list[list[str]] = [header]
    for row in rows:
        formatted = [str(row["horizon"])]
        for spec in series:
            label = spec.label or spec.policy_name
            cumulative = row[f"{label} cumulative"]
            normalized = row[f"{label} normalized"]
            formatted.append("n/a" if cumulative is None else f"{float(cumulative):,.1f}")
            formatted.append("n/a" if normalized is None else f"{100.0 * float(normalized):.2f}%")
        table.append(formatted)

    widths = [
        max(len(row[col_idx]) for row in table)
        for col_idx in range(len(table[0]))
    ]
    for row_idx, row in enumerate(table):
        line = " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        print(line)
        if row_idx == 0:
            print("-+-".join("-" * width for width in widths))

    return rows


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
        "zorder": 2,
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
            "zorder": 5,
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


def _optimistic_hire_bound_components(
    *,
    k: int,
    m: int,
    T: int,
    c: float,
    omega_max: int,
) -> Tuple[float, float, float, float]:
    """Return the constant, gamma, sqrt(gamma), and 1/sqrt(gamma) coefficients."""
    log_T = math.log(T)
    if log_T <= 0.0:
        raise ValueError("Optimistic-Hire auto gamma requires T > 1.")

    log_ratio = math.log((m * T) / ((k - m) * log_T))

    leading_term = 5.0 * math.sqrt(m * (k - m) * T * log_T)
    switching_term = (c + 1.0) * (k - m)
    if c > 0.0:
        switching_term += (k - m) * c * math.log(
            (c * m * T) / ((k - m) * log_T)
        )

    constant_term = (
        leading_term
        + switching_term
        + k
        + (4.0 * m * k) / T
        + (omega_max + c) * m * k
    )
    gamma_coeff = float(k)
    sqrt_gamma_coeff = (
        2.0
        * (k - m)
        * math.sqrt(4.0 * log_T)
        * (1.0 + 0.5 * log_ratio)
    )
    inverse_sqrt_gamma_coeff = (omega_max + c) * m * math.sqrt(k * T)
    return (
        constant_term,
        gamma_coeff,
        sqrt_gamma_coeff,
        inverse_sqrt_gamma_coeff,
    )


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

    constant_term, gamma_coeff, sqrt_gamma_coeff, inverse_sqrt_gamma_coeff = (
        _optimistic_hire_bound_components(
            k=k,
            m=m,
            T=T,
            c=c,
            omega_max=omega_max,
        )
    )
    sqrt_gamma = math.sqrt(gamma)

    return (
        constant_term
        + gamma_coeff * gamma
        + sqrt_gamma_coeff * sqrt_gamma
        + inverse_sqrt_gamma_coeff / sqrt_gamma
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

    _, gamma_coeff, sqrt_gamma_coeff, inverse_sqrt_gamma_coeff = (
        _optimistic_hire_bound_components(
            k=k,
            m=m,
            T=T,
            c=c,
            omega_max=omega_max,
        )
    )

    # With x = sqrt(gamma), the gamma-dependent terms are:
    #   gamma_coeff * x^2 + sqrt_gamma_coeff * x + inverse_sqrt_gamma_coeff / x.
    # The derivative is:
    #   2 * gamma_coeff * x + sqrt_gamma_coeff - inverse_sqrt_gamma_coeff / x^2 = 0
    # which becomes the cubic
    #   2 * gamma_coeff * x^3 + sqrt_gamma_coeff * x^2 - inverse_sqrt_gamma_coeff = 0.
    roots = np.roots(
        [
            2.0 * gamma_coeff,
            sqrt_gamma_coeff,
            0.0,
            -inverse_sqrt_gamma_coeff,
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
    review_interval: int = 6 * 30 * 24,
    work_trial_periods: int = 1,
    work_trial_rotation_periods: int = 90 * 24,
    threshold: float = 0.5,
    ucb_coef: float = 2.0,
    gamma: float | str = "auto",
    log_frontier_sizes: bool = False,
):
    """
    Factory to build a policy object consistent with the simulation interface.

    Supported names (case- and spacing-insensitive after lowercasing/stripping):
      - "epsilon-greedy", "eps", "epsilon", "egreedy"
      - "semiannualreview", "semiannual-review", "semi-annual-review"
      - "worktrial", "work-trial"
      - "threshold", "threshold-n"
      - "omm", "optimistic-matroid-maximization", "optimistic matroid maximization"
      - "optimistic-hire", "optimistic hire", "optimistic-hire-auto", "paper", "algorithm-1", "oh"
      - "agrawalhegdeteneketzis", "classic", "rarely-switch", "round-robin", "aht"
    """
    name = policy_name.lower().strip()

    # Naive baselines
    if name in {"eps", "epsilon", "epsilon-greedy", "egreedy"}:
        return EpsilonGreedyHiringPolicy(k=k, m=m, epsilon=epsilon, rng=rng)

    elif name in {"semiannualreview", "semiannual-review", "semi-annual-review"}:
        return SemiAnnualReview(k=k, m=m, review_interval=review_interval, rng=rng)

    elif name in {"worktrial", "work-trial"}:
        return WorkTrial(
            k=k,
            m=m,
            trial_periods=work_trial_periods,
            rotation_periods=work_trial_rotation_periods,
            rng=rng,
        )

    elif name == "threshold":
        return Threshold(k=k, m=m, threshold=threshold, rng=rng)
    elif threshold_match := re.fullmatch(r"threshold-([0-9]*\.?[0-9]+)", name):
        parsed_threshold = float(threshold_match.group(1))
        return Threshold(k=k, m=m, threshold=parsed_threshold, rng=rng)

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
    elif name in {"optimistic-hire", "optimistic hire", "optimistic-hire-auto", "paper", "algorithm-1", "oh"}:
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
        return OptimisticHire(
            k=k,
            m=m,
            gamma=resolved_gamma,
            horizon=T,
            rng=rng,
            log_frontier_sizes=log_frontier_sizes,
        )
    
    elif name in {"optimistic-hire-gamma-1"}:
        return OptimisticHire(
            k=k,
            m=m,
            gamma=(c + omega_max) ** 2 * m,
            horizon=T,
            rng=rng,
            log_frontier_sizes=log_frontier_sizes,
        )
    
    elif name in {"optimistic-hire-gamma-2"}:
        return OptimisticHire(
            k=k,
            m=m,
            gamma=(c + omega_max) * m,
            horizon=T,
            rng=rng,
            log_frontier_sizes=log_frontier_sizes,
        )
    
    elif name in {"optimistic-hire-gamma-3"}:
        return OptimisticHire(
            k=k,
            m=m,
            gamma=c * m,
            horizon=T,
            rng=rng,
            log_frontier_sizes=log_frontier_sizes,
        )
    
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
    review_interval: int,
    work_trial_periods: int,
    work_trial_rotation_periods: int,
    threshold: float,
    gamma: float | str,
    c: float,
    omega_max: int,
    seed: int,
    frontier_size_log: list[ChooseTargetFrontierSizeRecord] | None = None,
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
        review_interval=review_interval,
        work_trial_periods=work_trial_periods,
        work_trial_rotation_periods=work_trial_rotation_periods,
        threshold=threshold,
        ucb_coef=2.0,  # adjust if you want different default exploration strength
        gamma=gamma,
        log_frontier_sizes=frontier_size_log is not None,
    )

    oracle_per_period = env.optimal_expected_reward()
    if oracle_per_period is None:
        raise ValueError("True means required for regret simulation.")

    regret_increments = np.zeros(T, dtype=np.float64)

    for _ in range(T):
        replacements = policy.act(env)
        obs, total_reward, cost, feedback = env.step(replacements)

        policy.update(feedback)

        active_now = feedback.active_set
        active_expected = sum(means[i - 1] for i in active_now)

        # Pseudo-regret increment: gap to oracle + switching cost
        regret_increments[env.t - 2] = (oracle_per_period - active_expected) + cost

    if frontier_size_log is not None and isinstance(policy, OptimisticHire):
        frontier_size_log.extend(
            replace(record, policy_name=policy_name, episode_seed=seed)
            for record in policy.frontier_size_log
        )

    return np.cumsum(regret_increments)


def _run_episode_worker(
    *,
    policy_name: str,
    k: int,
    m: int,
    means: Sequence[float],
    reward_process_name: str,
    reward_stddev: float,
    reward_lower: float,
    reward_upper: float,
    delay_process_name: str,
    delay_lower: int,
    calendar_frequency: Optional[int],
    calendar_distribution: str,
    calendar_geom_p: float,
    T: int,
    epsilon: float,
    review_interval: int,
    work_trial_periods: int,
    work_trial_rotation_periods: int,
    threshold: float,
    gamma: float | str,
    c: float,
    omega_max: int,
    seed: int,
) -> np.ndarray:
    reward_sampler_factory = make_reward_sampler_factory(
        reward_process_name,
        means=means,
        reward_stddev=reward_stddev,
        reward_lower=reward_lower,
        reward_upper=reward_upper,
    )
    delay_sampler_factory = make_delay_sampler_factory(
        delay_process_name,
        means=means,
        omega_max=omega_max,
        delay_lower=delay_lower,
        calendar_frequency=calendar_frequency,
        calendar_distribution=calendar_distribution,
        calendar_geom_p=calendar_geom_p,
    )
    return run_episode(
        policy_name=policy_name,
        k=k,
        m=m,
        means=means,
        reward_samplers=reward_sampler_factory(seed),
        delay_sampler=delay_sampler_factory(seed),
        T=T,
        epsilon=epsilon,
        review_interval=review_interval,
        work_trial_periods=work_trial_periods,
        work_trial_rotation_periods=work_trial_rotation_periods,
        threshold=threshold,
        gamma=gamma,
        c=c,
        omega_max=omega_max,
        seed=seed,
    )


def _run_episode_worker_from_kwargs(kwargs: Mapping[str, Any]) -> np.ndarray:
    return _run_episode_worker(**dict(kwargs))

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
    reward_process_name: str = "bernoulli",
    reward_stddev: float = 0.1,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
    delay_process_name: str = "uniform",
    delay_lower: int = 1,
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
    epsilon: float = 0.1,
    review_interval: int = 6 * 30 * 24,
    work_trial_periods: int = 1,
    work_trial_rotation_periods: int = 90 * 24,
    threshold: float = 0.5,
    gamma: float | str = "auto",
    c: float = 1,
    omega_max: int = 3,
    frontier_size_log: list[ChooseTargetFrontierSizeRecord] | None = None,
    n_runs: int = 50,
    n_jobs: int = 1,
    seed0: int = 0,
) -> Tuple[Sequence[float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    # Simple mean profile with a clear top-m
    rng = random.Random(999)
    user_supplied_reward_samplers = reward_samplers is not None
    user_supplied_delay_sampler = delay_sampler is not None
    user_supplied_reward_factory = reward_sampler_factory is not None
    user_supplied_delay_factory = delay_sampler_factory is not None

    if reward_samplers is not None and reward_sampler_factory is not None:
        raise ValueError("Pass either reward_samplers or reward_sampler_factory, not both.")
    if delay_sampler is not None and delay_sampler_factory is not None:
        raise ValueError("Pass either delay_sampler or delay_sampler_factory, not both.")

    if means is None:
        means = sorted([rng.uniform(0.1, 0.9) for _ in range(k)], reverse=True)
    
    if reward_samplers is None and reward_sampler_factory is None:
        reward_sampler_factory = make_reward_sampler_factory(
            reward_process_name,
            means=means,
            reward_stddev=reward_stddev,
            reward_lower=reward_lower,
            reward_upper=reward_upper,
        )
    
    if delay_sampler is None and delay_sampler_factory is None:
        delay_sampler_factory = make_delay_sampler_factory(
            delay_process_name,
            means=means,
            omega_max=omega_max,
            delay_lower=delay_lower,
            calendar_frequency=calendar_frequency,
            calendar_distribution=calendar_distribution,
            calendar_geom_p=calendar_geom_p,
        )

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")
    if n_jobs > 1 and frontier_size_log is not None:
        raise ValueError(
            "Frontier-size logging is currently supported only when n_jobs == 1."
        )
    if n_jobs > 1 and (
        user_supplied_reward_samplers
        or user_supplied_delay_sampler
        or user_supplied_reward_factory
        or user_supplied_delay_factory
    ):
        raise ValueError(
            "Parallel simulate() currently supports only built-in reward/delay process settings, "
            "not custom sampler objects or factories."
        )

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for pname in policies:
        if n_jobs == 1:
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
                        review_interval=review_interval,
                        work_trial_periods=work_trial_periods,
                        work_trial_rotation_periods=work_trial_rotation_periods,
                        threshold=threshold,
                        gamma=gamma,
                        c=c,
                        omega_max=omega_max,
                        seed=episode_seed,
                        frontier_size_log=frontier_size_log,
                    )
                )
        else:
            max_workers = min(n_jobs, n_runs)
            try:
                mp_context = mp.get_context("fork")
            except ValueError:
                mp_context = None

            worker_kwargs = [
                {
                    "policy_name": pname,
                    "k": k,
                    "m": m,
                    "means": means,
                    "reward_process_name": reward_process_name,
                    "reward_stddev": reward_stddev,
                    "reward_lower": reward_lower,
                    "reward_upper": reward_upper,
                    "delay_process_name": delay_process_name,
                    "delay_lower": delay_lower,
                    "calendar_frequency": calendar_frequency,
                    "calendar_distribution": calendar_distribution,
                    "calendar_geom_p": calendar_geom_p,
                    "T": T,
                    "epsilon": epsilon,
                    "review_interval": review_interval,
                    "work_trial_periods": work_trial_periods,
                    "work_trial_rotation_periods": work_trial_rotation_periods,
                    "threshold": threshold,
                    "gamma": gamma,
                    "c": c,
                    "omega_max": omega_max,
                    "seed": seed0 + r,
                }
                for r in range(n_runs)
            ]
            executor_kwargs: Dict[str, Any] = {"max_workers": max_workers}
            if mp_context is not None:
                executor_kwargs["mp_context"] = mp_context
            try:
                with ProcessPoolExecutor(**executor_kwargs) as executor:
                    curves = list(executor.map(_run_episode_worker_from_kwargs, worker_kwargs))
            except (OSError, PermissionError):
                curves = [
                    _run_episode_worker_from_kwargs(kwargs)
                    for kwargs in worker_kwargs
                ]
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
    delay_lower: int = 1,
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
    labels: Optional[Sequence[str]] = None,
    n_runs: int = 20,
    n_jobs: int = 1,
    base_seed: int = 12345,  
    title: str = 'Regret by Policy'
) -> None:
    if labels is None:
        labels = policies

    series = [
        ExperimentSeries(policy_name=pname, label=plabel)
        for plabel, pname in zip(labels, policies)
    ]
    simulate_kwargs = {
        "k": k,
        "m": m,
        "T": T,
        "means": means,
        "delay_process_name": delay_process_name,
        "delay_lower": delay_lower,
        "calendar_frequency": calendar_frequency,
        "calendar_distribution": calendar_distribution,
        "calendar_geom_p": calendar_geom_p,
        "c": c,
        "omega_max": omega_max,
        "n_runs": n_runs,
        "n_jobs": n_jobs,
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
    delay_process_name: str = "uniform",
    delay_lower: int = 1,
    calendar_frequency: Optional[int] = None,
    calendar_distribution: str = "geom",
    calendar_geom_p: float = 0.5,
    n_runs: int = 20,
    n_jobs: int = 1,
    base_seed: int = 12345,
    y_up_lim: Optional[float] = 3500,
    save_path: Optional[str] = None,
    show_plot: bool = True,
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
            "delay_process_name": delay_process_name,
            "delay_lower": delay_lower,
            "calendar_frequency": calendar_frequency,
            "calendar_distribution": calendar_distribution,
            "calendar_geom_p": calendar_geom_p,
            "omega_max": omega_max,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
            "seed0": base_seed,
        },
        title=rf"Cumulative regret of {policy_name} across switching costs.",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=None if y_up_lim is None else (0, y_up_lim),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
        save_path=save_path,
        show_plot=show_plot,
    )

def run_omega_sweep(
    *,
    k: int,
    m: int,
    T: int,
    policy_name: str = 'optimistic-hire',
    means: Sequence[float],
    c: float,
    omega_values: Sequence[float],
    omega_process: str = 'stochastic',
    omega_value_type: str = "max",
    stochastic_lower_fraction: float = 0.5,
    stochastic_radius_scale: float = 1.2,
    n_runs: int = 20,
    n_jobs: int = 1,
    base_seed: int = 12345,
    y_up_lim: int = 2500,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    if omega_process not in {"stochastic", "adversarial"}:
        raise ValueError("delays process not one of: 'stochastic', 'adversarial'")
    if omega_value_type not in {"max", "mean"}:
        raise ValueError("omega_value_type must be one of: 'max', 'mean'.")
    if not (0.0 <= stochastic_lower_fraction <= 1.0):
        raise ValueError("stochastic_lower_fraction must be in [0, 1].")
    if stochastic_radius_scale < 0:
        raise ValueError("stochastic_radius_scale must be non-negative.")

    delay_name = "stochastic" if omega_process == "stochastic" else "adversarial"

    def _resolve_omega_spec(value: float) -> tuple[str, Dict[str, Any]]:
        omega_value = float(value)
        rounded_value = int(round(omega_value))
        if omega_process == "stochastic" and omega_value_type == "max":
            delay_lower = max(1, int(math.ceil(stochastic_lower_fraction * omega_value)))
            delay_upper = max(delay_lower, rounded_value)
            return (
                rf"$\omega_\max={rounded_value}$",
                {
                    "omega_max": delay_upper,
                    "delay_lower": delay_lower,
                    "delay_process_name": delay_name,
                },
            )

        if omega_process == "stochastic" and omega_value_type == "mean":
            radius = stochastic_radius_scale * omega_value
            delay_lower = max(1, int(math.ceil(omega_value - radius)))
            delay_upper = max(delay_lower, int(math.floor(omega_value + radius)))
            expected_delay = 0.5 * (delay_lower + delay_upper)
            return (
                rf"$E [\omega]={int(round(expected_delay))}$",
                {
                    "omega_max": delay_upper,
                    "delay_lower": delay_lower,
                    "delay_process_name": delay_name,
                },
            )

        return (
            rf"$\omega_\max={rounded_value}$",
            {
                "omega_max": rounded_value,
                "delay_process_name": delay_name,
            },
        )

    series = []
    for value in omega_values:
        label, sim_kwargs = _resolve_omega_spec(value)
        series.append(
            ExperimentSeries(
                policy_name=policy_name,
                label=label,
                sim_kwargs=sim_kwargs,
                plot_kwargs={"linewidth": 2},
            )
        )

    plot_regret_series(
        series=series,
        simulate_kwargs={
            "k": k,
            "m": m,
            "T": T,
            "means": means,
            "c": c,
            "n_runs": n_runs,
            "n_jobs": n_jobs,
            "seed0": base_seed,
        },
        title=rf"Cumulative regret of {policy_name} with {omega_process} delays.",
        xlabel="t",
        ylabel="Cumulative regret",
        figure_kwargs={"figsize": (8, 5)},
        ylim=(0, y_up_lim),
        grid_kwargs={"which": "both", "linestyle": "--", "alpha": 0.5},
        save_path=save_path,
        show_plot=show_plot,
    )
