from __future__ import annotations

import json
from pathlib import Path

from experiments.helpers import benchmark_output_dir, make_benchmark_means
from simulation import simulate
from experiments.simulation_setups.config_main import (

    K,
    M,
    N_JOBS,
    N_RUNS,
    OMEGA_MEAN,
    PERFORMANCE_MEANS,
    SWITCHING_COST,
    SIMULATE_KWARGS,
    BASE_SEED
)

output_dir = benchmark_output_dir(
    module_file=__file__,
    output_subdir="delayed_replace_ucb_ablations",
)

POLICIES = [
    ("delayed-replace-ucb", "DR-UCB"),
    ("delayed-replace-ucb-fixed-calendar", "DR-UCB + fixed calendar switching"),
    ("delayed-replace-ucb-no-screen", "DR-UCB without horizon screening"),
    ("delayed-replace-ucb-random-pairing", "DR-UCB + random pairing"),
]

HORIZONS = [
    ("3 months", 90),
    ("12 months", 365),
]

STRESS_TESTS = [
    {
        "label": "High switching cost (c=168)",
        "policies": [
            ("delayed-replace-ucb", "DR-UCB"),
            ("delayed-replace-ucb-no-screen", "DR-UCB without horizon screening"),
        ],
        "simulate_kwargs": {
            "c": 168,
        },
    }
]


def _fmt_regret(mean: float, std: float) -> str:
    return f"${int(round(mean)):,} \\pm {int(round(std)):,}$".replace(",", "{,}")


def _fmt_pct(mean: float, std: float) -> str:
    return f"${100.0 * mean:.2f}\\% \\pm {100.0 * std:.2f}\\%$"


def _oracle_reward() -> float:
    return float(sum(sorted((float(mu) for mu in PERFORMANCE_MEANS), reverse=True)[:M]))


def _run_summary(
    *,
    policies: list[tuple[str, str]],
    horizons: list[tuple[str, int]],
    simulate_kwargs: dict,
) -> dict[str, dict[str, dict[str, float]]]:
    oracle_reward = _oracle_reward()
    summary: dict[str, dict[str, dict[str, float]]] = {
        label: {} for _, label in policies
    }

    for horizon_label, horizon_period in horizons:
        _, results = simulate(
            policies=[policy_name for policy_name, _ in policies],
            k=K,
            m=M,
            T=horizon_period,
            means=PERFORMANCE_MEANS,
            delay_process_name="geometric",
            delay_geom_p=1/OMEGA_MEAN,
            gamma=f"auto-{OMEGA_MEAN}",
            n_runs=N_RUNS,
            n_jobs=N_JOBS,
            seed0=BASE_SEED,
            **simulate_kwargs,
        )

        for policy_name, label in policies:
            mean_curve, std_curve = results[policy_name]
            idx = horizon_period - 1
            summary[label][horizon_label] = {
                "cumulative_regret_mean": float(mean_curve[idx]),
                "cumulative_regret_std": float(std_curve[idx]),
                "normalized_loss_mean": float(mean_curve[idx] / (horizon_period * oracle_reward)),
                "normalized_loss_std": float(std_curve[idx] / (horizon_period * oracle_reward)),
            }

    return summary


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = _run_summary(
        policies=POLICIES,
        horizons=HORIZONS,
        simulate_kwargs={},
    )

    stress_summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for spec in STRESS_TESTS:
        stress_summaries[spec["label"]] = _run_summary(
            policies=spec["policies"],
            horizons=HORIZONS,
            simulate_kwargs=spec["simulate_kwargs"],
        )

    summary = {
        "benchmark_ablation": benchmark_summary,
        "stress_tests": stress_summaries,
    }

    latex_lines: list[str] = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Variant & 3-month regret & 12-month regret \\")
    latex_lines.append(r"    \midrule")
    for _, label in POLICIES:
        latex_lines.append(
            "    "
            + " & ".join(
                [
                    label,
                    _fmt_regret(
                        benchmark_summary[label]["3 months"]["cumulative_regret_mean"],
                        benchmark_summary[label]["3 months"]["cumulative_regret_std"],
                    ),
                    _fmt_regret(
                        benchmark_summary[label]["12 months"]["cumulative_regret_mean"],
                        benchmark_summary[label]["12 months"]["cumulative_regret_std"],
                    ),
                ]
            )
            + r" \\"
        )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        rf"    \caption{{Ablations of \textsc{{DR-UCB}} in the benchmark calibration of Section~\ref{{numerical:benchmarks}}. Each row averages {N_RUNS} replications.}}"
    )
    latex_lines.append(r"    \label{tab:da-ucb-ablations}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Variant & 3-month normalized loss & 12-month normalized loss \\")
    latex_lines.append(r"    \midrule")
    for _, label in POLICIES:
        latex_lines.append(
            "    "
            + " & ".join(
                [
                    label,
                    _fmt_pct(
                        benchmark_summary[label]["3 months"]["normalized_loss_mean"],
                        benchmark_summary[label]["3 months"]["normalized_loss_std"],
                    ),
                    _fmt_pct(
                        benchmark_summary[label]["12 months"]["normalized_loss_mean"],
                        benchmark_summary[label]["12 months"]["normalized_loss_std"],
                    ),
                ]
            )
            + r" \\"
        )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        r"    \caption{Normalized-loss version of Table~\ref{tab:da-ucb-ablations}.}"
    )
    latex_lines.append(r"    \label{tab:da-ucb-ablations-loss}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l r}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Scenario & Variant & 12-month regret \\")
    latex_lines.append(r"    \midrule")
    for spec in STRESS_TESTS:
        label = spec["label"]
        for _, policy_label in spec["policies"]:
            latex_lines.append(
                "    "
                + " & ".join(
                    [
                        label,
                        policy_label,
                        _fmt_regret(
                            stress_summaries[label][policy_label]["12 months"]["cumulative_regret_mean"],
                            stress_summaries[label][policy_label]["12 months"]["cumulative_regret_std"],
                        ),
                    ]
                )
                + r" \\"
            )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        rf"    \caption{{Stress-test ablations for \textsc{{DR-UCB}}. The high-cost scenario uses \(c=168\) with geometric delays; Each row averages {N_RUNS} replications.}}"
    )
    latex_lines.append(r"    \label{tab:da-ucb-ablations-stress}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    (output_dir / "delayed_replace_ucb_ablations.json").write_text(
        json.dumps(summary, indent=2)
    )
    (output_dir / "delayed_replace_ucb_ablations.tex").write_text(
        "\n".join(latex_lines)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
