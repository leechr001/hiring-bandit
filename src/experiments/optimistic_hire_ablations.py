from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmark_common import benchmark_output_dir, make_benchmark_means
from simulation import simulate


k = 150
m = 100
T = 5 * 365 * 24
c = 8
omega_max = 8
delay_lower = 8
n_runs = 4
n_jobs = 1
seed0 = 12345

means = make_benchmark_means(k=k, seed=seed0)
output_dir = benchmark_output_dir(
    module_file=__file__,
    output_subdir="optimistic_hire_ablations",
)

POLICIES = [
    ("optimistic-hire", "OH"),
    ("optimistic-hire-fixed-calendar", "OH + fixed calendar switching"),
    ("optimistic-hire-no-screen", "OH without horizon screening"),
    ("optimistic-hire-random-pairing", "OH + random pairing"),
]

HORIZONS = [
    ("12 months", 365 * 24),
    ("5 years", 5 * 365 * 24),
]

STRESS_TESTS = [
    {
        "label": "High switching cost (c=168)",
        "policies": [
            ("optimistic-hire", "OH"),
            ("optimistic-hire-no-screen", "OH without horizon screening"),
        ],
        "simulate_kwargs": {
            "c": 168,
            "delay_process_name": "uniform",
        },
    },
    {
        "label": "Adversarial delays",
        "policies": [
            ("optimistic-hire", "OH"),
            ("optimistic-hire-random-pairing", "OH + random pairing"),
        ],
        "simulate_kwargs": {
            "c": c,
            "delay_process_name": "adversarial",
        },
    },
]


def _fmt_regret(mean: float, std: float) -> str:
    return f"${int(round(mean)):,} \\pm {int(round(std)):,}$".replace(",", "{,}")


def _fmt_pct(mean: float, std: float) -> str:
    return f"${100.0 * mean:.2f}\\% \\pm {100.0 * std:.2f}\\%$"


def _oracle_reward() -> float:
    return float(sum(sorted((float(mu) for mu in means), reverse=True)[:m]))


def _run_summary(
    *,
    policies: list[tuple[str, str]],
    horizons: list[tuple[str, int]],
    simulate_kwargs: dict,
) -> dict[str, dict[str, dict[str, float]]]:
    _, results = simulate(
        policies=[policy_name for policy_name, _ in policies],
        k=k,
        m=m,
        T=T,
        means=means,
        omega_max=omega_max,
        delay_lower=delay_lower,
        gamma="auto",
        n_runs=n_runs,
        n_jobs=n_jobs,
        seed0=seed0,
        **simulate_kwargs,
    )

    oracle_reward = _oracle_reward()
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for policy_name, label in policies:
        mean_curve, std_curve = results[policy_name]
        policy_summary: dict[str, dict[str, float]] = {}
        for horizon_label, horizon_period in horizons:
            idx = horizon_period - 1
            policy_summary[horizon_label] = {
                "cumulative_regret_mean": float(mean_curve[idx]),
                "cumulative_regret_std": float(std_curve[idx]),
                "normalized_loss_mean": float(mean_curve[idx] / (horizon_period * oracle_reward)),
                "normalized_loss_std": float(std_curve[idx] / (horizon_period * oracle_reward)),
            }
        summary[label] = policy_summary
    return summary


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = _run_summary(
        policies=POLICIES,
        horizons=HORIZONS,
        simulate_kwargs={
            "c": c,
            "delay_process_name": "uniform",
        },
    )

    stress_summaries: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for spec in STRESS_TESTS:
        stress_summaries[spec["label"]] = _run_summary(
            policies=spec["policies"],
            horizons=[("5 years", T)],
            simulate_kwargs=dict(spec["simulate_kwargs"]),
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
    latex_lines.append(r"    Variant & 12-month regret & 5-year regret \\")
    latex_lines.append(r"    \midrule")
    for _, label in POLICIES:
        latex_lines.append(
            "    "
            + " & ".join(
                [
                    label,
                    _fmt_regret(
                        benchmark_summary[label]["12 months"]["cumulative_regret_mean"],
                        benchmark_summary[label]["12 months"]["cumulative_regret_std"],
                    ),
                    _fmt_regret(
                        benchmark_summary[label]["5 years"]["cumulative_regret_mean"],
                        benchmark_summary[label]["5 years"]["cumulative_regret_std"],
                    ),
                ]
            )
            + r" \\"
        )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        r"    \caption{Ablations of \textsc{Optimistic-Hire} in the benchmark calibration of Section~\ref{numerical:benchmarks}. Each row averages four replications.}"
    )
    latex_lines.append(r"    \label{tab:oh-ablations}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Variant & 12-month normalized loss & 5-year normalized loss \\")
    latex_lines.append(r"    \midrule")
    for _, label in POLICIES:
        latex_lines.append(
            "    "
            + " & ".join(
                [
                    label,
                    _fmt_pct(
                        benchmark_summary[label]["12 months"]["normalized_loss_mean"],
                        benchmark_summary[label]["12 months"]["normalized_loss_std"],
                    ),
                    _fmt_pct(
                        benchmark_summary[label]["5 years"]["normalized_loss_mean"],
                        benchmark_summary[label]["5 years"]["normalized_loss_std"],
                    ),
                ]
            )
            + r" \\"
        )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        r"    \caption{Normalized-loss version of Table~\ref{tab:oh-ablations}.}"
    )
    latex_lines.append(r"    \label{tab:oh-ablations-loss}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \small")
    latex_lines.append(r"    \setlength{\tabcolsep}{4pt}")
    latex_lines.append(r"    \begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l l r}")
    latex_lines.append(r"    \toprule")
    latex_lines.append(r"    Scenario & Variant & 5-year regret \\")
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
                            stress_summaries[label][policy_label]["5 years"]["cumulative_regret_mean"],
                            stress_summaries[label][policy_label]["5 years"]["cumulative_regret_std"],
                        ),
                    ]
                )
                + r" \\"
            )
    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(
        r"    \caption{Stress-test ablations for \textsc{Optimistic-Hire}. The high-cost scenario uses \(c=168\) with i.i.d.\ uniform delays; the adversarial-delay scenario uses the same cost as the benchmark but assigns long delays to beneficial replacements and short delays to detrimental ones. Each row averages four replications.}"
    )
    latex_lines.append(r"    \label{tab:oh-ablations-stress}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("")

    (output_dir / "optimistic_hire_ablations.json").write_text(
        json.dumps(summary, indent=2)
    )
    (output_dir / "optimistic_hire_ablations.tex").write_text(
        "\n".join(latex_lines)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
