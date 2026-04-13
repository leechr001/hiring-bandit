import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from experiments.benchmark_config_2 import (
    average_title,
    curve_cache_mode,
    cumulative_title,
    output_dir,
    planning_horizons,
    series,
    simulate_kwargs,
)
from simulation import (
    plot_average_regret_series,
    plot_regret_series,
    print_planning_horizon_regret_table,
    run_series_simulations,
)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_display_label(label: str) -> str:
    mapping = {
        "AgrawalHegdeTeneketzis": "AHT",
        "Semi-Annual Review": "SemiAnnualReview",
    }
    return mapping.get(label, label)


def _format_int_with_latex_commas(value: float) -> str:
    rounded = f"{int(round(value)):,}"
    return rounded.replace(",", "{,}")


def _format_cumulative(mean: float, std: float) -> str:
    return (
        f"${_format_int_with_latex_commas(mean)}"
        f" \\pm {_format_int_with_latex_commas(std)}$"
    )


def _format_normalized(mean: float, std: float) -> str:
    return f"${100.0 * mean:.2f}\\% \\pm {100.0 * std:.2f}\\%$"


def _build_latex_table(
    *,
    output_series,
    results,
    normalized_results,
    horizons,
) -> str:
    labels = [spec.label or spec.policy_name for spec in output_series]

    def build_subtable(table_labels, formatter, source_results) -> list[str]:
        lines: list[str] = []
        lines.append(r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r}")
        lines.append(r"\toprule")
        header = ["Horizon"] + [_escape_latex(_latex_display_label(label)) for label in table_labels]
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        for horizon_label, horizon_period in horizons:
            idx = int(horizon_period) - 1
            row = [_escape_latex(str(horizon_label))]
            for label in table_labels:
                mean_curve, std_curve = source_results[label]
                row.append(formatter(float(mean_curve[idx]), float(std_curve[idx])))
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular*}")
        return lines

    label_groups = [
        labels[group_start:group_start + 3]
        for group_start in range(0, len(labels), 3)
        if labels[group_start:group_start + 3]
    ]

    def append_metric_section(
        lines: list[str],
        *,
        formatter,
        source_results,
        caption: str,
        label: str,
    ) -> None:
        lines.append(r"    \small")
        lines.append(r"    \setlength{\tabcolsep}{4pt}")
        lines.append(r"    ")

        for group_index, table_labels in enumerate(label_groups):
            if group_index > 0:
                lines.append(r"    ")
                lines.append(r"    \vspace{0.5em}")
                lines.append(r"    ")
            for line in build_subtable(table_labels, formatter, source_results):
                lines.append(f"    {line}")

        lines.append(r"    ")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"    \centering")
    lines.append("")

    append_metric_section(
        lines,
        formatter=_format_cumulative,
        source_results=results,
        caption="Cumulative regret across policies and planning horizons.",
        label="tab:cumulative-regret",
    )

    lines.append("")
    lines.append(r"    \vspace{1em}")
    lines.append("")

    append_metric_section(
        lines,
        formatter=_format_normalized,
        source_results=normalized_results,
        caption="Normalized loss across policies and planning horizons.",
        label="tab:loss-percent",
    )

    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def _save_text(path: Path, content: str) -> None:
    path.write_text(content)


def _load_saved_results(
    *,
    curves_path: Path,
    output_series,
):
    curves = np.load(curves_path, allow_pickle=True)
    means = [float(value) for value in curves["means"].tolist()]

    saved_labels = [str(label) for label in curves["labels"].tolist()]
    expected_labels = [spec.label or spec.policy_name for spec in output_series]
    if saved_labels != expected_labels:
        raise ValueError(
            "Saved benchmark curves do not match the configured series order."
        )

    results = {}
    normalized_results = {}
    for label in expected_labels:
        slug = _slug(label)
        results[label] = (
            np.asarray(curves[f"{slug}_cumulative_mean"], dtype=np.float64),
            np.asarray(curves[f"{slug}_cumulative_std"], dtype=np.float64),
        )
        normalized_results[label] = (
            np.asarray(curves[f"{slug}_normalized_mean"], dtype=np.float64),
            np.asarray(curves[f"{slug}_normalized_std"], dtype=np.float64),
        )

    return means, results, normalized_results


def _resolve_curve_data(
    *,
    curves_path: Path,
    cache_mode: str,
    output_series,
    simulate_kwargs,
    average_title: str,
):
    mode = cache_mode.strip().lower()
    if mode not in {"auto", "load", "regenerate"}:
        raise ValueError(
            "curve_cache_mode must be one of: 'auto', 'load', 'regenerate'."
        )

    if mode == "load":
        if not curves_path.exists():
            raise FileNotFoundError(
                f"curve_cache_mode='load' requires an existing file at {curves_path}."
            )
        means, results, normalized_results = _load_saved_results(
            curves_path=curves_path,
            output_series=output_series,
        )
        return means, results, normalized_results, False

    if mode == "auto" and curves_path.exists():
        means, results, normalized_results = _load_saved_results(
            curves_path=curves_path,
            output_series=output_series,
        )
        return means, results, normalized_results, False

    means, results = run_series_simulations(
        series=output_series,
        simulate_kwargs=simulate_kwargs,
    )
    _, normalized_results = plot_average_regret_series(
        series=output_series,
        simulate_kwargs=simulate_kwargs,
        title=average_title,
        ylim=(0, 0.25),
        show_plot=False,
        precomputed=(means, results),
    )
    plt.close()
    return means, results, normalized_results, True


output_dir.mkdir(parents=True, exist_ok=True)
curves_path = output_dir / "benchmark_curves.npz"

means, results, normalized_results, should_write_curves = _resolve_curve_data(
    curves_path=curves_path,
    cache_mode=curve_cache_mode,
    output_series=series,
    simulate_kwargs=simulate_kwargs,
    average_title=average_title,
)

plot_regret_series(
    series=series,
    simulate_kwargs=simulate_kwargs,
    title=cumulative_title,
    ylim=(0, 100000),
    save_path=str(output_dir / "benchmark_cum_regret.png"),
    precomputed=(means, results),
)
plt.close()

plot_regret_series(
    series=series,
    simulate_kwargs=simulate_kwargs,
    title=average_title,
    ylabel="Normalized loss",
    ylim=(0, 0.25),
    y_axis_percent=True,
    save_path=str(output_dir / "benchmark_normalized_loss.png"),
    precomputed=(means, normalized_results),
)
plt.close()

print()
print_planning_horizon_regret_table(
    series=series,
    means=means,
    m=int(simulate_kwargs["m"]),
    results=results,
    horizons=planning_horizons,
)

if should_write_curves:
    periods = np.arange(1, len(next(iter(results.values()))[0]) + 1, dtype=np.int64)
    npz_payload = {
        "means": np.asarray(means, dtype=np.float64),
        "periods": periods,
        "labels": np.asarray([spec.label or spec.policy_name for spec in series], dtype=object),
    }
    for spec in series:
        label = spec.label or spec.policy_name
        slug = _slug(label)
        mean_curve, std_curve = results[label]
        norm_mean_curve, norm_std_curve = normalized_results[label]
        npz_payload[f"{slug}_cumulative_mean"] = np.asarray(mean_curve, dtype=np.float64)
        npz_payload[f"{slug}_cumulative_std"] = np.asarray(std_curve, dtype=np.float64)
        npz_payload[f"{slug}_normalized_mean"] = np.asarray(norm_mean_curve, dtype=np.float64)
        npz_payload[f"{slug}_normalized_std"] = np.asarray(norm_std_curve, dtype=np.float64)

    np.savez(output_dir / "benchmark_curves.npz", **npz_payload)
_save_text(
    output_dir / "benchmark_planning_horizons_mean_pm_std.tex",
    _build_latex_table(
        output_series=series,
        results=results,
        normalized_results=normalized_results,
        horizons=planning_horizons,
    ),
)

csv_path = output_dir / "benchmark_planning_horizons.csv"
if csv_path.exists():
    csv_path.unlink()

with (output_dir / "benchmark_metadata.json").open("w") as handle:
    json.dump(
        {
            "simulate_kwargs": simulate_kwargs,
            "planning_horizons": planning_horizons,
            "cumulative_title": cumulative_title,
            "average_title": average_title,
            "curve_cache_mode": curve_cache_mode,
            "artifacts": {
                "cumulative_plot": "benchmark_cum_regret.png",
                "normalized_plot": "benchmark_normalized_loss.png",
                "curves": "benchmark_curves.npz",
                "latex_table": "benchmark_planning_horizons_mean_pm_std.tex",
            },
        },
        handle,
        indent=2,
    )

print()
print(f"Saved benchmark artifacts to {output_dir}")
