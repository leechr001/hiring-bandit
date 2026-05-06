import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from simulation import (
    ExperimentSeries,
    plot_average_regret_series,
    plot_regret_series,
    print_planning_horizon_regret_table,
    run_series_simulations,
)
from experiments.simulation_setups.config_main import (
    DEFAULT_AVERAGE_TITLE,
    DEFAULT_CUMULATIVE_TITLE,
    DEFAULT_PLANNING_HORIZONS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def benchmark_output_dir(*, module_file: str, output_subdir: str) -> Path:
    return PROJECT_ROOT / "artifacts" / output_subdir


def make_benchmark_means(
    *,
    k: int,
    seed: int = 12345,
    low: float = 0.3,
    high: float = 0.7,
) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=k).tolist()


def default_benchmark_series(
    *,
    threshold_value: float,
    threshold_label: str,
) -> list[ExperimentSeries]:
    return [
        ExperimentSeries(
            policy_name="delayed-replace-ucb",
            label="DR-UCB",
            sim_kwargs={"gamma": "auto"},
        ),
        ExperimentSeries(
            policy_name="a-aht",
            label="Adapted-AHT",
        ),
        ExperimentSeries(
            policy_name="a-omm",
            label="Adapted-OMM",
        ),
        ExperimentSeries(
            policy_name="FixedScheduleGreedy",
            label="FixedScheduleGreedy",
        ),
        ExperimentSeries(
            policy_name="WorkTrial",
            label="WorkTrial",
        ),
        ExperimentSeries(
            policy_name="Threshold",
            label=threshold_label,
            sim_kwargs={"threshold": threshold_value},
        ),
    ]


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _json_safe_metadata(value: Any) -> Any:
    """Convert metadata values to JSON-safe summaries without mutating them."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, np.generic):
        return _json_safe_metadata(value.item())

    if isinstance(value, np.ndarray):
        return _json_safe_metadata(value.tolist())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_metadata(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_metadata(item) for item in value]

    if callable(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        qualname = getattr(value, "__qualname__", value.__class__.__qualname__)
        return f"<callable {module}.{qualname}>"

    return f"<non-serializable {value.__class__.__module__}.{value.__class__.__qualname__}>"


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
        "Adapted-AHT": "A-AHT",
        "Adapted-OMM": "A-OMM",
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
    output_series: Sequence[ExperimentSeries],
    results,
    normalized_results,
    horizons: Sequence[tuple[str, int]],
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
    output_series: Sequence[ExperimentSeries],
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
    curves_path: Optional[Path],
    cache_mode: str,
    output_series: Sequence[ExperimentSeries],
    simulate_kwargs: Mapping[str, Any],
    average_title: str,
    normalized_ylim: Optional[tuple[float, float]],
):
    mode = cache_mode.strip().lower()
    if mode not in {"auto", "load", "regenerate", "disabled"}:
        raise ValueError(
            "curve_cache_mode must be one of: 'auto', 'load', 'regenerate', 'disabled'."
        )

    if mode == "disabled":
        means, results = run_series_simulations(
            series=output_series,
            simulate_kwargs=simulate_kwargs,
        )
        _, normalized_results = plot_average_regret_series(
            series=output_series,
            simulate_kwargs=simulate_kwargs,
            title=average_title,
            ylim=normalized_ylim,
            show_plot=False,
            precomputed=(means, results),
        )
        plt.close()
        return means, results, normalized_results, False

    if curves_path is None:
        raise ValueError("curves_path is required unless curve_cache_mode='disabled'.")

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
        ylim=normalized_ylim,
        show_plot=False,
        precomputed=(means, results),
    )
    plt.close()
    return means, results, normalized_results, True


def _write_curve_cache(
    *,
    curves_path: Path,
    output_series: Sequence[ExperimentSeries],
    means,
    results,
    normalized_results,
) -> None:
    periods = np.arange(1, len(next(iter(results.values()))[0]) + 1, dtype=np.int64)
    npz_payload = {
        "means": np.asarray(means, dtype=np.float64),
        "periods": periods,
        "labels": np.asarray([spec.label or spec.policy_name for spec in output_series], dtype=object),
    }

    for spec in output_series:
        label = spec.label or spec.policy_name
        slug = _slug(label)
        mean_curve, std_curve = results[label]
        norm_mean_curve, norm_std_curve = normalized_results[label]
        npz_payload[f"{slug}_cumulative_mean"] = np.asarray(mean_curve, dtype=np.float64)
        npz_payload[f"{slug}_cumulative_std"] = np.asarray(std_curve, dtype=np.float64)
        npz_payload[f"{slug}_normalized_mean"] = np.asarray(norm_mean_curve, dtype=np.float64)
        npz_payload[f"{slug}_normalized_std"] = np.asarray(norm_std_curve, dtype=np.float64)

    np.savez(curves_path, **npz_payload)


def run_benchmark(
    *,
    series: Sequence[ExperimentSeries],
    simulate_kwargs: Mapping[str, Any],
    output_dir: Optional[Path] = None,
    cumulative_title: str = DEFAULT_CUMULATIVE_TITLE,
    average_title: str = DEFAULT_AVERAGE_TITLE,
    curve_cache_mode: str = "regenerate",
    planning_horizons: Sequence[tuple[str, int]] = DEFAULT_PLANNING_HORIZONS,
    cumulative_plot_name: str = "benchmark_cum_regret.png",
    normalized_plot_name: str = "benchmark_normalized_loss.png",
    curves_name: str = "benchmark_curves.npz",
    latex_table_name: str = "benchmark_planning_horizons_mean_pm_std.tex",
    metadata_name: str = "benchmark_metadata.json",
    cumulative_ylim: Optional[tuple[float, float]] = (0, 100000),
    normalized_ylim: Optional[tuple[float, float]] = (0, 0.25),
    save_artifacts: bool = True,
    show_plots: bool = True,
) -> None:
    if save_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when save_artifacts=True.")
        output_dir.mkdir(parents=True, exist_ok=True)
        curves_path = output_dir / curves_name
    else:
        curves_path = None

    means, results, normalized_results, should_write_curves = _resolve_curve_data(
        curves_path=curves_path,
        cache_mode=curve_cache_mode,
        output_series=series,
        simulate_kwargs=simulate_kwargs,
        average_title=average_title,
        normalized_ylim=normalized_ylim,
    )

    plot_regret_series(
        series=series,
        simulate_kwargs=simulate_kwargs,
        title=cumulative_title,
        ylim=cumulative_ylim,
        save_path=str(output_dir / cumulative_plot_name) if save_artifacts and output_dir is not None else None,
        show_plot=show_plots,
        precomputed=(means, results),
    )
    plt.close()

    plot_regret_series(
        series=series,
        simulate_kwargs=simulate_kwargs,
        title=average_title,
        ylabel="Normalized loss",
        ylim=normalized_ylim,
        y_axis_percent=True,
        save_path=str(output_dir / normalized_plot_name) if save_artifacts and output_dir is not None else None,
        show_plot=show_plots,
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

    if should_write_curves and curves_path is not None:
        _write_curve_cache(
            curves_path=curves_path,
            output_series=series,
            means=means,
            results=results,
            normalized_results=normalized_results,
        )

    if save_artifacts and output_dir is not None:
        _save_text(
            output_dir / latex_table_name,
            _build_latex_table(
                output_series=series,
                results=results,
                normalized_results=normalized_results,
                horizons=planning_horizons,
            ),
        )

        with (output_dir / metadata_name).open("w") as handle:
            metadata = {
                "simulate_kwargs": dict(simulate_kwargs),
                "planning_horizons": list(planning_horizons),
                "cumulative_title": cumulative_title,
                "average_title": average_title,
                "curve_cache_mode": curve_cache_mode,
                "cumulative_ylim": cumulative_ylim,
                "normalized_ylim": normalized_ylim,
                "artifacts": {
                    "cumulative_plot": cumulative_plot_name,
                    "normalized_plot": normalized_plot_name,
                    "curves": curves_name,
                    "latex_table": latex_table_name,
                },
            }
            json.dump(_json_safe_metadata(metadata), handle, indent=2)

        print()
        print(f"Saved benchmark artifacts to {output_dir}")
    else:
        print()
        print("Benchmark sandbox run finished without writing artifacts.")
