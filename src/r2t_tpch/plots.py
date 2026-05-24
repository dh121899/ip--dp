"""Matplotlib plots for experiment CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from r2t_tpch.metrics import is_success_status

COUNT_QUERY_IDS = {
    "q12_sja_count",
    "q3_spja_order_count",
    "q10_spja_customer_count",
}
REVENUE_QUERY_IDS = {"q5_sja_revenue", "q7_sja_revenue"}
LOG_FLOOR = 1e-12


def build_parser() -> argparse.ArgumentParser:
    """Build the plotting CLI parser."""

    parser = argparse.ArgumentParser(description="Plot R2T TPC-H experiment results.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment CSV outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory where figures will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate available figures from experiment CSVs."""

    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = _read_csv(results_dir / "results_runs.csv")
    summary = _read_csv(results_dir / "results_summary.csv")
    tau_trace = _read_csv(results_dir / "tau_trace.csv")
    contribution_stats = _read_csv(results_dir / "contribution_stats.csv")

    _print_status_counts(runs)
    valid_runs = _valid_result_rows(runs)
    valid_summary = _valid_summary_rows(summary)

    _plot_error_bar(valid_summary, figures_dir)
    _plot_error_boxplot(valid_runs, figures_dir)
    _plot_error_vs_epsilon(valid_summary, figures_dir)
    _plot_error_vs_gsq(valid_summary, figures_dir)
    _plot_error_runtime_vs_scale(valid_summary, figures_dir)
    _plot_tau_sweep(tau_trace, figures_dir)
    _plot_contribution_summary(contribution_stats, figures_dir)
    _plot_runtime_error_pareto(valid_summary, figures_dir)
    _plot_shifted_inverse_comparison(valid_summary, figures_dir)
    return 0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Skipping missing CSV: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        print(f"Skipping empty CSV: {path}")
        return pd.DataFrame()


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _print_status_counts(runs: pd.DataFrame) -> None:
    if runs.empty or not {"mechanism", "status"}.issubset(runs.columns):
        print("Status counts unavailable: run data missing")
        return
    print("Status counts by mechanism:")
    print(pd.crosstab(runs["mechanism"], runs["status"]).to_string())


def _valid_result_rows(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    if "is_valid_result" in runs:
        return runs[runs["is_valid_result"].map(_coerce_bool)]
    return runs[runs["status"].map(is_success_status)]


def _valid_summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    filtered = summary.copy()
    if "n_valid" in filtered:
        filtered = filtered[filtered["n_valid"].fillna(0).astype(float) > 0]
    if "mean_rel_error" in filtered:
        filtered = filtered[filtered["mean_rel_error"].notna()]
    return filtered


def _coerce_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _plot_error_bar(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or "mean_rel_error" not in summary:
        print("Skipping error_bar_by_query_mechanism.png: summary data missing")
        return
    _plot_error_bar_variant(
        summary,
        figures_dir / "error_bar_by_query_mechanism.png",
        "Relative Error by Query and Mechanism",
    )
    _plot_error_bar_variant(
        summary,
        figures_dir / "error_bar_by_query_mechanism_logy.png",
        "Relative Error by Query and Mechanism (log scale)",
        log_y=True,
    )
    _plot_error_bar_variant(
        summary,
        figures_dir / "error_bar_without_directlaplace.png",
        "Relative Error by Query and Mechanism excluding DirectLaplace",
        exclude_direct_laplace=True,
    )
    _plot_error_bar_variant(
        summary,
        figures_dir / "error_bar_count_queries.png",
        "Relative Error for Count and SPJA Queries",
        query_ids=COUNT_QUERY_IDS,
        log_y=True,
    )
    revenue_summary = summary[summary["query_id"].isin(REVENUE_QUERY_IDS)]
    _plot_error_bar_variant(
        summary,
        figures_dir / "error_bar_revenue_queries.png",
        "Relative Error for Revenue Queries",
        query_ids=REVENUE_QUERY_IDS,
        log_y=_needs_log_scale(revenue_summary, "median_rel_error"),
    )


def _plot_error_bar_variant(
    summary: pd.DataFrame,
    path: Path,
    title: str,
    *,
    log_y: bool = False,
    exclude_direct_laplace: bool = False,
    query_ids: set[str] | None = None,
) -> None:
    value_col = (
        "median_rel_error" if "median_rel_error" in summary else "mean_rel_error"
    )
    data = summary.dropna(subset=["query_id", "mechanism", value_col]).copy()
    if exclude_direct_laplace:
        data = data[data["mechanism"] != "DirectLaplace"]
    if query_ids is not None:
        data = data[data["query_id"].isin(query_ids)]
    if data.empty:
        print(f"Skipping {path.name}: no matching error rows")
        return
    if log_y:
        data[value_col] = data[value_col].clip(lower=LOG_FLOOR)
    pivot = data.pivot_table(
        index="query_id", columns="mechanism", values=value_col, aggfunc="mean"
    )
    width = max(9, 1.6 * len(pivot.index) + 3)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    pivot.plot(kind="bar", ax=ax)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Query")
    ax.set_ylabel(value_col)
    ax.legend(title="Mechanism")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def _plot_error_boxplot(runs: pd.DataFrame, figures_dir: Path) -> None:
    if runs.empty or not {"mechanism", "abs_error"}.issubset(runs.columns):
        print("Skipping error_boxplot_by_mechanism.png: run data missing")
        return
    grouped = [
        (name, group["abs_error"].dropna().to_numpy())
        for name, group in runs.groupby("mechanism")
    ]
    grouped = [(name, values) for name, values in grouped if len(values) > 0]
    groups = [values for _name, values in grouped]
    labels = [name for name, _values in grouped]
    if not groups:
        print("Skipping error_boxplot_by_mechanism.png: no groups")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title("Absolute Error by Mechanism")
    ax.set_xlabel("Mechanism")
    ax.set_ylabel("Absolute error")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, figures_dir / "error_boxplot_by_mechanism.png")


def _plot_error_vs_epsilon(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or summary["epsilon"].nunique(dropna=True) <= 1:
        print("Skipping error_vs_epsilon.png: need multiple epsilon values")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mechanism, group in summary.groupby("mechanism"):
        ax.plot(group["epsilon"], group["mean_rel_error"], marker="o", label=mechanism)
    ax.set_title("Relative Error vs Epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("mean_rel_error")
    ax.legend(title="Mechanism")
    _save(fig, figures_dir / "error_vs_epsilon.png")


def _plot_error_vs_gsq(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or summary["gsq"].nunique(dropna=True) <= 1:
        print("Skipping error_vs_gsq_logx.png: need multiple GSQ values")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mechanism, group in summary.groupby("mechanism"):
        clean = group.dropna(subset=["gsq"])
        ax.plot(clean["gsq"], clean["mean_rel_error"], marker="o", label=mechanism)
    ax.set_xscale("log")
    ax.set_title("Relative Error vs GSQ")
    ax.set_xlabel("GSQ")
    ax.set_ylabel("mean_rel_error")
    ax.legend(title="Mechanism")
    _save(fig, figures_dir / "error_vs_gsq_logx.png")


def _plot_error_runtime_vs_scale(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or summary["scale_factor"].nunique(dropna=True) <= 1:
        print("Skipping error_runtime_vs_scale.png: need multiple scale factors")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mechanism, group in summary.groupby("mechanism"):
        ax.plot(
            group["scale_factor"],
            group["mean_runtime_sec"],
            marker="o",
            label=mechanism,
        )
    ax.set_title("Runtime vs Scale Factor")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Mean runtime (sec)")
    ax.legend(title="Mechanism")
    _save(fig, figures_dir / "error_runtime_vs_scale.png")


def _plot_tau_sweep(tau_trace: pd.DataFrame, figures_dir: Path) -> None:
    if tau_trace.empty or not {"tau", "candidate_value"}.issubset(tau_trace.columns):
        print("Skipping tau_sweep_with_r2t_marker.png: tau trace missing")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for query_id, group in tau_trace.groupby("query_id"):
        ordered = group.sort_values("tau")
        ax.plot(ordered["tau"], ordered["candidate_value"], marker="o", label=query_id)
        winners = ordered[ordered["is_winner"].astype(bool)]
        if not winners.empty:
            ax.scatter(
                winners["tau"],
                winners["candidate_value"],
                marker="*",
                s=120,
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set_title("R2T Tau Sweep")
    ax.set_xlabel("tau")
    ax.set_ylabel("candidate_value")
    ax.legend(title="Query")
    _save(fig, figures_dir / "tau_sweep_with_r2t_marker.png")


def _plot_contribution_summary(stats: pd.DataFrame, figures_dir: Path) -> None:
    if stats.empty or not {"query_id", "max_contribution"}.issubset(stats.columns):
        print("Skipping contribution_ccdf.png: contribution stats missing")
        return
    _plot_contribution_summary_variant(
        stats,
        figures_dir / "contribution_ccdf.png",
        "Contribution Distribution Summary",
    )
    _plot_contribution_summary_variant(
        stats,
        figures_dir / "contribution_summary_logy.png",
        "Contribution Distribution Summary (log scale)",
        log_y=True,
    )
    _plot_contribution_summary_variant(
        stats,
        figures_dir / "contribution_summary_revenue_queries.png",
        "Contribution Summary for Revenue Queries",
        query_ids=REVENUE_QUERY_IDS,
    )
    _plot_contribution_summary_variant(
        stats,
        figures_dir / "contribution_summary_count_queries.png",
        "Contribution Summary for Count and SPJA Queries",
        query_ids=COUNT_QUERY_IDS,
        log_y=True,
    )


def _plot_contribution_summary_variant(
    stats: pd.DataFrame,
    path: Path,
    title: str,
    *,
    log_y: bool = False,
    query_ids: set[str] | None = None,
) -> None:
    value_cols = [
        col
        for col in ["max_contribution", "p95", "median_contribution"]
        if col in stats.columns
    ]
    if not value_cols:
        print(f"Skipping {path.name}: contribution value columns missing")
        return
    data = stats.copy()
    if query_ids is not None:
        data = data[data["query_id"].isin(query_ids)]
    if data.empty:
        print(f"Skipping {path.name}: no matching contribution rows")
        return
    if log_y:
        data[value_cols] = data[value_cols].clip(lower=LOG_FLOOR)
    width = max(9, 1.6 * len(data["query_id"].unique()) + 3)
    fig, ax = plt.subplots(figsize=(width, 5.5))
    data.plot(
        x="query_id",
        y=value_cols,
        kind="bar",
        ax=ax,
    )
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Query")
    ax.set_ylabel("Contribution value")
    ax.legend(title="Statistic")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, path)


def _needs_log_scale(data: pd.DataFrame, value_col: str) -> bool:
    if data.empty or value_col not in data:
        return False
    values = data[value_col].dropna()
    values = values[values > 0]
    if len(values) < 2:
        return False
    return values.max() / values.min() > 100


def _plot_runtime_error_pareto(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or not {"mean_runtime_sec", "mean_rel_error"}.issubset(summary):
        print("Skipping runtime_vs_error_pareto.png: summary data missing")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for mechanism, group in summary.groupby("mechanism"):
        ax.scatter(group["mean_runtime_sec"], group["mean_rel_error"], label=mechanism)
    ax.set_title("Runtime vs Relative Error")
    ax.set_xlabel("Mean runtime (sec)")
    ax.set_ylabel("mean_rel_error")
    ax.legend(title="Mechanism")
    _save(fig, figures_dir / "runtime_vs_error_pareto.png")


def _plot_shifted_inverse_comparison(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty or "mechanism" not in summary:
        print("Skipping optional_shifted_inverse_comparison.png: summary missing")
        return
    shifted = summary[summary["mechanism"].astype(str).str.contains("ShiftedInverse")]
    if shifted.empty:
        print("Skipping optional_shifted_inverse_comparison.png: no shifted results")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    shifted.pivot_table(
        index="query_id",
        columns="mechanism",
        values="mean_rel_error",
        aggfunc="mean",
    ).plot(kind="bar", ax=ax)
    ax.set_title("Optional ShiftedInverse Comparison")
    ax.set_xlabel("Query")
    ax.set_ylabel("mean_rel_error")
    ax.legend(title="Mechanism")
    _save(fig, figures_dir / "optional_shifted_inverse_comparison.png")


if __name__ == "__main__":
    raise SystemExit(main())
