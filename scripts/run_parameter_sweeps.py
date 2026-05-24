"""Run real-data epsilon and GSQ parameter sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError

from r2t_tpch.metrics import summarize_runs

EPSILON_VALUES = ["0.1", "0.2", "0.5", "1.0", "2.0", "5.0"]
GSQ_VALUES = [
    "256",
    "512",
    "1024",
    "4096",
    "16384",
    "65536",
    "262144",
    "524288",
    "1048576",
]
ALL_QUERIES = [
    "q5_sja_revenue",
    "q7_sja_revenue",
    "q12_sja_count",
    "q3_spja_order_count",
    "q10_spja_customer_count",
]
COUNT_QUERIES = ["q12_sja_count", "q10_spja_customer_count"]
COUNT_QUERY_IDS = {
    "q12_sja_count",
    "q3_spja_order_count",
    "q10_spja_customer_count",
}
REVENUE_QUERY_IDS = {"q5_sja_revenue", "q7_sja_revenue"}
MECHANISMS = ["DirectLaplace", "FixedTau", "R2T"]
FIXED_TAUS = ["128", "512", "2048", "8192", "32768", "131072"]
LOG_FLOOR = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SF=0.1 parameter sweeps.")
    parser.add_argument("--sweep", choices=["epsilon", "gsq"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scale-factor", type=float, required=True)
    parser.add_argument("--expected-scale-factor", type=float, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-out-dir", required=True)
    parser.add_argument("--base-figures-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_out_dir = Path(args.base_out_dir)
    base_figures_dir = Path(args.base_figures_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    base_figures_dir.mkdir(parents=True, exist_ok=True)

    commands = build_sweep_commands(args, python_executable=sys.executable)
    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

    combine_sweep_outputs(base_out_dir)
    generate_sweep_plots(args.sweep, base_out_dir, base_figures_dir)
    return 0


def build_sweep_commands(
    args: argparse.Namespace, python_executable: str
) -> list[list[str]]:
    """Build subprocess commands for the requested sweep."""

    commands: list[list[str]] = []
    if args.sweep == "epsilon":
        for epsilon in EPSILON_VALUES:
            out_dir = Path(args.base_out_dir) / f"eps_{epsilon}"
            commands.append(
                _base_command(args, python_executable, out_dir)
                + [
                    "--queries",
                    *ALL_QUERIES,
                    "--mechanisms",
                    *MECHANISMS,
                    "--epsilon",
                    epsilon,
                    "--beta",
                    "0.1",
                    "--gsq",
                    "524288",
                    "--fixed-taus",
                    *FIXED_TAUS,
                ]
            )
    else:
        for gsq in GSQ_VALUES:
            out_dir = Path(args.base_out_dir) / f"gsq_{gsq}"
            commands.append(
                _base_command(args, python_executable, out_dir)
                + [
                    "--queries",
                    *COUNT_QUERIES,
                    "--mechanisms",
                    *MECHANISMS,
                    "--epsilon",
                    "1.0",
                    "--beta",
                    "0.1",
                    "--gsq",
                    gsq,
                    "--fixed-taus",
                    *FIXED_TAUS,
                ]
            )
    return commands


def combine_sweep_outputs(base_out_dir: Path) -> None:
    """Combine per-parameter experiment CSVs under `base_out_dir`."""

    combined_runs: pd.DataFrame | None = None
    for file_name in [
        "results_runs.csv",
        "tau_trace.csv",
        "contribution_stats.csv",
    ]:
        frames = []
        for path in sorted(base_out_dir.glob(f"*/{file_name}")):
            if path.stat().st_size == 0:
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "sweep_subdir", path.parent.name)
            frame = _fill_sweep_value_columns(frame, path.parent.name)
            frames.append(frame)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if file_name == "contribution_stats.csv":
                combined = combined.drop_duplicates(
                    subset=["sweep_subdir", "query_id", "query_type"]
                )
            combined.to_csv(base_out_dir / file_name, index=False)
            if file_name == "results_runs.csv":
                combined_runs = combined
            print(f"Wrote {base_out_dir / file_name}")
        else:
            print(f"No {file_name} files found under {base_out_dir}")
    if combined_runs is not None:
        summarize_runs(combined_runs).to_csv(
            base_out_dir / "results_summary.csv", index=False
        )
        print(f"Wrote {base_out_dir / 'results_summary.csv'}")


def generate_sweep_plots(sweep: str, base_out_dir: Path, figures_dir: Path) -> None:
    """Generate parameter-sweep plots from combined CSVs."""

    summary = _read_csv(base_out_dir / "results_summary.csv")
    runs = _read_csv(base_out_dir / "results_runs.csv")
    if summary.empty:
        print("Skipping sweep plots: combined summary missing")
        return
    valid_summary = _valid_summary(summary)
    valid_runs = _valid_runs(runs)
    if sweep == "epsilon":
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon.png",
        )
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon (log scale)",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon_logy.png",
            logy=True,
        )
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon excluding DirectLaplace",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon_without_directlaplace.png",
            exclude_direct_laplace=True,
        )
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon for Count and SPJA Queries",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon_count_queries.png",
            query_ids=COUNT_QUERY_IDS,
            logy=True,
        )
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon for Revenue Queries",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon_revenue_queries.png",
            query_ids=REVENUE_QUERY_IDS,
            logy=_needs_log_scale(
                valid_summary[valid_summary["query_id"].isin(REVENUE_QUERY_IDS)],
                "median_rel_error",
            ),
        )
        _plot_line(
            valid_summary,
            x_col="epsilon",
            y_col="mean_runtime_sec",
            title="Runtime vs Epsilon",
            xlabel="epsilon",
            ylabel="mean_runtime_sec",
            path=figures_dir / "runtime_vs_epsilon.png",
        )
        _plot_query_mechanism_lines(
            valid_summary,
            x_col="epsilon",
            y_col="median_rel_error",
            title="Median Relative Error vs Epsilon by Query",
            xlabel="epsilon",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_epsilon_by_query.png",
        )
    else:
        _plot_line(
            valid_summary,
            x_col="gsq",
            y_col="median_rel_error",
            title="Median Relative Error vs GSQ",
            xlabel="GSQ",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_gsq_logx.png",
            logx=True,
        )
        _plot_line(
            valid_summary,
            x_col="gsq",
            y_col="median_rel_error",
            title="Median Relative Error vs GSQ (log x and log y)",
            xlabel="GSQ",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_gsq_logx_logy.png",
            logx=True,
            logy=True,
        )
        _plot_line(
            valid_summary,
            x_col="gsq",
            y_col="median_rel_error",
            title="Median Relative Error vs GSQ excluding DirectLaplace",
            xlabel="GSQ",
            ylabel="median_rel_error",
            path=figures_dir / "error_vs_gsq_without_directlaplace.png",
            logx=True,
            exclude_direct_laplace=True,
        )
        _plot_line(
            valid_summary,
            x_col="gsq",
            y_col="mean_runtime_sec",
            title="Runtime vs GSQ",
            xlabel="GSQ",
            ylabel="mean_runtime_sec",
            path=figures_dir / "runtime_vs_gsq_logx.png",
            logx=True,
        )
        _plot_r2t_winner_tau(valid_runs, figures_dir / "r2t_winner_tau_vs_gsq.png")


def _base_command(
    args: argparse.Namespace, python_executable: str, out_dir: Path
) -> list[str]:
    return [
        python_executable,
        "-m",
        "r2t_tpch.run_experiment",
        "--data-mode",
        "real",
        "--data-dir",
        args.data_dir,
        "--scale-factor",
        str(args.scale_factor),
        "--expected-scale-factor",
        str(args.expected_scale_factor),
        "--validate-row-counts",
        "--allow-approx-row-counts",
        "--out-dir",
        str(out_dir),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
    ]


def _fill_sweep_value_columns(frame: pd.DataFrame, subdir_name: str) -> pd.DataFrame:
    patched = frame.copy()
    if subdir_name.startswith("eps_"):
        patched["epsilon"] = float(subdir_name.removeprefix("eps_"))
    if subdir_name.startswith("gsq_"):
        patched["gsq"] = float(subdir_name.removeprefix("gsq_"))
    return patched


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _valid_summary(summary: pd.DataFrame) -> pd.DataFrame:
    filtered = summary.copy()
    if "n_valid" in filtered:
        filtered = filtered[filtered["n_valid"].fillna(0).astype(float) > 0]
    return filtered.dropna(subset=["median_rel_error"])


def _valid_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    if "is_valid_result" in runs:
        return runs[runs["is_valid_result"].map(_coerce_bool)]
    return runs[runs["status"].astype(str).str.lower().isin({"ok", "success"})]


def _plot_line(
    summary: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    logx: bool = False,
    logy: bool = False,
    exclude_direct_laplace: bool = False,
    query_ids: set[str] | None = None,
) -> None:
    if summary.empty or not {x_col, y_col, "mechanism"}.issubset(summary.columns):
        print(f"Skipping {path.name}: data missing")
        return
    data = summary.dropna(subset=[x_col, y_col, "mechanism"]).copy()
    if exclude_direct_laplace:
        data = data[data["mechanism"] != "DirectLaplace"]
    if query_ids is not None:
        data = data[data["query_id"].isin(query_ids)]
    if data.empty:
        print(f"Skipping {path.name}: no matching valid rows")
        return
    if logy:
        data[y_col] = data[y_col].clip(lower=LOG_FLOOR)
    grouped = (
        data.groupby([x_col, "mechanism"], dropna=False)[y_col]
        .mean()
        .reset_index()
        .sort_values(x_col)
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mechanism, group in grouped.groupby("mechanism"):
        ax.plot(group[x_col], group[y_col], marker="o", label=mechanism)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Mechanism")
    _save(fig, path)


def _plot_query_mechanism_lines(
    summary: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    if summary.empty or len(summary["query_id"].unique()) * len(
        summary["mechanism"].unique()
    ) > 20:
        print(f"Skipping {path.name}: plot would be too cluttered")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for (query_id, mechanism), group in summary.groupby(["query_id", "mechanism"]):
        ordered = group.sort_values(x_col)
        ax.plot(
            ordered[x_col],
            ordered[y_col],
            marker="o",
            label=f"{query_id}:{mechanism}",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    _save(fig, path)


def _plot_r2t_winner_tau(runs: pd.DataFrame, path: Path) -> None:
    if runs.empty or not {"gsq", "winner_tau", "query_id", "mechanism"}.issubset(
        runs.columns
    ):
        print(f"Skipping {path.name}: R2T winner data missing")
        return
    r2t_runs = runs[runs["mechanism"] == "R2T"].dropna(subset=["winner_tau"])
    if r2t_runs.empty:
        print(f"Skipping {path.name}: no valid R2T rows")
        return
    grouped = (
        r2t_runs.groupby(["gsq", "query_id"], dropna=False)["winner_tau"]
        .median()
        .reset_index()
        .sort_values("gsq")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for query_id, group in grouped.groupby("query_id"):
        ax.plot(group["gsq"], group["winner_tau"], marker="o", label=query_id)
    ax.set_xscale("log")
    ax.set_title("R2T Winner Tau vs GSQ")
    ax.set_xlabel("GSQ")
    ax.set_ylabel("median winner_tau")
    ax.legend(title="Query")
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved {path}")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _needs_log_scale(data: pd.DataFrame, value_col: str) -> bool:
    if data.empty or value_col not in data:
        return False
    values = data[value_col].dropna()
    values = values[values > 0]
    if len(values) < 2:
        return False
    return values.max() / values.min() > 100


if __name__ == "__main__":
    raise SystemExit(main())
