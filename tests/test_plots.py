"""Tests for plotting CLI on tiny experiment output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from r2t_tpch.plots import main as plots_main
from r2t_tpch.run_experiment import main as run_main


def test_plots_cli_creates_at_least_one_png(tmp_path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"

    run_main(
        [
            "--data-mode",
            "tiny",
            "--out-dir",
            str(results_dir),
            "--runs",
            "1",
            "--seed",
            "42",
            "--queries",
            "q5_sja_revenue",
            "--mechanisms",
            "DirectLaplace",
            "R2T",
            "--gsq",
            "8",
        ]
    )
    exit_code = plots_main(
        ["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    )

    assert exit_code == 0
    assert any(figures_dir.glob("*.png"))


def test_plots_ignore_not_applicable(tmp_path) -> None:
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    runs = pd.DataFrame(
        [
            {
                "query_id": "q5_sja_revenue",
                "mechanism": "R2T",
                "status": "ok",
                "is_valid_result": True,
                "abs_error": 1.0,
                "rel_error": 0.1,
                "total_time_sec": 0.01,
            },
            {
                "query_id": "q7_sja_revenue",
                "mechanism": "ShiftedInverseSum",
                "status": "not_applicable",
                "is_valid_result": False,
                "abs_error": np.nan,
                "rel_error": np.nan,
                "total_time_sec": 0.0,
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "query_id": "q5_sja_revenue",
                "mechanism": "R2T",
                "epsilon": 1.0,
                "gsq": 8.0,
                "scale_factor": np.nan,
                "n_valid": 1,
                "mean_rel_error": 0.1,
                "median_rel_error": 0.1,
                "mean_runtime_sec": 0.01,
            },
            {
                "query_id": "q7_sja_revenue",
                "mechanism": "ShiftedInverseSum",
                "epsilon": 1.0,
                "gsq": 8.0,
                "scale_factor": np.nan,
                "n_valid": 0,
                "mean_rel_error": np.nan,
                "median_rel_error": np.nan,
                "mean_runtime_sec": 0.0,
            },
        ]
    )
    runs.to_csv(results_dir / "results_runs.csv", index=False)
    summary.to_csv(results_dir / "results_summary.csv", index=False)
    pd.DataFrame().to_csv(results_dir / "tau_trace.csv", index=False)
    pd.DataFrame(
        [
            {
                "query_id": "q5_sja_revenue",
                "max_contribution": 1000.0,
                "p95": 500.0,
                "median_contribution": 100.0,
            },
            {
                "query_id": "q12_sja_count",
                "max_contribution": 10.0,
                "p95": 5.0,
                "median_contribution": 1.0,
            },
        ]
    ).to_csv(results_dir / "contribution_stats.csv", index=False)

    plot_args = ["--results-dir", str(results_dir), "--figures-dir", str(figures_dir)]
    assert plots_main(plot_args) == 0
    assert (figures_dir / "error_boxplot_by_mechanism.png").exists()
    assert (figures_dir / "error_bar_by_query_mechanism_logy.png").exists()
    assert (figures_dir / "error_bar_without_directlaplace.png").exists()
    assert (figures_dir / "error_bar_revenue_queries.png").exists()
    assert (figures_dir / "contribution_summary_logy.png").exists()
    assert (figures_dir / "contribution_summary_revenue_queries.png").exists()
    assert (figures_dir / "contribution_summary_count_queries.png").exists()
