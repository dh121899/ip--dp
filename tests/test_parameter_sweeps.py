"""Tests for parameter sweep orchestration helpers."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def _load_sweeps_module():
    path = Path("scripts/run_parameter_sweeps.py").resolve()
    spec = importlib.util.spec_from_file_location("run_parameter_sweeps", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_script_builds_commands() -> None:
    sweeps = _load_sweeps_module()
    args = argparse.Namespace(
        sweep="epsilon",
        data_dir="data/raw/sf0.1",
        scale_factor=0.1,
        expected_scale_factor=0.1,
        runs=10,
        seed=42,
        base_out_dir="results/sweeps/epsilon_sf0.1",
        base_figures_dir="figures/sweeps/epsilon_sf0.1",
    )

    command = sweeps.build_sweep_commands(args, "python")[0]

    assert "--data-mode" in command
    assert "real" in command
    assert command[command.index("--data-dir") + 1] == "data/raw/sf0.1"
    assert command[command.index("--scale-factor") + 1] == "0.1"
    assert command[command.index("--expected-scale-factor") + 1] == "0.1"
    assert "--validate-row-counts" in command
    assert "--allow-approx-row-counts" in command


def test_sweep_combination(tmp_path) -> None:
    sweeps = _load_sweeps_module()
    base = tmp_path / "sweep"
    for subdir in ["eps_0.1", "eps_0.2"]:
        out = base / subdir
        out.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "data_mode": "real",
                    "data_source": "real_tpch",
                    "scale_factor": 0.1,
                    "query_id": "q12_sja_count",
                    "query_type": "SJA",
                    "mechanism": "R2T",
                    "epsilon": 1.0,
                    "gsq": 524288.0,
                    "run_id": 0,
                    "seed": 42,
                    "true_answer": 10.0,
                    "private_answer": 11.0,
                    "signed_error": 1.0,
                    "abs_error": 1.0,
                    "rel_error": 0.1,
                    "squared_error": 1.0,
                    "total_time_sec": 0.1,
                    "status": "ok",
                    "is_valid_result": True,
                }
            ]
        ).to_csv(out / "results_runs.csv", index=False)
        pd.DataFrame(
            [
                {
                    "query_id": "q12_sja_count",
                    "mechanism": "R2T",
                    "epsilon": 1.0,
                    "median_rel_error": 0.1,
                    "n_valid": 1,
                }
            ]
        ).to_csv(out / "results_summary.csv", index=False)

    sweeps.combine_sweep_outputs(base)

    combined = pd.read_csv(base / "results_runs.csv")
    assert len(combined) == 2
    assert "sweep_subdir" in combined


def test_sweep_plot_functions(tmp_path) -> None:
    sweeps = _load_sweeps_module()
    base = tmp_path / "results"
    figures = tmp_path / "figures"
    base.mkdir()
    pd.DataFrame(
        [
            {
                "query_id": "q12_sja_count",
                "mechanism": "R2T",
                "epsilon": 0.1,
                "gsq": 524288,
                "median_rel_error": 0.2,
                "mean_runtime_sec": 0.1,
                "n_valid": 1,
            },
            {
                "query_id": "q12_sja_count",
                "mechanism": "R2T",
                "epsilon": 1.0,
                "gsq": 524288,
                "median_rel_error": 0.1,
                "mean_runtime_sec": 0.2,
                "n_valid": 1,
            },
            {
                "query_id": "q5_sja_revenue",
                "mechanism": "DirectLaplace",
                "epsilon": 0.1,
                "gsq": 524288,
                "median_rel_error": 10.0,
                "mean_runtime_sec": 0.05,
                "n_valid": 1,
            },
            {
                "query_id": "q5_sja_revenue",
                "mechanism": "DirectLaplace",
                "epsilon": 1.0,
                "gsq": 524288,
                "median_rel_error": 1.0,
                "mean_runtime_sec": 0.05,
                "n_valid": 1,
            },
        ]
    ).to_csv(base / "results_summary.csv", index=False)
    pd.DataFrame().to_csv(base / "results_runs.csv", index=False)

    sweeps.generate_sweep_plots("epsilon", base, figures)

    assert (figures / "error_vs_epsilon.png").exists()
    assert (figures / "error_vs_epsilon_logy.png").exists()
    assert (figures / "error_vs_epsilon_without_directlaplace.png").exists()
    assert (figures / "error_vs_epsilon_count_queries.png").exists()
    assert (figures / "error_vs_epsilon_revenue_queries.png").exists()
    assert (figures / "runtime_vs_epsilon.png").exists()
