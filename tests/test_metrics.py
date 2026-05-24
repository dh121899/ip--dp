"""Tests for experiment metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from r2t_tpch.contributions import ContributionData
from r2t_tpch.mechanisms_r2t import MechanismResult
from r2t_tpch.metrics import (
    compute_contribution_stats,
    compute_run_metrics,
    summarize_runs,
)


def test_metrics_summary_works_on_tiny_results_dataframe() -> None:
    runs = pd.DataFrame(
        [
            _run_row(0, 10.0, 11.0),
            _run_row(1, 10.0, 8.0),
            _run_row(2, 10.0, 13.0),
        ]
    )

    summary = summarize_runs(runs)

    assert len(summary) == 1
    assert summary.loc[0, "n_runs"] == 3
    assert summary.loc[0, "mean_abs_error"] == 2.0
    assert summary.loc[0, "rmse"] > 0


def test_contribution_stats_works_on_sja_and_spja() -> None:
    sja = ContributionData(
        query_id="q5_sja_revenue",
        query_type="SJA",
        join_values=np.asarray([10.0, 5.0, 1.0], dtype=np.float64),
        private_ids_per_join_row=[["A"], ["A"], ["B"]],
        projection_keys=None,
        projected_values=None,
        true_answer=16.0,
        extraction_time_sec=0.0,
    )
    spja = ContributionData(
        query_id="q3_spja_order_count",
        query_type="SPJA",
        join_values=np.asarray([1.0, 1.0], dtype=np.float64),
        private_ids_per_join_row=[["A"], ["B"]],
        projection_keys=[("ORDER", 1), ("ORDER", 1)],
        projected_values={("ORDER", 1): 1.0},
        true_answer=1.0,
        extraction_time_sec=0.0,
    )

    sja_stats = compute_contribution_stats(sja)
    spja_stats = compute_contribution_stats(spja)

    assert sja_stats["dsq"] == 15.0
    assert np.isnan(sja_stats["isq_proxy"])
    assert spja_stats["isq_proxy"] == 1.0
    assert np.isnan(spja_stats["dsq"])


def test_not_applicable_metrics_are_nan() -> None:
    data = _sja_contribution_data()
    result = MechanismResult(
        mechanism="ShiftedInverseSum",
        answer=16.0,
        true_answer=16.0,
        epsilon=1.0,
        beta=0.1,
        gsq=8.0,
        tau=None,
        winner_tau=None,
        noise_scale=None,
        status="not_applicable",
        runtime_sec=0.0,
        lp_time_sec=0.0,
        extra={},
    )

    row = compute_run_metrics(
        contrib_data=data,
        mechanism_result=result,
        metadata={},
        run_id=0,
        seed=42,
    )

    assert np.isnan(row["private_answer"])
    assert np.isnan(row["abs_error"])
    assert np.isnan(row["rel_error"])
    assert np.isnan(row["squared_error"])
    assert row["is_valid_result"] is False


def test_summary_excludes_not_applicable() -> None:
    runs = pd.DataFrame(
        [
            {**_run_row(0, 10.0, 12.0), "mechanism": "R2T", "is_valid_result": True},
            {
                **_run_row(1, 10.0, np.nan),
                "mechanism": "ShiftedInverseSum",
                "status": "not_applicable",
                "private_answer": np.nan,
                "signed_error": np.nan,
                "abs_error": np.nan,
                "rel_error": np.nan,
                "squared_error": np.nan,
                "is_valid_result": False,
            },
        ]
    )

    summary = summarize_runs(runs)
    shifted = summary[summary["mechanism"] == "ShiftedInverseSum"].iloc[0]
    r2t = summary[summary["mechanism"] == "R2T"].iloc[0]

    assert r2t["n_valid"] == 1
    assert r2t["mean_abs_error"] == 2.0
    assert shifted["n_valid"] == 0
    assert shifted["n_not_applicable"] == 1
    assert np.isnan(shifted["mean_abs_error"])


def _run_row(
    run_id: int, true_answer: float, private_answer: float
) -> dict[str, object]:
    signed = private_answer - true_answer
    return {
        "data_mode": "tiny",
        "data_source": "tiny_synthetic",
        "scale_factor": None,
        "query_id": "q5_sja_revenue",
        "query_type": "SJA",
        "mechanism": "DirectLaplace",
        "epsilon": 1.0,
        "gsq": 8.0,
        "run_id": run_id,
        "true_answer": true_answer,
        "private_answer": private_answer,
        "signed_error": signed,
        "abs_error": abs(signed),
        "rel_error": abs(signed) / true_answer,
        "squared_error": signed**2,
        "total_time_sec": 0.1,
        "status": "ok",
    }


def _sja_contribution_data() -> ContributionData:
    join_values = np.asarray([10.0, 5.0, 1.0], dtype=np.float64)
    return ContributionData(
        query_id="q5_sja_revenue",
        query_type="SJA",
        join_values=join_values,
        private_ids_per_join_row=[["A"], ["A"], ["B"]],
        projection_keys=None,
        projected_values=None,
        true_answer=16.0,
        extraction_time_sec=0.0,
    )
