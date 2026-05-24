"""Tests for the experiment CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from r2t_tpch.contributions import ContributionData
from r2t_tpch.run_experiment import _run_shifted_inverse, main


def test_tiny_mode_creates_expected_csvs(tmp_path) -> None:
    out_dir = tmp_path / "results"

    exit_code = main(
        [
            "--data-mode",
            "tiny",
            "--out-dir",
            str(out_dir),
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

    assert exit_code == 0
    for file_name in [
        "results_runs.csv",
        "results_summary.csv",
        "tau_trace.csv",
        "contribution_stats.csv",
    ]:
        assert Path(out_dir, file_name).is_file()


def test_shifted_inverse_scheme_a() -> None:
    rng = np.random.default_rng(123)

    q5_sum, _trace = _run_shifted_inverse(
        requested="ShiftedInverseSum",
        contrib_data=_sja_data("q5_sja_revenue"),
        epsilon=1.0,
        beta=0.1,
        gsq=100.0,
        rng=rng,
    )
    q12_count, _trace = _run_shifted_inverse(
        requested="ShiftedInverseCount",
        contrib_data=_sja_data("q12_sja_count"),
        epsilon=1.0,
        beta=0.1,
        gsq=100.0,
        rng=rng,
    )
    q7_sum, _trace = _run_shifted_inverse(
        requested="ShiftedInverseSum",
        contrib_data=_sja_data("q7_sja_revenue"),
        epsilon=1.0,
        beta=0.1,
        gsq=100.0,
        rng=rng,
    )

    assert q5_sum.status == "success"
    assert q12_count.status == "success"
    assert q7_sum.status == "not_applicable"
    assert np.isnan(q7_sum.answer)

    for query_id in ["q3_spja_order_count", "q10_spja_customer_count"]:
        for mechanism in ["ShiftedInverseSum", "ShiftedInverseCount"]:
            result, _trace = _run_shifted_inverse(
                requested=mechanism,
                contrib_data=_spja_data(query_id),
                epsilon=1.0,
                beta=0.1,
                gsq=100.0,
                rng=rng,
            )
            assert result.status == "not_applicable"
            assert np.isnan(result.answer)


def _sja_data(query_id: str) -> ContributionData:
    join_values = np.asarray([10.0, 5.0, 1.0], dtype=np.float64)
    return ContributionData(
        query_id=query_id,
        query_type="SJA",
        join_values=join_values,
        private_ids_per_join_row=[["A"], ["A"], ["B"]],
        projection_keys=None,
        projected_values=None,
        true_answer=float(join_values.sum()),
        extraction_time_sec=0.0,
    )


def _spja_data(query_id: str) -> ContributionData:
    projection_keys = [("ORDER", 1), ("ORDER", 2)]
    projected_values = dict.fromkeys(projection_keys, 1.0)
    return ContributionData(
        query_id=query_id,
        query_type="SPJA",
        join_values=np.ones(len(projection_keys), dtype=np.float64),
        private_ids_per_join_row=[["A"], ["B"]],
        projection_keys=projection_keys,
        projected_values=projected_values,
        true_answer=float(sum(projected_values.values())),
        extraction_time_sec=0.0,
    )
