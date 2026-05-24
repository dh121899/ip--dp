"""Tests for the optional lightweight ShiftedInverse module."""

from __future__ import annotations

import numpy as np

from r2t_tpch.contributions import ContributionData
from r2t_tpch.mechanisms_shifted_inverse import (
    apply_shifted_inverse,
    per_user_contributions,
    shifted_inverse_count_l1,
    shifted_inverse_sum_l1,
)


def test_per_user_contributions_on_simple_sja() -> None:
    data = _simple_sja()

    result = per_user_contributions(data)

    assert result == {"A": 15.0, "B": 1.0}


def test_shifted_inverse_sum_l1_returns_numeric_answer_and_trace() -> None:
    rng = np.random.default_rng(123)

    result, trace = shifted_inverse_sum_l1(
        [10.0, 5.0, 1.0], epsilon=1.0, beta=0.1, D=20.0, rng=rng
    )

    assert result.status == "success"
    assert result.tau0 > 0
    assert np.isfinite(result.answer)
    assert trace


def test_shifted_inverse_count_l1_returns_numeric_answer() -> None:
    rng = np.random.default_rng(123)

    result, trace = shifted_inverse_count_l1(
        [3.0, 2.0, 1.0], epsilon=1.0, beta=0.1, D=10.0, rng=rng
    )

    assert result.mechanism == "ShiftedInverseCount"
    assert result.status == "success"
    assert np.isfinite(result.answer)
    assert trace


def test_apply_shifted_inverse_on_sja_returns_success() -> None:
    rng = np.random.default_rng(123)

    result, trace = apply_shifted_inverse(
        _simple_sja(), epsilon=1.0, beta=0.1, D=20.0, rng=rng
    )

    assert result.status == "success"
    assert np.isfinite(result.answer)
    assert trace


def test_apply_shifted_inverse_on_spja_returns_not_applicable() -> None:
    rng = np.random.default_rng(123)

    result, trace = apply_shifted_inverse(
        _simple_spja(), epsilon=1.0, beta=0.1, D=2.0, rng=rng
    )

    assert result.status == "not_applicable"
    assert np.isnan(result.answer)
    assert trace == []


def _simple_sja() -> ContributionData:
    join_values = np.asarray([10.0, 5.0, 1.0], dtype=np.float64)
    return ContributionData(
        query_id="q5_sja_revenue",
        query_type="SJA",
        join_values=join_values,
        private_ids_per_join_row=[["A"], ["A"], ["B"]],
        projection_keys=None,
        projected_values=None,
        true_answer=float(join_values.sum()),
        extraction_time_sec=0.0,
    )


def _simple_spja() -> ContributionData:
    projection_keys = [("ORDER", 1), ("ORDER", 2)]
    projected_values = dict.fromkeys(projection_keys, 1.0)
    return ContributionData(
        query_id="q3_spja_order_count",
        query_type="SPJA",
        join_values=np.ones(len(projection_keys), dtype=np.float64),
        private_ids_per_join_row=[["A"], ["B"]],
        projection_keys=projection_keys,
        projected_values=projected_values,
        true_answer=float(sum(projected_values.values())),
        extraction_time_sec=0.0,
    )
