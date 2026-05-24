"""Tests for LP truncation formulations."""

from __future__ import annotations

import numpy as np

from r2t_tpch.contributions import ContributionData
from r2t_tpch.lp_truncation import truncate_sja_lp, truncate_spja_lp


def test_sja_lp_caps_private_contribution() -> None:
    data = _sja_data([5.0, 4.0, 3.0], [["A"], ["A"], ["B"]])

    result = truncate_sja_lp(data, tau=6.0)

    assert result.lp_success
    assert result.q_tau <= data.true_answer
    assert result.q_tau == 9.0
    assert result.n_private_entities == 2


def test_sja_large_tau_returns_true_answer() -> None:
    data = _sja_data([5.0, 4.0, 3.0], [["A"], ["A"], ["B"]])

    result = truncate_sja_lp(data, tau=9.0)

    assert result.lp_success
    assert result.q_tau == data.true_answer


def test_empty_sja_returns_zero() -> None:
    data = _sja_data([], [])

    result = truncate_sja_lp(data, tau=1.0)

    assert result.lp_success
    assert result.q_tau == 0.0


def test_spja_projection_overlap_counts_once() -> None:
    data = _spja_data(
        projection_keys=[("ORDER", 1), ("ORDER", 1)],
        private_ids=[["A"], ["B"]],
    )

    result = truncate_spja_lp(data, tau=1.0)

    assert result.lp_success
    assert result.q_tau <= data.true_answer
    assert result.q_tau == 1.0


def test_empty_spja_returns_zero() -> None:
    data = ContributionData(
        query_id="empty_spja",
        query_type="SPJA",
        join_values=np.asarray([], dtype=np.float64),
        private_ids_per_join_row=[],
        projection_keys=[],
        projected_values={},
        true_answer=0.0,
        extraction_time_sec=0.0,
    )

    result = truncate_spja_lp(data, tau=1.0)

    assert result.lp_success
    assert result.q_tau == 0.0


def _sja_data(values: list[float], private_ids: list[list[str]]) -> ContributionData:
    join_values = np.asarray(values, dtype=np.float64)
    return ContributionData(
        query_id="hand_sja",
        query_type="SJA",
        join_values=join_values,
        private_ids_per_join_row=private_ids,
        projection_keys=None,
        projected_values=None,
        true_answer=float(join_values.sum()),
        extraction_time_sec=0.0,
    )


def _spja_data(
    projection_keys: list[tuple[str, int]], private_ids: list[list[str]]
) -> ContributionData:
    projected_values = dict.fromkeys(projection_keys, 1.0)
    return ContributionData(
        query_id="hand_spja",
        query_type="SPJA",
        join_values=np.ones(len(projection_keys), dtype=np.float64),
        private_ids_per_join_row=private_ids,
        projection_keys=projection_keys,
        projected_values=projected_values,
        true_answer=float(sum(projected_values.values())),
        extraction_time_sec=0.0,
    )

