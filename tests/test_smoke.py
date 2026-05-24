"""Smoke tests for the scaffold behavior."""

from __future__ import annotations

import numpy as np

from r2t_tpch.contributions import ContributionBounds, ContributionData
from r2t_tpch.load_tpch import load_tpch
from r2t_tpch.lp_truncation import build_placeholder_plan
from r2t_tpch.mechanisms_r2t import R2TConfig, validate_r2t_config
from r2t_tpch.metrics import mean_absolute_error
from r2t_tpch.queries import (
    QueryType,
    list_queries,
    q3_spja_order_count,
    q5_sja_revenue,
    q7_sja_revenue,
    q10_spja_customer_count,
    q12_sja_count,
)
from r2t_tpch.schema import table_names


def test_registered_queries_include_sja_and_spja() -> None:
    query_types = {query.query_type for query in list_queries()}
    query_names = {query.name for query in list_queries()}

    assert QueryType.SJA in query_types
    assert QueryType.SPJA in query_types
    assert {
        "q5_sja_revenue",
        "q7_sja_revenue",
        "q12_sja_count",
        "q3_spja_order_count",
        "q10_spja_customer_count",
    }.issubset(query_names)


def test_schema_has_core_tpch_tables() -> None:
    names = set(table_names())

    assert {"customer", "orders", "lineitem"}.issubset(names)


def test_basic_placeholders_work() -> None:
    bounds = ContributionBounds(lower=1.0, upper=3.5)
    plan = build_placeholder_plan("lineitem_revenue_sum", threshold=10.0)

    assert bounds.width() == 2.5
    assert plan.query_name == "lineitem_revenue_sum"
    assert plan.threshold == 10.0


def test_privacy_config_validation() -> None:
    validate_r2t_config(R2TConfig(epsilon=1.0, delta=0.0))


def test_mean_absolute_error() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    estimate = np.array([2.0, 2.0, 1.0])

    assert mean_absolute_error(actual, estimate) == 1.0


def test_loader_creates_tiny_data_when_tbl_files_are_absent(tmp_path) -> None:
    con = load_tpch(tmp_path / "missing", use_cache=False)

    counts = {
        name: con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in table_names()
    }
    assert all(count > 0 for count in counts.values())


def test_sja_queries_return_contribution_data_and_sum_to_true_answer(tmp_path) -> None:
    con = load_tpch(tmp_path / "missing", use_cache=False)

    results = [
        q5_sja_revenue(con),
        q5_sja_revenue(con, multi_private=True),
        q7_sja_revenue(con),
        q12_sja_count(con),
        q12_sja_count(con, privacy_mode="customer"),
    ]

    for result in results:
        assert isinstance(result, ContributionData)
        assert result.query_type == "SJA"
        assert result.private_ids_per_join_row
        assert result.projection_keys is None
        assert result.projected_values is None
        assert result.true_answer == float(result.join_values.sum())


def test_spja_queries_project_distinct_keys_and_sum_to_true_answer(tmp_path) -> None:
    con = load_tpch(tmp_path / "missing", use_cache=False)

    for result in [q3_spja_order_count(con), q10_spja_customer_count(con)]:
        assert isinstance(result, ContributionData)
        assert result.query_type == "SPJA"
        assert result.projection_keys
        assert result.projected_values
        assert result.true_answer == sum(result.projected_values.values())


def test_empty_query_behavior_returns_zero(tmp_path) -> None:
    con = load_tpch(tmp_path / "missing", use_cache=False)

    empty_results = [
        q5_sja_revenue(con, region="NO_SUCH_REGION"),
        q7_sja_revenue(con, nation1="NO_SUCH_1", nation2="NO_SUCH_2"),
        q12_sja_count(con, shipmodes=("NO_SUCH_MODE",)),
        q3_spja_order_count(con, segment="NO_SUCH_SEGMENT"),
        q10_spja_customer_count(con, date="2099-01-01"),
    ]

    for result in empty_results:
        assert isinstance(result, ContributionData)
        assert result.true_answer == 0.0
