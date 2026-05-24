"""Tests for TPC-H data-mode helpers."""

from __future__ import annotations

import pytest

from r2t_tpch.load_tpch import load_tpch_with_mode, validate_tpch_row_counts


def test_row_count_validation_passes_for_approx_sf001() -> None:
    validate_tpch_row_counts(
        {
            "region": 5,
            "nation": 25,
            "supplier": 100,
            "customer": 1500,
            "orders": 15000,
            "part": 2000,
            "partsupp": 8000,
            "lineitem": 60175,
        },
        sf=0.01,
        strict=False,
    )


def test_real_mode_missing_tbl_files_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required TPC-H"):
        load_tpch_with_mode(tmp_path, data_mode="real")

