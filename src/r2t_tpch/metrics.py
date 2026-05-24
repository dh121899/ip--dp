"""Metrics for reproducibility experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from r2t_tpch.contributions import ContributionData
from r2t_tpch.load_tpch import row_count_metadata

ROW_COUNT_FIELDS = (
    "row_count_region",
    "row_count_nation",
    "row_count_supplier",
    "row_count_customer",
    "row_count_orders",
    "row_count_lineitem",
    "row_count_part",
    "row_count_partsupp",
)
SUCCESS_STATUSES = {"ok", "success"}
NON_RESULT_STATUSES = {"not_applicable", "failed", "error", "lp_failed"}


def is_success_status(status: str) -> bool:
    """Return whether a status represents a valid mechanism answer."""

    return str(status).lower() in SUCCESS_STATUSES


def is_non_result_status(status: str) -> bool:
    """Return whether a status should be excluded from error metrics."""

    normalized = str(status).lower()
    return normalized in NON_RESULT_STATUSES or normalized.startswith(
        ("failed", "error", "lp_failed")
    )


def mean_absolute_error(actual: np.ndarray, estimate: np.ndarray) -> float:
    """Compute mean absolute error for matching arrays."""

    actual_arr = np.asarray(actual, dtype=float)
    estimate_arr = np.asarray(estimate, dtype=float)
    if actual_arr.shape != estimate_arr.shape:
        msg = "actual and estimate must have matching shapes"
        raise ValueError(msg)
    return float(np.mean(np.abs(actual_arr - estimate_arr)))


def compute_run_metrics(
    *,
    contrib_data: ContributionData,
    mechanism_result: Any,
    metadata: dict[str, Any],
    run_id: int,
    seed: int,
    query_time_sec: float | None = None,
) -> dict[str, Any]:
    """Compute one CSV row for a query/mechanism/run result."""

    true_answer = float(contrib_data.true_answer)
    status = str(getattr(mechanism_result, "status", "unknown"))
    is_valid_result = is_success_status(status)
    private_answer = (
        float(getattr(mechanism_result, "answer", np.nan))
        if is_valid_result
        else np.nan
    )
    signed_error = private_answer - true_answer if is_valid_result else np.nan
    abs_error = abs(signed_error) if is_valid_result else np.nan
    rel_error = (
        abs_error / abs(true_answer)
        if is_valid_result and true_answer != 0
        else np.nan
    )
    runtime_sec = float(getattr(mechanism_result, "runtime_sec", 0.0))
    lp_time_sec = float(getattr(mechanism_result, "lp_time_sec", 0.0))
    query_time = (
        float(contrib_data.extraction_time_sec)
        if query_time_sec is None
        else float(query_time_sec)
    )

    row = {
        "data_mode": metadata.get("data_mode"),
        "data_source": metadata.get("data_source"),
        "data_dir": metadata.get("data_dir"),
        "scale_factor": metadata.get("scale_factor"),
        "expected_scale_factor": metadata.get("expected_scale_factor"),
        "query_id": contrib_data.query_id,
        "query_type": contrib_data.query_type,
        "mechanism": getattr(mechanism_result, "mechanism", "unknown"),
        "epsilon": getattr(mechanism_result, "epsilon", np.nan),
        "beta": getattr(mechanism_result, "beta", None),
        "gsq": getattr(mechanism_result, "gsq", None),
        "run_id": int(run_id),
        "seed": int(seed),
        "true_answer": true_answer,
        "private_answer": private_answer,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": signed_error**2,
        "winner_tau": getattr(mechanism_result, "winner_tau", None),
        "noise_scale": getattr(mechanism_result, "noise_scale", None),
        "query_time_sec": query_time,
        "lp_time_sec": lp_time_sec,
        "total_time_sec": query_time + runtime_sec,
        "status": status,
        "is_valid_result": is_valid_result,
    }
    row.update({field: metadata.get(field) for field in ROW_COUNT_FIELDS})
    return row


def summarize_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run metrics by data/query/mechanism parameters."""

    if runs_df.empty:
        return pd.DataFrame()

    group_cols = [
        "data_mode",
        "data_source",
        "scale_factor",
        "query_id",
        "query_type",
        "mechanism",
        "epsilon",
        "gsq",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in runs_df.groupby(group_cols, dropna=False):
        valid = _valid_rows(group)
        signed = valid["signed_error"].astype(float)
        abs_error = valid["abs_error"].astype(float)
        rel_error = valid["rel_error"].astype(float)
        statuses = group["status"].astype(str).str.lower()
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n_runs": int(len(group)),
                "n_total": int(len(group)),
                "n_valid": int(len(valid)),
                "n_ok": int((statuses == "ok").sum()),
                "n_success": int((statuses == "success").sum()),
                "n_not_applicable": int((statuses == "not_applicable").sum()),
                "n_failed": int(statuses.str.startswith(("failed", "error")).sum()),
                "mean_abs_error": float(abs_error.mean()),
                "median_abs_error": float(abs_error.median()),
                "trimmed_mean_abs_error": _trimmed_mean(abs_error.to_numpy(), 0.2),
                "mean_rel_error": float(rel_error.mean()),
                "median_rel_error": float(rel_error.median()),
                "rmse": float(np.sqrt(valid["squared_error"].astype(float).mean())),
                "bias": float(signed.mean()),
                "std_error": float(signed.std(ddof=0)),
                "p90_abs_error": float(abs_error.quantile(0.90)),
                "p95_abs_error": float(abs_error.quantile(0.95)),
                "mean_runtime_sec": float(group["total_time_sec"].astype(float).mean()),
                "timeout_rate": float((group["status"] == "timeout").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_contribution_stats(
    contrib_data: ContributionData,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-query contribution distribution metadata."""

    metadata = metadata or {}
    private_contribs = _per_private_raw_contributions(contrib_data)
    values = np.asarray(list(private_contribs.values()), dtype=float)
    if len(values) == 0:
        values = np.asarray([0.0], dtype=float)

    sorted_values = np.sort(values)[::-1]
    n_projection_keys = (
        len(contrib_data.projected_values or {})
        if contrib_data.projected_values is not None
        else 0
    )
    row = {
        "data_mode": metadata.get("data_mode"),
        "data_source": metadata.get("data_source"),
        "data_dir": metadata.get("data_dir"),
        "scale_factor": metadata.get("scale_factor"),
        "expected_scale_factor": metadata.get("expected_scale_factor"),
        "query_id": contrib_data.query_id,
        "query_type": contrib_data.query_type,
        "n_private_entities": len(private_contribs),
        "n_join_rows": int(len(contrib_data.join_values)),
        "n_projection_keys": int(n_projection_keys),
        "true_answer": float(contrib_data.true_answer),
        "max_contribution": float(np.max(values)),
        "mean_contribution": float(np.mean(values)),
        "median_contribution": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "gini": _gini(values),
        "top1_share": _top_share(sorted_values, 1),
        "top10_share": _top_share(sorted_values, 10),
        "dsq": np.nan,
        "isq_proxy": np.nan,
    }
    if contrib_data.query_type == "SJA":
        row["dsq"] = row["max_contribution"]
    elif contrib_data.query_type == "SPJA":
        row["isq_proxy"] = row["max_contribution"]
    row.update({field: metadata.get(field) for field in ROW_COUNT_FIELDS})
    return row


def metadata_with_row_counts(
    *,
    data_mode: str,
    data_source: str,
    data_dir: str,
    scale_factor: float | None,
    expected_scale_factor: float | None,
    row_counts: dict[str, int],
) -> dict[str, Any]:
    """Build standard experiment metadata including row counts."""

    metadata: dict[str, Any] = {
        "data_mode": data_mode,
        "data_source": data_source,
        "data_dir": data_dir,
        "scale_factor": scale_factor,
        "expected_scale_factor": expected_scale_factor,
    }
    metadata.update(row_count_metadata(row_counts))
    return metadata


def _per_private_raw_contributions(contrib_data: ContributionData) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for row_value, private_ids in zip(
        contrib_data.join_values,
        contrib_data.private_ids_per_join_row,
        strict=True,
    ):
        for private_id in set(private_ids):
            values[private_id] += float(row_value)
    return dict(values)


def _valid_rows(group: pd.DataFrame) -> pd.DataFrame:
    if "is_valid_result" in group:
        valid_mask = group["is_valid_result"].map(_coerce_bool)
        return group[valid_mask]
    return group[group["status"].map(is_success_status)]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return float("nan")
    clean.sort()
    trim = int(len(clean) * trim_fraction)
    if trim == 0 or 2 * trim >= len(clean):
        return float(clean.mean())
    return float(clean[trim:-trim].mean())


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.all(arr == 0):
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_arr) / np.sum(sorted_arr) - (n + 1)) / n)


def _top_share(sorted_desc_values: np.ndarray, n_top: int) -> float:
    total = float(sorted_desc_values.sum())
    if total == 0:
        return 0.0
    return float(sorted_desc_values[:n_top].sum() / total)
