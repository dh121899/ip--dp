"""LP-based truncation for R2T query answers."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from r2t_tpch.contributions import ContributionData


@dataclass(frozen=True)
class TruncationPlan:
    """Small compatibility placeholder retained for scaffold smoke tests."""

    query_name: str
    threshold: float
    objective_value: float | None = None


@dataclass(frozen=True)
class TruncationResult:
    """Metadata-rich LP truncation result."""

    q_tau: float
    tau: float
    query_type: str
    lp_status: str
    lp_success: bool
    lp_message: str
    lp_time_sec: float
    n_join_rows: int
    n_private_entities: int
    n_projection_keys: int
    n_variables: int
    n_constraints: int


def build_placeholder_plan(query_name: str, threshold: float = 0.0) -> TruncationPlan:
    """Create a placeholder truncation plan for smoke tests and examples."""

    return TruncationPlan(query_name=query_name, threshold=threshold)


def truncate_sja_lp(contrib_data: ContributionData, tau: float) -> TruncationResult:
    """Solve the SJA truncation LP for a fixed contribution cap `tau`."""

    _validate_tau(tau)
    start = time.perf_counter()
    values = _nonnegative_values(contrib_data.join_values)
    n_join_rows = len(values)
    private_to_rows = _private_to_rows(contrib_data.private_ids_per_join_row)
    n_private_entities = len(private_to_rows)

    if n_join_rows == 0:
        return _empty_result(
            tau=tau,
            query_type="SJA",
            lp_time_sec=time.perf_counter() - start,
            n_private_entities=n_private_entities,
        )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []

    for constraint_idx, row_indexes in enumerate(private_to_rows.values()):
        for row_idx in row_indexes:
            rows.append(constraint_idx)
            cols.append(row_idx)
            data.append(1.0)
        b_ub.append(float(tau))

    a_ub = coo_matrix(
        (data, (rows, cols)), shape=(n_private_entities, n_join_rows)
    ).tocsr()
    result = linprog(
        c=-np.ones(n_join_rows, dtype=float),
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=[(0.0, float(value)) for value in values],
        method="highs",
    )
    lp_time_sec = time.perf_counter() - start
    q_tau = float(-result.fun) if result.success else float("nan")

    return TruncationResult(
        q_tau=q_tau,
        tau=float(tau),
        query_type="SJA",
        lp_status=str(result.status),
        lp_success=bool(result.success),
        lp_message=str(result.message),
        lp_time_sec=lp_time_sec,
        n_join_rows=n_join_rows,
        n_private_entities=n_private_entities,
        n_projection_keys=0,
        n_variables=n_join_rows,
        n_constraints=n_private_entities,
    )


def truncate_spja_lp(contrib_data: ContributionData, tau: float) -> TruncationResult:
    """Solve the projection-aware SPJA truncation LP for a fixed cap `tau`."""

    _validate_tau(tau)
    start = time.perf_counter()
    values = _nonnegative_values(contrib_data.join_values)
    projection_keys = list(contrib_data.projection_keys or [])
    projected_values = dict(contrib_data.projected_values or {})
    n_join_rows = len(values)
    private_to_rows = _private_to_rows(contrib_data.private_ids_per_join_row)
    n_private_entities = len(private_to_rows)
    distinct_projection_keys = list(projected_values)
    n_projection_keys = len(distinct_projection_keys)

    if n_projection_keys == 0:
        return _empty_result(
            tau=tau,
            query_type="SPJA",
            lp_time_sec=time.perf_counter() - start,
            n_join_rows=n_join_rows,
            n_private_entities=n_private_entities,
        )
    if len(projection_keys) != n_join_rows:
        msg = "SPJA contribution data must have one projection key per join row"
        raise ValueError(msg)

    key_to_v_idx = {key: idx for idx, key in enumerate(distinct_projection_keys)}
    n_variables = n_join_rows + n_projection_keys
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_ub: list[float] = []
    constraint_idx = 0

    for row_idx, key in enumerate(projection_keys):
        if key not in key_to_v_idx:
            msg = f"projection key {key!r} missing from projected_values"
            raise ValueError(msg)
        rows.append(constraint_idx + key_to_v_idx[key])
        cols.append(row_idx)
        data.append(-1.0)

    for _key, v_idx in key_to_v_idx.items():
        rows.append(constraint_idx + v_idx)
        cols.append(n_join_rows + v_idx)
        data.append(1.0)
        b_ub.append(0.0)
    constraint_idx += n_projection_keys

    for row_indexes in private_to_rows.values():
        for row_idx in row_indexes:
            rows.append(constraint_idx)
            cols.append(row_idx)
            data.append(1.0)
        b_ub.append(float(tau))
        constraint_idx += 1

    n_constraints = n_projection_keys + n_private_entities
    a_ub = coo_matrix((data, (rows, cols)), shape=(n_constraints, n_variables)).tocsr()
    c = np.concatenate(
        [np.zeros(n_join_rows, dtype=float), -np.ones(n_projection_keys, dtype=float)]
    )
    bounds = [(0.0, float(value)) for value in values]
    bounds.extend(
        (0.0, float(projected_values[key])) for key in distinct_projection_keys
    )

    result = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    lp_time_sec = time.perf_counter() - start
    q_tau = float(-result.fun) if result.success else float("nan")

    return TruncationResult(
        q_tau=q_tau,
        tau=float(tau),
        query_type="SPJA",
        lp_status=str(result.status),
        lp_success=bool(result.success),
        lp_message=str(result.message),
        lp_time_sec=lp_time_sec,
        n_join_rows=n_join_rows,
        n_private_entities=n_private_entities,
        n_projection_keys=n_projection_keys,
        n_variables=n_variables,
        n_constraints=n_constraints,
    )


def truncate_query(contrib_data: ContributionData, tau: float) -> TruncationResult:
    """Dispatch to the truncation LP matching `contrib_data.query_type`."""

    if contrib_data.query_type == "SJA":
        return truncate_sja_lp(contrib_data, tau)
    if contrib_data.query_type == "SPJA":
        return truncate_spja_lp(contrib_data, tau)
    msg = f"unsupported query_type: {contrib_data.query_type!r}"
    raise ValueError(msg)


def _validate_tau(tau: float) -> None:
    if tau < 0:
        msg = "tau must be non-negative"
        raise ValueError(msg)


def _nonnegative_values(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    if np.any(values_arr < 0):
        msg = "LP truncation requires non-negative contribution values"
        raise ValueError(msg)
    return values_arr


def _private_to_rows(private_ids_per_join_row: list[list[str]]) -> dict[str, list[int]]:
    private_to_rows: dict[str, list[int]] = defaultdict(list)
    for row_idx, private_ids in enumerate(private_ids_per_join_row):
        for private_id in set(private_ids):
            private_to_rows[private_id].append(row_idx)
    return dict(private_to_rows)


def _empty_result(
    *,
    tau: float,
    query_type: str,
    lp_time_sec: float,
    n_join_rows: int = 0,
    n_private_entities: int = 0,
) -> TruncationResult:
    return TruncationResult(
        q_tau=0.0,
        tau=float(tau),
        query_type=query_type,
        lp_status="empty",
        lp_success=True,
        lp_message="No rows to truncate.",
        lp_time_sec=lp_time_sec,
        n_join_rows=n_join_rows,
        n_private_entities=n_private_entities,
        n_projection_keys=0,
        n_variables=0,
        n_constraints=0,
    )
