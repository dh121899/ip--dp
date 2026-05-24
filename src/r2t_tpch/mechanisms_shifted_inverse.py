"""Optional lightweight ShiftedInverse-style mechanisms.

This module is an educational approximation for simple monotonic sum/count
functions on per-user L1 contribution arrays. It is not a full implementation
of the Shifted Inverse paper.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from r2t_tpch.contributions import ContributionData


@dataclass(frozen=True)
class ShiftedInverseConfig:
    """Configuration placeholder for optional ShiftedInverse experiments."""

    epsilon: float
    shift: float = 0.0


@dataclass(frozen=True)
class ShiftedInverseResult:
    """Result for the optional lightweight ShiftedInverse approximation."""

    mechanism: str
    answer: float
    true_answer: float
    epsilon: float
    beta: float
    D: float
    tau0: int
    status: str
    runtime_sec: float
    extra: dict[str, Any]


@dataclass(frozen=True)
class ShiftedInverseTraceRow:
    """Trace row for the sampled shifted-inverse index/interval."""

    mechanism: str
    query_id: str
    j: int
    f_down: float
    score: float
    interval_left: float | None
    interval_right: float | None
    weight: float | None


def validate_shifted_inverse_config(config: ShiftedInverseConfig) -> None:
    """Validate basic optional mechanism parameters."""

    if config.epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)


def per_user_contributions(contrib_data: ContributionData) -> dict[str, float]:
    """Aggregate SJA join-row values by referenced private user.

    Rows with multiple private ids add the row value to each referenced id. SPJA
    data is intentionally not expanded here because this optional module does
    not implement projection-aware shifted inverse.
    """

    if contrib_data.query_type == "SPJA":
        return {}
    if contrib_data.query_type != "SJA":
        msg = f"unsupported query_type: {contrib_data.query_type!r}"
        raise ValueError(msg)

    result: dict[str, float] = {}
    for value, private_ids in zip(
        contrib_data.join_values, contrib_data.private_ids_per_join_row, strict=True
    ):
        for private_id in private_ids:
            result[private_id] = result.get(private_id, 0.0) + float(value)
    return result


def shifted_inverse_sum_l1(
    contributions: Sequence[float] | np.ndarray,
    epsilon: float,
    beta: float,
    D: float,
    rng: np.random.Generator,
    query_id: str = "unknown",
) -> tuple[ShiftedInverseResult, list[ShiftedInverseTraceRow]]:
    """Approximate ShiftedInverse for a monotonic sum under L1 contribution data.

    This simplified educational routine computes down-neighbor sums after
    removing the `j` largest user contributions, scores indices by distance
    from `tau0`, samples an index with exponential weights, and returns a value
    from the selected down-sum interval.
    """

    start = time.perf_counter()
    values = _validate_contribution_array(contributions)
    true_answer = float(values.sum())
    _validate_inputs(epsilon, beta, D, true_answer)

    tau0 = int(math.ceil((2.0 / epsilon) * math.log((D + 1.0) / beta)))
    sorted_values = np.sort(values)[::-1]
    max_j = 2 * tau0
    f_down_values = _down_sums(true_answer, sorted_values, max_j)
    weights = _stable_weights(max_j=max_j, tau0=tau0, epsilon=epsilon)

    chosen_index = int(rng.choice(np.arange(max_j + 1), p=weights))
    interval_left, interval_right = _interval_bounds(f_down_values, chosen_index)
    if interval_left == interval_right:
        answer = interval_left
    else:
        answer = float(rng.uniform(interval_left, interval_right))

    trace = [
        ShiftedInverseTraceRow(
            mechanism="ShiftedInverseSum",
            query_id=query_id,
            j=j,
            f_down=float(f_down_values[j]),
            score=float(-abs(j - tau0)),
            interval_left=(
                _interval_bounds(f_down_values, j)[0] if j == chosen_index else None
            ),
            interval_right=(
                _interval_bounds(f_down_values, j)[1] if j == chosen_index else None
            ),
            weight=float(weights[j]),
        )
        for j in range(max_j + 1)
    ]
    result = ShiftedInverseResult(
        mechanism="ShiftedInverseSum",
        answer=float(answer),
        true_answer=true_answer,
        epsilon=float(epsilon),
        beta=float(beta),
        D=float(D),
        tau0=tau0,
        status="success",
        runtime_sec=time.perf_counter() - start,
        extra={
            "query_id": query_id,
            "n_users": int(len(values)),
            "chosen_j": chosen_index,
            "approximation": "educational_l1_sum_only",
        },
    )
    return result, trace


def shifted_inverse_count_l1(
    counts: Sequence[float] | np.ndarray,
    epsilon: float,
    beta: float,
    D: float,
    rng: np.random.Generator,
    query_id: str = "unknown",
) -> tuple[ShiftedInverseResult, list[ShiftedInverseTraceRow]]:
    """Approximate ShiftedInverse for a monotonic count from per-user counts."""

    result, trace = shifted_inverse_sum_l1(
        counts,
        epsilon=epsilon,
        beta=beta,
        D=D,
        rng=rng,
        query_id=query_id,
    )
    renamed_result = ShiftedInverseResult(
        mechanism="ShiftedInverseCount",
        answer=result.answer,
        true_answer=result.true_answer,
        epsilon=result.epsilon,
        beta=result.beta,
        D=result.D,
        tau0=result.tau0,
        status=result.status,
        runtime_sec=result.runtime_sec,
        extra={**result.extra, "approximation": "educational_l1_count_only"},
    )
    renamed_trace = [
        ShiftedInverseTraceRow(
            mechanism="ShiftedInverseCount",
            query_id=row.query_id,
            j=row.j,
            f_down=row.f_down,
            score=row.score,
            interval_left=row.interval_left,
            interval_right=row.interval_right,
            weight=row.weight,
        )
        for row in trace
    ]
    return renamed_result, renamed_trace


def apply_shifted_inverse(
    contrib_data: ContributionData,
    epsilon: float,
    beta: float,
    D: float,
    rng: np.random.Generator,
) -> tuple[ShiftedInverseResult, list[ShiftedInverseTraceRow]]:
    """Apply the optional lightweight ShiftedInverse routine when applicable."""

    start = time.perf_counter()
    if contrib_data.query_type == "SPJA":
        return (
            ShiftedInverseResult(
                mechanism="ShiftedInverseOptional",
                answer=float("nan"),
                true_answer=float(contrib_data.true_answer),
                epsilon=float(epsilon),
                beta=float(beta),
                D=float(D),
                tau0=0,
                status="not_applicable",
                runtime_sec=time.perf_counter() - start,
                extra={
                    "query_id": contrib_data.query_id,
                    "reason": (
                        "SPJA projection-aware shifted inverse is not implemented."
                    ),
                },
            ),
            [],
        )

    user_values = list(per_user_contributions(contrib_data).values())
    if "count" in contrib_data.query_id or "q12" in contrib_data.query_id:
        return shifted_inverse_count_l1(
            user_values,
            epsilon=epsilon,
            beta=beta,
            D=D,
            rng=rng,
            query_id=contrib_data.query_id,
        )
    return shifted_inverse_sum_l1(
        user_values,
        epsilon=epsilon,
        beta=beta,
        D=D,
        rng=rng,
        query_id=contrib_data.query_id,
    )


def _validate_contribution_array(
    contributions: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1:
        msg = "contributions must be a one-dimensional array"
        raise ValueError(msg)
    if np.any(values < 0):
        msg = "contributions must be nonnegative"
        raise ValueError(msg)
    return values


def _validate_inputs(
    epsilon: float, beta: float, D: float, true_answer: float
) -> None:
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if beta <= 0 or beta >= 1:
        msg = "beta must be between 0 and 1"
        raise ValueError(msg)
    if D < true_answer:
        msg = "D must be at least the true answer"
        raise ValueError(msg)


def _down_sums(
    true_answer: float, sorted_values: np.ndarray, max_j: int
) -> list[float]:
    cumulative = np.concatenate([[0.0], np.cumsum(sorted_values)])
    n_users = len(sorted_values)
    f_down_values = []
    for j in range(max_j + 1):
        if j > n_users:
            f_down_values.append(0.0)
        else:
            f_down_values.append(float(max(0.0, true_answer - cumulative[j])))
    return f_down_values


def _stable_weights(max_j: int, tau0: int, epsilon: float) -> np.ndarray:
    scores = np.asarray([-abs(j - tau0) for j in range(max_j + 1)], dtype=float)
    logits = epsilon * scores / 2.0
    logits -= float(logits.max())
    weights = np.exp(logits)
    return weights / weights.sum()


def _interval_bounds(values: list[float], index: int) -> tuple[float, float]:
    center = float(values[index])
    left_neighbor = float(values[index - 1]) if index > 0 else center
    right_neighbor = float(values[index + 1]) if index + 1 < len(values) else center
    left = min(center, right_neighbor)
    right = max(center, left_neighbor)
    return left, right
