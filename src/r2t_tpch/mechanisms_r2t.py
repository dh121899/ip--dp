"""R2T and baseline mechanisms."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from r2t_tpch.contributions import ContributionData
from r2t_tpch.lp_truncation import TruncationResult, truncate_query


@dataclass(frozen=True)
class R2TConfig:
    """Configuration placeholder retained for compatibility."""

    epsilon: float
    delta: float = 0.0


@dataclass(frozen=True)
class MechanismResult:
    """Output and metadata for a private or diagnostic mechanism run."""

    mechanism: str
    answer: float
    true_answer: float
    epsilon: float
    beta: float | None
    gsq: float | None
    tau: float | None
    winner_tau: float | None
    noise_scale: float | None
    status: str
    runtime_sec: float
    lp_time_sec: float
    extra: dict[str, Any]


@dataclass(frozen=True)
class TauTraceRow:
    """Per-tau diagnostic row for the R2T race."""

    mechanism: str
    query_id: str
    run_id: int
    tau: float
    q_tau: float
    noise_scale: float
    laplace_noise: float
    penalty: float
    candidate_value: float
    is_winner: bool
    lp_status: str
    lp_time_sec: float


def validate_r2t_config(config: R2TConfig) -> None:
    """Validate basic privacy parameters."""

    if config.epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if config.delta < 0:
        msg = "delta must be non-negative"
        raise ValueError(msg)


def laplace_noise(scale: float, rng: np.random.Generator) -> float:
    """Draw Laplace noise using a NumPy random generator."""

    if scale < 0:
        msg = "scale must be non-negative"
        raise ValueError(msg)
    if scale == 0:
        return 0.0
    return float(rng.laplace(loc=0.0, scale=scale))


def direct_laplace(
    true_answer: float,
    gsq: float,
    epsilon: float,
    rng: np.random.Generator,
) -> MechanismResult:
    """Apply the direct global-sensitivity Laplace baseline."""

    _validate_epsilon_gsq(epsilon, gsq)
    start = time.perf_counter()
    scale = float(gsq) / float(epsilon)
    answer = float(true_answer) + laplace_noise(scale, rng)
    return MechanismResult(
        mechanism="DirectLaplace",
        answer=answer,
        true_answer=float(true_answer),
        epsilon=float(epsilon),
        beta=None,
        gsq=float(gsq),
        tau=None,
        winner_tau=None,
        noise_scale=scale,
        status="ok",
        runtime_sec=time.perf_counter() - start,
        lp_time_sec=0.0,
        extra={},
    )


def fixed_tau(
    contrib_data: ContributionData,
    tau: float,
    epsilon: float,
    rng: np.random.Generator,
) -> MechanismResult:
    """Apply a fixed truncation cap followed by Laplace noise."""

    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    start = time.perf_counter()
    truncation = truncate_query(contrib_data, tau)
    scale = float(tau) / float(epsilon)
    status = "ok" if truncation.lp_success else f"lp_failed:{truncation.lp_status}"
    answer = (
        truncation.q_tau + laplace_noise(scale, rng)
        if truncation.lp_success
        else float("nan")
    )
    return MechanismResult(
        mechanism="FixedTau",
        answer=float(answer),
        true_answer=float(contrib_data.true_answer),
        epsilon=float(epsilon),
        beta=None,
        gsq=None,
        tau=float(tau),
        winner_tau=float(tau),
        noise_scale=scale,
        status=status,
        runtime_sec=time.perf_counter() - start,
        lp_time_sec=truncation.lp_time_sec,
        extra=_truncation_extra(truncation),
    )


def r2t(
    contrib_data: ContributionData,
    gsq: float,
    epsilon: float,
    beta: float,
    rng: np.random.Generator,
    run_id: int = 0,
) -> tuple[MechanismResult, list[TauTraceRow]]:
    """Run the R2T race over powers-of-two truncation thresholds."""

    _validate_r2t_inputs(gsq, epsilon, beta)
    start = time.perf_counter()
    tau_grid = _powers_of_two_grid(gsq)
    log_gsq = math.log2(float(gsq))
    q_zero = truncate_query(contrib_data, 0.0)
    trace: list[TauTraceRow] = []
    lp_time_sec = q_zero.lp_time_sec
    best_value = q_zero.q_tau if q_zero.lp_success else float("-inf")
    winner_tau = 0.0
    winner_idx: int | None = None
    status = "ok" if q_zero.lp_success else f"lp_failed:{q_zero.lp_status}"

    if log_gsq == 0:
        penalty_multiplier = 0.0
    else:
        penalty_multiplier = log_gsq * math.log(log_gsq / beta)

    for tau in tau_grid:
        truncation = truncate_query(contrib_data, tau)
        lp_time_sec += truncation.lp_time_sec
        noise_scale = log_gsq * tau / epsilon
        noise = laplace_noise(noise_scale, rng)
        penalty = penalty_multiplier * tau / epsilon
        candidate = (
            truncation.q_tau + noise - penalty
            if truncation.lp_success
            else float("nan")
        )
        if not truncation.lp_success:
            status = f"lp_failed:{truncation.lp_status}"
        elif status == "ok" and candidate > best_value:
            best_value = candidate
            winner_tau = tau
            winner_idx = len(trace)
        trace.append(
            TauTraceRow(
                mechanism="R2T",
                query_id=contrib_data.query_id,
                run_id=run_id,
                tau=float(tau),
                q_tau=float(truncation.q_tau),
                noise_scale=float(noise_scale),
                laplace_noise=float(noise),
                penalty=float(penalty),
                candidate_value=float(candidate),
                is_winner=False,
                lp_status=truncation.lp_status,
                lp_time_sec=truncation.lp_time_sec,
            )
        )

    if winner_idx is not None:
        trace[winner_idx] = _mark_winner(trace[winner_idx])
    answer = best_value if status == "ok" else float("nan")
    result = MechanismResult(
        mechanism="R2T",
        answer=float(answer),
        true_answer=float(contrib_data.true_answer),
        epsilon=float(epsilon),
        beta=float(beta),
        gsq=float(gsq),
        tau=None,
        winner_tau=float(winner_tau),
        noise_scale=None,
        status=status,
        runtime_sec=time.perf_counter() - start,
        lp_time_sec=lp_time_sec,
        extra={
            "tau_grid": tau_grid,
            "q_zero": q_zero.q_tau,
            "q_zero_status": q_zero.lp_status,
            "n_join_rows": q_zero.n_join_rows,
            "n_private_entities": q_zero.n_private_entities,
            "n_projection_keys": q_zero.n_projection_keys,
            "n_variables": q_zero.n_variables,
            "n_constraints": q_zero.n_constraints,
        },
    )
    return result, trace


def oracle_best_tau_non_dp(
    contrib_data: ContributionData,
    tau_grid: list[float],
    epsilon: float,
    rng: np.random.Generator,
) -> MechanismResult:
    """Diagnostic non-DP oracle that chooses tau using the true answer."""

    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if not tau_grid:
        msg = "tau_grid must be non-empty"
        raise ValueError(msg)

    start = time.perf_counter()
    best_tau: float | None = None
    best_truncation: TruncationResult | None = None
    best_error = float("inf")
    lp_time_sec = 0.0
    status = "ok"
    for tau in tau_grid:
        truncation = truncate_query(contrib_data, tau)
        lp_time_sec += truncation.lp_time_sec
        if not truncation.lp_success:
            status = f"lp_failed:{truncation.lp_status}"
            continue
        error = abs(truncation.q_tau - contrib_data.true_answer)
        if error < best_error:
            best_error = error
            best_tau = float(tau)
            best_truncation = truncation

    if best_tau is None or best_truncation is None:
        answer = float("nan")
        scale = None
    else:
        scale = best_tau / epsilon
        answer = best_truncation.q_tau + laplace_noise(scale, rng)

    return MechanismResult(
        mechanism="OracleBestTauNonDP",
        answer=float(answer),
        true_answer=float(contrib_data.true_answer),
        epsilon=float(epsilon),
        beta=None,
        gsq=None,
        tau=best_tau,
        winner_tau=best_tau,
        noise_scale=scale,
        status=status,
        runtime_sec=time.perf_counter() - start,
        lp_time_sec=lp_time_sec,
        extra={"warning": "Diagnostic only; this mechanism is not DP."},
    )


def _validate_epsilon_gsq(epsilon: float, gsq: float) -> None:
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if gsq < 0:
        msg = "gsq must be non-negative"
        raise ValueError(msg)


def _validate_r2t_inputs(gsq: float, epsilon: float, beta: float) -> None:
    if gsq < 1:
        msg = "gsq must be at least 1"
        raise ValueError(msg)
    if epsilon <= 0:
        msg = "epsilon must be positive"
        raise ValueError(msg)
    if beta <= 0 or beta >= 1:
        msg = "beta must be between 0 and 1"
        raise ValueError(msg)


def _powers_of_two_grid(gsq: float) -> list[float]:
    max_power = int(math.floor(math.log2(float(gsq))))
    return [float(2**power) for power in range(max_power + 1)]


def _truncation_extra(truncation: TruncationResult) -> dict[str, Any]:
    return {
        "q_tau": truncation.q_tau,
        "lp_status": truncation.lp_status,
        "lp_success": truncation.lp_success,
        "lp_message": truncation.lp_message,
        "n_join_rows": truncation.n_join_rows,
        "n_private_entities": truncation.n_private_entities,
        "n_projection_keys": truncation.n_projection_keys,
        "n_variables": truncation.n_variables,
        "n_constraints": truncation.n_constraints,
    }


def _mark_winner(row: TauTraceRow) -> TauTraceRow:
    return TauTraceRow(
        mechanism=row.mechanism,
        query_id=row.query_id,
        run_id=row.run_id,
        tau=row.tau,
        q_tau=row.q_tau,
        noise_scale=row.noise_scale,
        laplace_noise=row.laplace_noise,
        penalty=row.penalty,
        candidate_value=row.candidate_value,
        is_winner=True,
        lp_status=row.lp_status,
        lp_time_sec=row.lp_time_sec,
    )
