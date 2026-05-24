"""Tests for R2T and baseline mechanisms."""

from __future__ import annotations

import numpy as np
import pytest

from r2t_tpch.contributions import ContributionData
from r2t_tpch.mechanisms_r2t import direct_laplace, fixed_tau, r2t


def test_r2t_returns_numeric_answer_and_trace() -> None:
    rng = np.random.default_rng(123)
    data = _simple_sja()

    result, trace = r2t(data, gsq=8.0, epsilon=1.0, beta=0.1, rng=rng, run_id=7)

    assert result.mechanism == "R2T"
    assert result.status == "ok"
    assert np.isfinite(result.answer)
    assert trace
    assert result.winner_tau in {0.0, 1.0, 2.0, 4.0, 8.0}
    assert all(row.run_id == 7 for row in trace)


def test_fixed_tau_returns_numeric_answer() -> None:
    rng = np.random.default_rng(123)

    result = fixed_tau(_simple_sja(), tau=2.0, epsilon=1.0, rng=rng)

    assert result.mechanism == "FixedTau"
    assert result.status == "ok"
    assert np.isfinite(result.answer)
    assert result.tau == 2.0


def test_direct_laplace_returns_numeric_answer() -> None:
    rng = np.random.default_rng(123)

    result = direct_laplace(true_answer=5.0, gsq=2.0, epsilon=1.0, rng=rng)

    assert result.mechanism == "DirectLaplace"
    assert np.isfinite(result.answer)


def test_invalid_r2t_inputs_raise_clear_errors() -> None:
    rng = np.random.default_rng(123)
    data = _simple_sja()

    with pytest.raises(ValueError, match="epsilon"):
        r2t(data, gsq=8.0, epsilon=0.0, beta=0.1, rng=rng)
    with pytest.raises(ValueError, match="beta"):
        r2t(data, gsq=8.0, epsilon=1.0, beta=1.0, rng=rng)
    with pytest.raises(ValueError, match="gsq"):
        r2t(data, gsq=0.5, epsilon=1.0, beta=0.1, rng=rng)
    with pytest.raises(ValueError, match="epsilon"):
        fixed_tau(data, tau=1.0, epsilon=0.0, rng=rng)
    with pytest.raises(ValueError, match="gsq"):
        direct_laplace(true_answer=1.0, gsq=-1.0, epsilon=1.0, rng=rng)


def _simple_sja() -> ContributionData:
    join_values = np.asarray([2.0, 1.0], dtype=np.float64)
    return ContributionData(
        query_id="simple_sja",
        query_type="SJA",
        join_values=join_values,
        private_ids_per_join_row=[["A"], ["B"]],
        projection_keys=None,
        projected_values=None,
        true_answer=float(join_values.sum()),
        extraction_time_sec=0.0,
    )
