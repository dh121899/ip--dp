"""Tests for Laplace baseline helpers."""

from __future__ import annotations

import numpy as np

from r2t_tpch.mechanisms_r2t import direct_laplace, laplace_noise


def test_laplace_noise_returns_zero_for_zero_scale() -> None:
    rng = np.random.default_rng(123)

    assert laplace_noise(0.0, rng) == 0.0


def test_direct_laplace_uses_expected_scale_metadata() -> None:
    rng = np.random.default_rng(123)
    result = direct_laplace(true_answer=10.0, gsq=4.0, epsilon=2.0, rng=rng)

    assert result.mechanism == "DirectLaplace"
    assert result.noise_scale == 2.0
    assert result.true_answer == 10.0
    assert np.isfinite(result.answer)

