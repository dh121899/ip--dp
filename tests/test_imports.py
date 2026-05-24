"""Import tests for the package scaffold."""

from __future__ import annotations


def test_package_imports() -> None:
    import r2t_tpch
    from r2t_tpch import (
        contributions,
        load_tpch,
        lp_truncation,
        mechanisms_r2t,
        mechanisms_shifted_inverse,
        metrics,
        plots,
        queries,
        run_experiment,
        schema,
    )

    assert r2t_tpch.__version__
    assert contributions
    assert load_tpch
    assert lp_truncation
    assert mechanisms_r2t
    assert mechanisms_shifted_inverse
    assert metrics
    assert plots
    assert queries
    assert run_experiment
    assert schema
