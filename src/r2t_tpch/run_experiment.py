"""Experiment runner for TPC-H-derived R2T comparisons."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from r2t_tpch.contributions import ContributionData
from r2t_tpch.load_tpch import (
    LoadedTPCH,
    load_tpch_with_mode,
    validate_tpch_row_counts,
)
from r2t_tpch.mechanisms_r2t import (
    MechanismResult,
    direct_laplace,
    fixed_tau,
    oracle_best_tau_non_dp,
    r2t,
)
from r2t_tpch.mechanisms_shifted_inverse import (
    ShiftedInverseResult,
    apply_shifted_inverse,
)
from r2t_tpch.metrics import (
    compute_contribution_stats,
    compute_run_metrics,
    metadata_with_row_counts,
    summarize_runs,
)
from r2t_tpch.queries import (
    q3_spja_order_count,
    q5_sja_revenue,
    q7_sja_revenue,
    q10_spja_customer_count,
    q12_sja_count,
)

DEFAULT_QUERIES = [
    "q5_sja_revenue",
    "q7_sja_revenue",
    "q12_sja_count",
    "q3_spja_order_count",
    "q10_spja_customer_count",
]
DEFAULT_MECHANISMS = ["DirectLaplace", "FixedTau", "R2T"]
QUERY_FUNCTIONS = {
    "q5_sja_revenue": q5_sja_revenue,
    "q7_sja_revenue": q7_sja_revenue,
    "q12_sja_count": q12_sja_count,
    "q3_spja_order_count": q3_spja_order_count,
    "q10_spja_customer_count": q10_spja_customer_count,
}
VALID_MECHANISMS = [
    "DirectLaplace",
    "FixedTau",
    "R2T",
    "OracleBestTauNonDP",
    "ShiftedInverseSum",
    "ShiftedInverseCount",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run R2T mechanisms on TPC-H-derived scalar queries."
    )
    parser.add_argument("--data-dir", help="Directory containing TPC-H .tbl files.")
    parser.add_argument(
        "--data-mode",
        choices=["auto", "real", "tiny"],
        default="auto",
        help="Data loading mode.",
    )
    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help="Equivalent to --data-mode real.",
    )
    parser.add_argument("--scale-factor", type=float, default=None)
    parser.add_argument("--expected-scale-factor", type=float, default=None)
    parser.add_argument("--validate-row-counts", action="store_true")
    parser.add_argument("--allow-approx-row-counts", action="store_true")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--queries", nargs="+", choices=DEFAULT_QUERIES, default=None)
    parser.add_argument("--query", choices=DEFAULT_QUERIES, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        choices=VALID_MECHANISMS,
        default=DEFAULT_MECHANISMS,
    )
    parser.add_argument("--epsilon", type=float, nargs="+", default=[1.0])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gsq", type=float, nargs="+", default=[1024.0])
    parser.add_argument(
        "--fixed-taus",
        type=float,
        nargs="+",
        default=[128.0, 512.0, 2048.0],
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment and write CSV outputs."""

    args = build_parser().parse_args(argv)
    selected_queries = _selected_queries(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_tpch_with_mode(
        args.data_dir,
        data_mode=args.data_mode,
        require_real_data=args.require_real_data,
    )
    _validate_loaded_data(loaded, args)
    _print_data_summary(loaded, args)

    metadata = metadata_with_row_counts(
        data_mode="real" if args.require_real_data else args.data_mode,
        data_source=loaded.data_source,
        data_dir=loaded.data_dir,
        scale_factor=args.scale_factor,
        expected_scale_factor=args.expected_scale_factor,
        row_counts=loaded.row_counts,
    )

    runs: list[dict[str, Any]] = []
    tau_trace: list[dict[str, Any]] = []
    shifted_trace: list[dict[str, Any]] = []
    contribution_stats: list[dict[str, Any]] = []

    for query_id in selected_queries:
        query_start = time.perf_counter()
        contrib_data = QUERY_FUNCTIONS[query_id](loaded.connection)
        query_time = time.perf_counter() - query_start
        contribution_stats.append(compute_contribution_stats(contrib_data, metadata))
        n_private = contribution_stats[-1]["n_private_entities"]
        print(
            f"query_id={query_id} query_type={contrib_data.query_type} "
            f"true_answer={contrib_data.true_answer:.6g} "
            f"n_join_rows={len(contrib_data.join_values)} "
            f"n_private_entities={n_private}"
        )

        for run_id in range(args.runs):
            run_seed = args.seed + run_id
            for epsilon in args.epsilon:
                for gsq in args.gsq:
                    for mechanism in args.mechanisms:
                        print(
                            f"mechanism={mechanism} query_id={query_id} "
                            f"run_id={run_id}"
                        )
                        _run_mechanism(
                            mechanism=mechanism,
                            contrib_data=contrib_data,
                            metadata=metadata,
                            run_id=run_id,
                            seed=run_seed,
                            epsilon=float(epsilon),
                            beta=float(args.beta),
                            gsq=float(gsq),
                            fixed_taus=args.fixed_taus,
                            query_time_sec=query_time,
                            runs=runs,
                            tau_trace=tau_trace,
                            shifted_trace=shifted_trace,
                        )

    runs_df = pd.DataFrame(runs)
    summary_df = summarize_runs(runs_df)
    pd.DataFrame(contribution_stats).to_csv(
        out_dir / "contribution_stats.csv", index=False
    )
    runs_df.to_csv(out_dir / "results_runs.csv", index=False)
    summary_df.to_csv(out_dir / "results_summary.csv", index=False)
    pd.DataFrame(tau_trace).to_csv(out_dir / "tau_trace.csv", index=False)
    if shifted_trace:
        pd.DataFrame(shifted_trace).to_csv(
            out_dir / "shifted_inverse_trace.csv", index=False
        )
    print(f"Wrote results to {out_dir}")
    return 0


def _selected_queries(args: argparse.Namespace) -> list[str]:
    if args.query:
        return [args.query]
    return args.queries or DEFAULT_QUERIES


def _validate_loaded_data(loaded: LoadedTPCH, args: argparse.Namespace) -> None:
    if loaded.data_source == "real_tpch" and any(
        count == 0 for count in loaded.row_counts.values()
    ):
        zero_tables = [name for name, count in loaded.row_counts.items() if count == 0]
        msg = f"Unexpected zero-row TPC-H tables: {', '.join(zero_tables)}"
        raise ValueError(msg)
    if args.validate_row_counts:
        if args.expected_scale_factor is None:
            msg = "--expected-scale-factor is required with --validate-row-counts"
            raise ValueError(msg)
        validate_tpch_row_counts(
            loaded.row_counts,
            args.expected_scale_factor,
            strict=not args.allow_approx_row_counts,
        )


def _print_data_summary(loaded: LoadedTPCH, args: argparse.Namespace) -> None:
    print(
        f"data_mode={args.data_mode} data_source={loaded.data_source} "
        f"scale_factor={args.scale_factor}"
    )
    print(f"data_dir={loaded.data_dir}")
    for name, count in loaded.row_counts.items():
        print(f"row_count_{name}={count}")


def _run_mechanism(
    *,
    mechanism: str,
    contrib_data: ContributionData,
    metadata: dict[str, Any],
    run_id: int,
    seed: int,
    epsilon: float,
    beta: float,
    gsq: float,
    fixed_taus: list[float],
    query_time_sec: float,
    runs: list[dict[str, Any]],
    tau_trace: list[dict[str, Any]],
    shifted_trace: list[dict[str, Any]],
) -> None:
    if mechanism == "FixedTau":
        for tau in fixed_taus:
            result = _safe_mechanism_result(
                mechanism,
                lambda rng, tau=tau: fixed_tau(contrib_data, tau, epsilon, rng),
                seed,
                contrib_data.true_answer,
                epsilon,
                beta,
                gsq,
            )
            runs.append(
                compute_run_metrics(
                    contrib_data=contrib_data,
                    mechanism_result=result,
                    metadata=metadata,
                    run_id=run_id,
                    seed=seed,
                    query_time_sec=query_time_sec,
                )
            )
        return

    rng = np.random.default_rng(seed)
    if mechanism == "DirectLaplace":
        result = _safe_call(
            mechanism,
            lambda: direct_laplace(contrib_data.true_answer, gsq, epsilon, rng),
            contrib_data.true_answer,
            epsilon,
            beta,
            gsq,
        )
    elif mechanism == "R2T":
        result, trace_rows = _safe_r2t(contrib_data, gsq, epsilon, beta, rng, run_id)
        tau_trace.extend(
            {**asdict(row), **_trace_metadata(metadata), "epsilon": epsilon, "gsq": gsq}
            for row in trace_rows
        )
    elif mechanism == "OracleBestTauNonDP":
        result = _safe_call(
            mechanism,
            lambda: oracle_best_tau_non_dp(
                contrib_data,
                tau_grid=[float(tau) for tau in fixed_taus],
                epsilon=epsilon,
                rng=rng,
            ),
            contrib_data.true_answer,
            epsilon,
            beta,
            gsq,
        )
    elif mechanism in {"ShiftedInverseSum", "ShiftedInverseCount"}:
        result, trace_rows = _run_shifted_inverse(
            requested=mechanism,
            contrib_data=contrib_data,
            epsilon=epsilon,
            beta=beta,
            gsq=gsq,
            rng=rng,
        )
        shifted_trace.extend(
            {**asdict(row), **_trace_metadata(metadata), "epsilon": epsilon}
            for row in trace_rows
        )
    else:
        result = _failure_result(
            mechanism,
            "unsupported mechanism",
            contrib_data.true_answer,
            epsilon,
            beta,
            gsq,
        )

    runs.append(
        compute_run_metrics(
            contrib_data=contrib_data,
            mechanism_result=result,
            metadata=metadata,
            run_id=run_id,
            seed=seed,
            query_time_sec=query_time_sec,
        )
    )


def _safe_mechanism_result(
    mechanism: str,
    fn: Any,
    seed: int,
    true_answer: float,
    epsilon: float,
    beta: float,
    gsq: float,
) -> MechanismResult:
    rng = np.random.default_rng(seed)
    return _safe_call(mechanism, lambda: fn(rng), true_answer, epsilon, beta, gsq)


def _safe_call(
    mechanism: str,
    fn: Any,
    true_answer: float,
    epsilon: float,
    beta: float,
    gsq: float,
) -> MechanismResult:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - experiment rows should record failures.
        return _failure_result(mechanism, str(exc), true_answer, epsilon, beta, gsq)


def _safe_r2t(
    contrib_data: ContributionData,
    gsq: float,
    epsilon: float,
    beta: float,
    rng: np.random.Generator,
    run_id: int,
) -> tuple[MechanismResult, list[Any]]:
    try:
        return r2t(contrib_data, gsq, epsilon, beta, rng, run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - experiment rows should record failures.
        return (
            _failure_result(
                "R2T", str(exc), contrib_data.true_answer, epsilon, beta, gsq
            ),
            [],
        )


def _run_shifted_inverse(
    *,
    requested: str,
    contrib_data: ContributionData,
    epsilon: float,
    beta: float,
    gsq: float,
    rng: np.random.Generator,
) -> tuple[ShiftedInverseResult, list[Any]]:
    if not _is_shifted_inverse_applicable(requested, contrib_data.query_id):
        return _shifted_not_applicable(requested, contrib_data, epsilon, beta, gsq), []
    try:
        result, trace = apply_shifted_inverse(
            contrib_data,
            epsilon=epsilon,
            beta=beta,
            D=max(float(gsq), float(contrib_data.true_answer)),
            rng=rng,
        )
        return replace(result, mechanism=requested), trace
    except Exception as exc:  # noqa: BLE001 - experiment rows should record failures.
        return (
            _shifted_failure(requested, contrib_data, epsilon, beta, gsq, str(exc)),
            [],
        )


def _is_shifted_inverse_applicable(mechanism: str, query_id: str) -> bool:
    return (mechanism == "ShiftedInverseSum" and query_id == "q5_sja_revenue") or (
        mechanism == "ShiftedInverseCount" and query_id == "q12_sja_count"
    )


def _failure_result(
    mechanism: str,
    message: str,
    true_answer: float,
    epsilon: float,
    beta: float,
    gsq: float,
) -> MechanismResult:
    return MechanismResult(
        mechanism=mechanism,
        answer=float("nan"),
        true_answer=float(true_answer),
        epsilon=float(epsilon),
        beta=float(beta),
        gsq=float(gsq),
        tau=None,
        winner_tau=None,
        noise_scale=None,
        status=f"failed:{message}",
        runtime_sec=0.0,
        lp_time_sec=0.0,
        extra={"error": message},
    )


def _shifted_not_applicable(
    mechanism: str,
    contrib_data: ContributionData,
    epsilon: float,
    beta: float,
    gsq: float,
) -> ShiftedInverseResult:
    return ShiftedInverseResult(
        mechanism=mechanism,
        answer=float("nan"),
        true_answer=float(contrib_data.true_answer),
        epsilon=float(epsilon),
        beta=float(beta),
        D=float(max(gsq, contrib_data.true_answer)),
        tau0=0,
        status="not_applicable",
        runtime_sec=0.0,
        extra={
            "query_id": contrib_data.query_id,
            "reason": "unsupported_optional_shifted_inverse_query",
        },
    )


def _shifted_failure(
    mechanism: str,
    contrib_data: ContributionData,
    epsilon: float,
    beta: float,
    gsq: float,
    message: str,
) -> ShiftedInverseResult:
    return ShiftedInverseResult(
        mechanism=mechanism,
        answer=float("nan"),
        true_answer=float(contrib_data.true_answer),
        epsilon=float(epsilon),
        beta=float(beta),
        D=float(max(gsq, contrib_data.true_answer)),
        tau0=0,
        status=f"failed:{message}",
        runtime_sec=0.0,
        extra={"error": message},
    )


def _trace_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_mode": metadata.get("data_mode"),
        "data_source": metadata.get("data_source"),
        "scale_factor": metadata.get("scale_factor"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
