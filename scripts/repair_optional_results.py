"""Repair optional-mechanism non-result rows in existing experiment CSVs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from r2t_tpch.metrics import is_success_status, summarize_runs

ERROR_FIELDS = [
    "private_answer",
    "signed_error",
    "abs_error",
    "rel_error",
    "squared_error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair optional ShiftedInverse non-result rows."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing results_runs.csv and results_summary.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    runs_path = results_dir / "results_runs.csv"
    summary_path = results_dir / "results_summary.csv"
    if not runs_path.exists():
        msg = f"Missing results_runs.csv: {runs_path}"
        raise FileNotFoundError(msg)

    _backup(runs_path, results_dir / "results_runs.before_task5_2_fix.csv")
    if summary_path.exists():
        _backup(summary_path, results_dir / "results_summary.before_task5_2_fix.csv")

    runs = pd.read_csv(runs_path)
    runs = repair_runs(runs)
    runs.to_csv(runs_path, index=False)
    summarize_runs(runs).to_csv(summary_path, index=False)
    print(f"Repaired {runs_path}")
    print(f"Recomputed {summary_path}")
    return 0


def repair_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Repair optional non-result rows in a results_runs DataFrame."""

    repaired = runs.copy()
    if "reason" not in repaired:
        repaired["reason"] = pd.Series([pd.NA] * len(repaired), dtype="string")
    else:
        repaired["reason"] = repaired["reason"].astype("string")

    q7_shifted_sum = (
        (repaired["query_id"] == "q7_sja_revenue")
        & (repaired["mechanism"] == "ShiftedInverseSum")
    )
    repaired.loc[q7_shifted_sum, "status"] = "not_applicable"
    repaired.loc[q7_shifted_sum, "reason"] = (
        "unsupported_optional_shifted_inverse_query"
    )

    valid_mask = repaired["status"].map(is_success_status)
    repaired["is_valid_result"] = valid_mask
    repaired.loc[~valid_mask, ERROR_FIELDS] = np.nan
    return repaired


def _backup(source: Path, destination: Path) -> None:
    if destination.exists():
        print(f"Backup already exists, keeping {destination}")
        return
    shutil.copy2(source, destination)
    print(f"Created backup {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
