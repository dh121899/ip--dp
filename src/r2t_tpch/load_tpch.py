"""Utilities for loading TPC-H `.tbl` data into DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from r2t_tpch.schema import TPCH_TABLES, TableSpec, table_names

REQUIRED_TBL_FILES = tuple(f"{table.name}.tbl" for table in TPCH_TABLES)
ROW_COUNT_COLUMNS = tuple(f"row_count_{table.name}" for table in TPCH_TABLES)


@dataclass(frozen=True)
class LoadedTPCH:
    """Loaded DuckDB connection and data-source metadata."""

    connection: duckdb.DuckDBPyConnection
    data_mode: str
    data_source: str
    data_dir: str
    row_counts: dict[str, int]


def resolve_data_dir(path: str | Path = "data/raw") -> Path:
    """Resolve the configured TPC-H data directory."""

    return Path(path).expanduser().resolve()


def list_available_tbl_files(path: str | Path = "data/raw") -> list[Path]:
    """List available `.tbl` files without requiring bundled benchmark data."""

    data_dir = resolve_data_dir(path)
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.tbl"))


def missing_required_tbl_files(path: str | Path) -> list[str]:
    """Return required TPC-H `.tbl` files missing from `path`."""

    data_dir = resolve_data_dir(path)
    return [
        file_name
        for file_name in REQUIRED_TBL_FILES
        if not (data_dir / file_name).is_file()
    ]


def has_required_tbl_files(path: str | Path) -> bool:
    """Return whether all required TPC-H `.tbl` files exist."""

    return not missing_required_tbl_files(path)


def load_tpch(
    data_dir: str | Path = "data/raw",
    *,
    cache_dir: str | Path | None = "data/cache",
    use_cache: bool = True,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load TPC-H tables into DuckDB, falling back to tiny synthetic data.

    Real TPC-H `.tbl` files are expected to use the standard trailing pipe
    delimiter. When no required files are found, deterministic tiny tables are
    created for tests and CI.
    """

    con = connection or duckdb.connect(database=":memory:")
    source_dir = resolve_data_dir(data_dir)
    found_files = {path.name for path in list_available_tbl_files(source_dir)}

    if not found_files:
        create_tiny_tpch(con)
        return con

    missing = sorted(set(REQUIRED_TBL_FILES) - found_files)
    if missing:
        msg = f"Missing required TPC-H .tbl files: {', '.join(missing)}"
        raise FileNotFoundError(msg)

    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if cache_path and use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)

    for table in TPCH_TABLES:
        parquet_path = cache_path / f"{table.name}.parquet" if cache_path else None
        if use_cache and parquet_path and parquet_path.exists():
            _load_parquet(con, table.name, parquet_path)
            continue

        tbl_path = source_dir / f"{table.name}.tbl"
        _load_tbl(con, table, tbl_path)
        if use_cache and parquet_path:
            con.execute(f"COPY {table.name} TO ? (FORMAT PARQUET)", [str(parquet_path)])

    return con


def load_tpch_real(
    data_dir: str | Path,
    *,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load real TPC-H `.tbl` files, never falling back to tiny data."""

    source_dir = resolve_data_dir(data_dir)
    missing = missing_required_tbl_files(source_dir)
    if missing:
        msg = f"Missing required TPC-H .tbl files in {source_dir}: {', '.join(missing)}"
        raise FileNotFoundError(msg)

    con = connection or duckdb.connect(database=":memory:")
    for table in TPCH_TABLES:
        _load_tbl(con, table, source_dir / f"{table.name}.tbl")

    row_counts = count_loaded_rows(con)
    zero_tables = [name for name, count in row_counts.items() if count == 0]
    if zero_tables:
        msg = f"Unexpected zero-row TPC-H tables: {', '.join(zero_tables)}"
        raise ValueError(msg)
    return con


def load_tpch_tiny(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Load deterministic tiny synthetic data."""

    return create_tiny_tpch(connection)


def load_tpch_with_mode(
    data_dir: str | Path | None = None,
    *,
    data_mode: str = "auto",
    require_real_data: bool = False,
) -> LoadedTPCH:
    """Load TPC-H data according to tiny/auto/real data mode."""

    mode = "real" if require_real_data else data_mode
    if mode not in {"auto", "real", "tiny"}:
        msg = "data_mode must be one of: auto, real, tiny"
        raise ValueError(msg)

    data_dir_label = str(data_dir or "")
    if mode == "tiny":
        con = load_tpch_tiny()
        return LoadedTPCH(
            con, mode, "tiny_synthetic", data_dir_label, count_loaded_rows(con)
        )

    if mode == "real":
        if data_dir is None:
            msg = "--data-dir is required when --data-mode real is used"
            raise FileNotFoundError(msg)
        con = load_tpch_real(data_dir)
        return LoadedTPCH(
            con,
            mode,
            "real_tpch",
            str(resolve_data_dir(data_dir)),
            count_loaded_rows(con),
        )

    if data_dir is not None and has_required_tbl_files(data_dir):
        con = load_tpch_real(data_dir)
        return LoadedTPCH(
            con,
            mode,
            "real_tpch",
            str(resolve_data_dir(data_dir)),
            count_loaded_rows(con),
        )

    print("WARNING: using tiny synthetic data, not real TPC-H data.")
    con = load_tpch_tiny()
    return LoadedTPCH(
        con, mode, "tiny_synthetic", data_dir_label, count_loaded_rows(con)
    )


def create_tiny_tpch(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Create tiny deterministic TPC-H-like tables for tests and CI."""

    con = connection or duckdb.connect(database=":memory:")
    frames = _tiny_frames()
    for table in TPCH_TABLES:
        frame = frames[table.name]
        con.register(f"_{table.name}_tiny", frame)
        con.execute(
            f"CREATE OR REPLACE TABLE {table.name} "
            f"AS SELECT * FROM _{table.name}_tiny"
        )
        con.unregister(f"_{table.name}_tiny")
    return con


def count_loaded_rows(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Count rows for every recognized TPC-H table."""

    return {
        table.name: int(con.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0])
        for table in TPCH_TABLES
    }


def row_count_metadata(row_counts: dict[str, int]) -> dict[str, int]:
    """Return row-count metadata columns for experiment CSVs."""

    return {f"row_count_{name}": int(row_counts.get(name, 0)) for name in table_names()}


def validate_tpch_row_counts(
    row_counts: dict[str, int],
    sf: float,
    strict: bool = True,
) -> None:
    """Validate row counts against expected TPC-H scale-factor counts."""

    if sf <= 0:
        msg = "scale factor must be positive"
        raise ValueError(msg)

    exact_expected = {"region": 5, "nation": 25}
    scaled_expected = {
        "supplier": 10_000 * sf,
        "customer": 150_000 * sf,
        "orders": 1_500_000 * sf,
        "part": 200_000 * sf,
        "partsupp": 800_000 * sf,
        "lineitem": 6_000_000 * sf,
    }
    failures: list[str] = []

    for name, expected in exact_expected.items():
        actual = int(row_counts.get(name, -1))
        if actual != expected:
            failures.append(f"{name}: expected {expected}, actual {actual}")

    tolerance = 0.05 if strict else 0.35
    for name, expected_float in scaled_expected.items():
        expected = max(1, int(round(expected_float)))
        actual = int(row_counts.get(name, -1))
        allowed = max(2, int(round(expected * tolerance)))
        if abs(actual - expected) > allowed:
            failures.append(
                f"{name}: expected about {expected}, actual {actual}, "
                f"tolerance +/- {allowed}"
            )

    if failures:
        msg = "TPC-H row-count validation failed: " + "; ".join(failures)
        raise ValueError(msg)


def _load_parquet(
    con: duckdb.DuckDBPyConnection, table_name: str, parquet_path: Path
) -> None:
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet(?)",
        [str(parquet_path)],
    )


def _load_tbl(con: duckdb.DuckDBPyConnection, table: TableSpec, tbl_path: Path) -> None:
    cols_with_trailing = [*table.columns, ("__trailing", "VARCHAR")]
    column_spec = ", ".join(
        f"'{name}': '{dtype}'" for name, dtype in cols_with_trailing
    )
    select_cols = ", ".join(table.column_names)
    path_sql = str(tbl_path).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table.name} AS
        SELECT {select_cols}
        FROM read_csv(
            '{path_sql}',
            delim='|',
            header=false,
            columns={{{column_spec}}},
            dateformat='%Y-%m-%d'
        )
        """
    )


def _tiny_frames() -> dict[str, pd.DataFrame]:
    date = pd.to_datetime
    return {
        "region": pd.DataFrame(
            [
                (0, "ASIA", "synthetic asia"),
                (1, "EUROPE", "synthetic europe"),
            ],
            columns=["r_regionkey", "r_name", "r_comment"],
        ),
        "nation": pd.DataFrame(
            [
                (1, "CHINA", 0, "synthetic china"),
                (2, "FRANCE", 1, "synthetic france"),
                (3, "GERMANY", 1, "synthetic germany"),
            ],
            columns=["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        ),
        "supplier": pd.DataFrame(
            [
                (10, "Supplier#10", "addr", 1, "10-000", 100.0, "china supplier"),
                (20, "Supplier#20", "addr", 2, "20-000", 200.0, "france supplier"),
                (30, "Supplier#30", "addr", 3, "30-000", 300.0, "germany supplier"),
            ],
            columns=[
                "s_suppkey",
                "s_name",
                "s_address",
                "s_nationkey",
                "s_phone",
                "s_acctbal",
                "s_comment",
            ],
        ),
        "customer": pd.DataFrame(
            [
                (1, "Customer#1", "addr", 1, "10-001", 1000.0, "BUILDING", "asia"),
                (2, "Customer#2", "addr", 3, "30-002", 2000.0, "BUILDING", "germany"),
                (3, "Customer#3", "addr", 2, "20-003", 3000.0, "AUTOMOBILE", "france"),
            ],
            columns=[
                "c_custkey",
                "c_name",
                "c_address",
                "c_nationkey",
                "c_phone",
                "c_acctbal",
                "c_mktsegment",
                "c_comment",
            ],
        ),
        "orders": pd.DataFrame(
            [
                (100, 1, "O", 100.0, date("1994-03-01"), "1-URGENT", "Clerk#1", 0, ""),
                (101, 2, "O", 200.0, date("1995-02-10"), "2-HIGH", "Clerk#2", 0, ""),
                (102, 3, "O", 300.0, date("1993-11-01"), "3-MEDIUM", "Clerk#3", 0, ""),
                (103, 3, "O", 400.0, date("1996-02-01"), "4-NOT", "Clerk#4", 0, ""),
            ],
            columns=[
                "o_orderkey",
                "o_custkey",
                "o_orderstatus",
                "o_totalprice",
                "o_orderdate",
                "o_orderpriority",
                "o_clerk",
                "o_shippriority",
                "o_comment",
            ],
        ),
        "lineitem": pd.DataFrame(
            [
                (
                    100,
                    1000,
                    10,
                    1,
                    5.0,
                    100.0,
                    0.10,
                    0.0,
                    "N",
                    "O",
                    date("1994-04-01"),
                    date("1994-04-03"),
                    date("1994-04-05"),
                    "DELIVER",
                    "MAIL",
                    "",
                ),
                (
                    101,
                    1001,
                    20,
                    1,
                    7.0,
                    200.0,
                    0.05,
                    0.0,
                    "N",
                    "O",
                    date("1995-04-01"),
                    date("1995-04-03"),
                    date("1995-04-05"),
                    "DELIVER",
                    "SHIP",
                    "",
                ),
                (
                    102,
                    1002,
                    30,
                    1,
                    9.0,
                    300.0,
                    0.10,
                    0.0,
                    "R",
                    "F",
                    date("1993-11-20"),
                    date("1993-11-22"),
                    date("1993-11-25"),
                    "RETURN",
                    "RAIL",
                    "",
                ),
                (
                    103,
                    1003,
                    30,
                    1,
                    11.0,
                    400.0,
                    0.00,
                    0.0,
                    "N",
                    "O",
                    date("1996-06-01"),
                    date("1996-06-03"),
                    date("1996-06-05"),
                    "DELIVER",
                    "AIR",
                    "",
                ),
            ],
            columns=[
                "l_orderkey",
                "l_partkey",
                "l_suppkey",
                "l_linenumber",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_returnflag",
                "l_linestatus",
                "l_shipdate",
                "l_commitdate",
                "l_receiptdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment",
            ],
        ),
        "part": pd.DataFrame(
            [
                (1000, "Part#1000", "MFGR#1", "Brand#1", "SMALL", 1, "BOX", 10.0, ""),
                (1001, "Part#1001", "MFGR#1", "Brand#2", "MEDIUM", 2, "BAG", 20.0, ""),
                (1002, "Part#1002", "MFGR#2", "Brand#3", "LARGE", 3, "CAN", 30.0, ""),
                (1003, "Part#1003", "MFGR#2", "Brand#4", "LARGE", 4, "BOX", 40.0, ""),
            ],
            columns=[
                "p_partkey",
                "p_name",
                "p_mfgr",
                "p_brand",
                "p_type",
                "p_size",
                "p_container",
                "p_retailprice",
                "p_comment",
            ],
        ),
        "partsupp": pd.DataFrame(
            [
                (1000, 10, 100, 1.0, ""),
                (1001, 20, 100, 2.0, ""),
                (1002, 30, 100, 3.0, ""),
                (1003, 30, 100, 4.0, ""),
            ],
            columns=[
                "ps_partkey",
                "ps_suppkey",
                "ps_availqty",
                "ps_supplycost",
                "ps_comment",
            ],
        ),
    }
