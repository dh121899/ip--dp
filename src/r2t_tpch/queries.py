"""TPC-H-derived scalar query extraction functions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import duckdb
import numpy as np

from r2t_tpch.contributions import ContributionData


class QueryType(StrEnum):
    """Supported query families for the R2T experiments."""

    SJA = "SJA"
    SPJA = "SPJA"


@dataclass(frozen=True)
class QuerySpec:
    """Metadata for a TPC-H-derived scalar query."""

    name: str
    query_type: QueryType
    description: str


QUERY_REGISTRY: tuple[QuerySpec, ...] = (
    QuerySpec(
        name="q5_sja_revenue",
        query_type=QueryType.SJA,
        description="Scalar TPC-H Q5 revenue query.",
    ),
    QuerySpec(
        name="q7_sja_revenue",
        query_type=QueryType.SJA,
        description="Scalar TPC-H Q7 revenue query.",
    ),
    QuerySpec(
        name="q12_sja_count",
        query_type=QueryType.SJA,
        description="Scalar TPC-H Q12 lineitem count query.",
    ),
    QuerySpec(
        name="q3_spja_order_count",
        query_type=QueryType.SPJA,
        description="TPC-H Q3-style distinct order count query.",
    ),
    QuerySpec(
        name="q10_spja_customer_count",
        query_type=QueryType.SPJA,
        description="TPC-H Q10-style distinct returned-customer count query.",
    ),
)


def list_queries() -> list[QuerySpec]:
    """Return query metadata registered by the extraction layer."""

    return list(QUERY_REGISTRY)


def q5_sja_revenue(
    con: duckdb.DuckDBPyConnection,
    *,
    region: str = "ASIA",
    date: str = "1994-01-01",
    multi_private: bool = False,
) -> ContributionData:
    """Extract scalar TPC-H Q5 revenue contributions."""

    start = time.perf_counter()
    rows = con.execute(
        """
        SELECT
            CAST(l.l_extendedprice * (1 - l.l_discount) AS DOUBLE) AS join_value,
            c.c_custkey,
            s.s_suppkey
        FROM customer AS c
        JOIN orders AS o ON c.c_custkey = o.o_custkey
        JOIN lineitem AS l ON o.o_orderkey = l.l_orderkey
        JOIN supplier AS s ON l.l_suppkey = s.s_suppkey
        JOIN nation AS n ON c.c_nationkey = n.n_nationkey
        JOIN region AS r ON n.n_regionkey = r.r_regionkey
        WHERE c.c_nationkey = s.s_nationkey
          AND r.r_name = ?
          AND o.o_orderdate >= CAST(? AS DATE)
          AND o.o_orderdate < CAST(? AS DATE) + INTERVAL 1 YEAR
        """,
        [region, date, date],
    ).fetchall()

    private_ids = []
    for _value, custkey, suppkey in rows:
        ids = [f"C:{custkey}"]
        if multi_private:
            ids.append(f"S:{suppkey}")
        private_ids.append(ids)

    return _sja_data("q5_sja_revenue", rows, private_ids, start)


def q7_sja_revenue(
    con: duckdb.DuckDBPyConnection,
    *,
    nation1: str = "FRANCE",
    nation2: str = "GERMANY",
) -> ContributionData:
    """Extract scalar TPC-H Q7 revenue contributions."""

    start = time.perf_counter()
    rows = con.execute(
        """
        SELECT
            CAST(l.l_extendedprice * (1 - l.l_discount) AS DOUBLE) AS join_value,
            c.c_custkey,
            s.s_suppkey
        FROM supplier AS s
        JOIN lineitem AS l ON s.s_suppkey = l.l_suppkey
        JOIN orders AS o ON o.o_orderkey = l.l_orderkey
        JOIN customer AS c ON c.c_custkey = o.o_custkey
        JOIN nation AS n1 ON s.s_nationkey = n1.n_nationkey
        JOIN nation AS n2 ON c.c_nationkey = n2.n_nationkey
        WHERE (
            (n1.n_name = ? AND n2.n_name = ?)
            OR (n1.n_name = ? AND n2.n_name = ?)
        )
          AND l.l_shipdate >= DATE '1995-01-01'
          AND l.l_shipdate <= DATE '1996-12-31'
        """,
        [nation1, nation2, nation2, nation1],
    ).fetchall()

    private_ids = [
        [f"C:{custkey}", f"S:{suppkey}"] for _value, custkey, suppkey in rows
    ]
    return _sja_data("q7_sja_revenue", rows, private_ids, start)


def q12_sja_count(
    con: duckdb.DuckDBPyConnection,
    *,
    shipmodes: tuple[str, ...] = ("MAIL", "SHIP"),
    date: str = "1994-01-01",
    privacy_mode: str = "order",
) -> ContributionData:
    """Extract scalar TPC-H Q12 count contributions."""

    if privacy_mode not in {"order", "customer"}:
        msg = "privacy_mode must be 'order' or 'customer'"
        raise ValueError(msg)

    start = time.perf_counter()
    rows = con.execute(
        """
        SELECT
            1.0 AS join_value,
            o.o_orderkey,
            o.o_custkey
        FROM orders AS o
        JOIN lineitem AS l ON o.o_orderkey = l.l_orderkey
        WHERE l.l_shipmode IN (SELECT unnest(?))
          AND l.l_commitdate < l.l_receiptdate
          AND l.l_shipdate < l.l_commitdate
          AND l.l_receiptdate >= CAST(? AS DATE)
          AND l.l_receiptdate < CAST(? AS DATE) + INTERVAL 1 YEAR
        """,
        [list(shipmodes), date, date],
    ).fetchall()

    if privacy_mode == "customer":
        private_ids = [[f"C:{custkey}"] for _value, _orderkey, custkey in rows]
    else:
        private_ids = [[f"O:{orderkey}"] for _value, orderkey, _custkey in rows]
    return _sja_data("q12_sja_count", rows, private_ids, start)


def q3_spja_order_count(
    con: duckdb.DuckDBPyConnection,
    *,
    segment: str = "BUILDING",
    date: str = "1995-03-15",
) -> ContributionData:
    """Extract TPC-H Q3-style distinct qualifying order contributions."""

    start = time.perf_counter()
    rows = con.execute(
        """
        SELECT
            o.o_orderkey,
            c.c_custkey
        FROM customer AS c
        JOIN orders AS o ON c.c_custkey = o.o_custkey
        JOIN lineitem AS l ON o.o_orderkey = l.l_orderkey
        WHERE c.c_mktsegment = ?
          AND o.o_orderdate < CAST(? AS DATE)
          AND l.l_shipdate > CAST(? AS DATE)
        """,
        [segment, date, date],
    ).fetchall()

    keys = [("ORDER", int(orderkey)) for orderkey, _custkey in rows]
    private_ids = [[f"C:{custkey}"] for _orderkey, custkey in rows]
    return _spja_data("q3_spja_order_count", keys, private_ids, start)


def q10_spja_customer_count(
    con: duckdb.DuckDBPyConnection,
    *,
    date: str = "1993-10-01",
) -> ContributionData:
    """Extract TPC-H Q10-style distinct returned-customer contributions."""

    start = time.perf_counter()
    rows = con.execute(
        """
        SELECT
            c.c_custkey
        FROM customer AS c
        JOIN orders AS o ON c.c_custkey = o.o_custkey
        JOIN lineitem AS l ON o.o_orderkey = l.l_orderkey
        JOIN nation AS n ON c.c_nationkey = n.n_nationkey
        WHERE o.o_orderdate >= CAST(? AS DATE)
          AND o.o_orderdate < CAST(? AS DATE) + INTERVAL 3 MONTH
          AND l.l_returnflag = 'R'
        """,
        [date, date],
    ).fetchall()

    keys = [("CUSTOMER", int(custkey)) for (custkey,) in rows]
    private_ids = [[f"C:{custkey}"] for (custkey,) in rows]
    return _spja_data("q10_spja_customer_count", keys, private_ids, start)


def _sja_data(
    query_id: str,
    rows: list[tuple[Any, ...]],
    private_ids: list[list[str]],
    start_time: float,
) -> ContributionData:
    values = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
    return ContributionData(
        query_id=query_id,
        query_type="SJA",
        join_values=values,
        private_ids_per_join_row=private_ids,
        projection_keys=None,
        projected_values=None,
        true_answer=float(values.sum()),
        extraction_time_sec=time.perf_counter() - start_time,
    )


def _spja_data(
    query_id: str,
    projection_keys: list[Any],
    private_ids: list[list[str]],
    start_time: float,
) -> ContributionData:
    projected_values = dict.fromkeys(projection_keys, 1.0)
    values = np.ones(len(projection_keys), dtype=np.float64)
    return ContributionData(
        query_id=query_id,
        query_type="SPJA",
        join_values=values,
        private_ids_per_join_row=private_ids,
        projection_keys=projection_keys,
        projected_values=projected_values,
        true_answer=float(sum(projected_values.values())),
        extraction_time_sec=time.perf_counter() - start_time,
    )
