"""TPC-H schema metadata used by the experiment pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TableSpec:
    """TPC-H table metadata used for loading `.tbl` files."""

    name: str
    columns: tuple[tuple[str, str], ...]
    primary_key: str | None = None

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return column names in TPC-H file order."""

        return tuple(name for name, _dtype in self.columns)


TPCH_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "region",
        (
            ("r_regionkey", "INTEGER"),
            ("r_name", "VARCHAR"),
            ("r_comment", "VARCHAR"),
        ),
        "r_regionkey",
    ),
    TableSpec(
        "nation",
        (
            ("n_nationkey", "INTEGER"),
            ("n_name", "VARCHAR"),
            ("n_regionkey", "INTEGER"),
            ("n_comment", "VARCHAR"),
        ),
        "n_nationkey",
    ),
    TableSpec(
        "supplier",
        (
            ("s_suppkey", "INTEGER"),
            ("s_name", "VARCHAR"),
            ("s_address", "VARCHAR"),
            ("s_nationkey", "INTEGER"),
            ("s_phone", "VARCHAR"),
            ("s_acctbal", "DOUBLE"),
            ("s_comment", "VARCHAR"),
        ),
        "s_suppkey",
    ),
    TableSpec(
        "customer",
        (
            ("c_custkey", "INTEGER"),
            ("c_name", "VARCHAR"),
            ("c_address", "VARCHAR"),
            ("c_nationkey", "INTEGER"),
            ("c_phone", "VARCHAR"),
            ("c_acctbal", "DOUBLE"),
            ("c_mktsegment", "VARCHAR"),
            ("c_comment", "VARCHAR"),
        ),
        "c_custkey",
    ),
    TableSpec(
        "orders",
        (
            ("o_orderkey", "INTEGER"),
            ("o_custkey", "INTEGER"),
            ("o_orderstatus", "VARCHAR"),
            ("o_totalprice", "DOUBLE"),
            ("o_orderdate", "DATE"),
            ("o_orderpriority", "VARCHAR"),
            ("o_clerk", "VARCHAR"),
            ("o_shippriority", "INTEGER"),
            ("o_comment", "VARCHAR"),
        ),
        "o_orderkey",
    ),
    TableSpec(
        "lineitem",
        (
            ("l_orderkey", "INTEGER"),
            ("l_partkey", "INTEGER"),
            ("l_suppkey", "INTEGER"),
            ("l_linenumber", "INTEGER"),
            ("l_quantity", "DOUBLE"),
            ("l_extendedprice", "DOUBLE"),
            ("l_discount", "DOUBLE"),
            ("l_tax", "DOUBLE"),
            ("l_returnflag", "VARCHAR"),
            ("l_linestatus", "VARCHAR"),
            ("l_shipdate", "DATE"),
            ("l_commitdate", "DATE"),
            ("l_receiptdate", "DATE"),
            ("l_shipinstruct", "VARCHAR"),
            ("l_shipmode", "VARCHAR"),
            ("l_comment", "VARCHAR"),
        ),
        None,
    ),
    TableSpec(
        "part",
        (
            ("p_partkey", "INTEGER"),
            ("p_name", "VARCHAR"),
            ("p_mfgr", "VARCHAR"),
            ("p_brand", "VARCHAR"),
            ("p_type", "VARCHAR"),
            ("p_size", "INTEGER"),
            ("p_container", "VARCHAR"),
            ("p_retailprice", "DOUBLE"),
            ("p_comment", "VARCHAR"),
        ),
        "p_partkey",
    ),
    TableSpec(
        "partsupp",
        (
            ("ps_partkey", "INTEGER"),
            ("ps_suppkey", "INTEGER"),
            ("ps_availqty", "INTEGER"),
            ("ps_supplycost", "DOUBLE"),
            ("ps_comment", "VARCHAR"),
        ),
        None,
    ),
)

TPCH_TABLE_MAP: Mapping[str, TableSpec] = MappingProxyType(
    {table.name: table for table in TPCH_TABLES}
)


def table_names() -> list[str]:
    """Return the TPC-H table names recognized by the loader."""

    return [table.name for table in TPCH_TABLES]
