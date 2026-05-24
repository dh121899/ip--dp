# Data Directory

Real TPC-H data is not included in this repository.

Generate TPC-H `.tbl` files using the official TPC-H dbgen tool and place them under:

```text
data/raw/sf0.01/
data/raw/sf0.1/
```

Each scale-factor folder should contain:

```text
region.tbl
nation.tbl
supplier.tbl
customer.tbl
orders.tbl
lineitem.tbl
part.tbl
partsupp.tbl
```

Generated cache files may be stored under `data/cache/`, but both `data/raw/` and `data/cache/` are ignored by Git.

This project is not an official TPC-H benchmark result.
