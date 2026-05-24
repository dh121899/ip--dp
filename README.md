# R2T-Based Differential Privacy Mechanisms for TPC-H Query Processing

This repository implements and evaluates R2T-based differential privacy mechanisms on TPC-H-derived relational queries with foreign-key constraints.

The project is based on:

- R2T paper: https://www.cse.ust.hk/~yike/R2T.pdf
- ShiftedInverse paper: https://www.cse.ust.hk/~yike/ShiftedInverse.pdf
- TPC-H specification: https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp

This project is **not an official TPC-H benchmark result**. TPC-H data and query structures are used only as an experimental workload for evaluating differentially private query mechanisms.

## Implemented Mechanisms

- **DirectLaplace**: baseline mechanism using global sensitivity.
- **FixedTau family**: baseline using public fixed truncation thresholds.
- **R2T**: main mechanism using LP-based truncation and adaptive threshold selection.
- **ShiftedInverse approximation**: optional lightweight educational approximation for selected SJA sum/count cases.

The ShiftedInverse implementation is not a full reproduction of the ShiftedInverse paper.

## Implemented Queries

| Query ID | Type | Description |
|---|---|---|
| `q5_sja_revenue` | SJA | Scalar TPC-H Q5 revenue |
| `q7_sja_revenue` | SJA | Scalar TPC-H Q7 revenue |
| `q12_sja_count` | SJA | Scalar TPC-H Q12 count |
| `q3_spja_order_count` | SPJA | TPC-H Q3-style distinct order count |
| `q10_spja_customer_count` | SPJA | TPC-H Q10-style distinct customer count |

## Project Structure

```text
src/r2t_tpch/
  schema.py
  load_tpch.py
  contributions.py
  queries.py
  lp_truncation.py
  mechanisms_r2t.py
  mechanisms_shifted_inverse.py
  metrics.py
  plots.py
  run_experiment.py

scripts/
  repair_optional_results.py
  run_parameter_sweeps.py

tests/
  unit and smoke tests

data/
  README.md
```

## Data

Real TPC-H `.tbl` files are **not included** in this repository.

Generate TPC-H data externally using the official TPC-H tools, then place the files under:

```text
data/raw/sf0.01/
data/raw/sf0.1/
```

Required files:

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

Example WSL command for generating SF=0.1:

```bash
cd "/mnt/c/R2T-TOOLS/TPC-H V3.0.1/dbgen"
chmod +x dbgen
./dbgen -f -s 0.1

mkdir -p /mnt/c/r2t-tpch-dp/data/raw/sf0.1
cp region.tbl nation.tbl supplier.tbl customer.tbl orders.tbl lineitem.tbl part.tbl partsupp.tbl \
  /mnt/c/r2t-tpch-dp/data/raw/sf0.1/
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

If the project uses `requirements.txt` instead:

```powershell
pip install -r requirements.txt
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

## Run Main Experiment

Example for real TPC-H SF=0.1:

```powershell
.\.venv\Scripts\python.exe -m r2t_tpch.run_experiment `
  --data-mode real `
  --data-dir data\raw\sf0.1 `
  --scale-factor 0.1 `
  --expected-scale-factor 0.1 `
  --validate-row-counts `
  --allow-approx-row-counts `
  --out-dir results\sf0.1_real `
  --queries q5_sja_revenue q7_sja_revenue q12_sja_count q3_spja_order_count q10_spja_customer_count `
  --mechanisms DirectLaplace FixedTau R2T ShiftedInverseSum ShiftedInverseCount `
  --epsilon 1.0 `
  --beta 0.1 `
  --gsq 524288 `
  --fixed-taus 128 512 2048 8192 32768 131072 `
  --runs 30 `
  --seed 42
```

Generate plots:

```powershell
.\.venv\Scripts\python.exe -m r2t_tpch.plots `
  --results-dir results\sf0.1_real `
  --figures-dir figures\sf0.1_real
```

## Parameter Sweeps

Epsilon sweep:

```powershell
.\.venv\Scripts\python.exe scripts\run_parameter_sweeps.py `
  --sweep epsilon `
  --data-dir data\raw\sf0.1 `
  --scale-factor 0.1 `
  --expected-scale-factor 0.1 `
  --runs 10 `
  --seed 42 `
  --base-out-dir results\sweeps\epsilon_sf0.1 `
  --base-figures-dir figures\sweeps\epsilon_sf0.1
```

GSQ sweep:

```powershell
.\.venv\Scripts\python.exe scripts\run_parameter_sweeps.py `
  --sweep gsq `
  --data-dir data\raw\sf0.1 `
  --scale-factor 0.1 `
  --expected-scale-factor 0.1 `
  --runs 10 `
  --seed 42 `
  --base-out-dir results\sweeps\gsq_sf0.1 `
  --base-figures-dir figures\sweeps\gsq_sf0.1
```

## Outputs

The experiment runner generates:

- `results_runs.csv`
- `results_summary.csv`
- `tau_trace.csv`
- `contribution_stats.csv`
- `shifted_inverse_trace.csv`

The plotting module generates figures for error comparison, runtime analysis, contribution statistics, tau sweep, and parameter sweeps.

## Notes

- Real TPC-H data is not committed.
- Full generated results and figures are ignored by default.
- Selected figures or small summary files may be placed under `docs/` if needed.
- For SPJA queries, contribution statistics report `isq_proxy`, not the full theoretical ISQ.
- The optional ShiftedInverse module is a lightweight approximation only.
