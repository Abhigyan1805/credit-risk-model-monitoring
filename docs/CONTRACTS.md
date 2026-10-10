# Interface contracts

Everything between components is a **Parquet file under the data root** (`CRM_DATA_DIR`, default `./data`,
git-ignored). No database, no services: pandas + pyarrow only. Paths are resolved only through
`common/paths.py`; windows/constants only through `common/config.py`; column specs are mirrored in
`common/schemas.py` (`validate_prepared`, `validate_scored`, `validate_results`). If you need to change a
contract, change all three (doc, schema, test) and tell the lead.

## 1. Scope, windows, label (`common/config.py`)

| item | value |
|---|---|
| data | LendingClub accepted loans 2007-2018Q4 (Kaggle `wordsforthewise/lending-club`, CC0-1.0) |
| scope | **36-month loans only**, issued 2012-01 .. 2016-12 (a 36-month vintage issued up to 2015-12 has fully matured by the data end, so its outcomes are not right-censored) |
| development window | issue months 2012-01 .. 2013-12 (`window == "dev"`); train 2012-01..2013-06, out-of-time validation 2013-07..2013-12 |
| monitoring vintages | monthly, 2014-01 .. 2016-12 (36 vintages, `window == "monitor"`) |
| default label | `default_flag` = 1 for `Charged Off` / `Default`, 0 for `Fully Paid`, **NA** for anything else (Current, Late, In Grace) -- excluded from outcome metrics |
| observable outcomes | vintages 2014-01..2015-12 have outcomes (`outcome_observable=True`); vintages 2016-01..2016-12 are monitored on population stability only (their resolved loans are biased towards early defaults, so outcome metrics for them are `NA`) |

A *vintage* = all loans with `issue_month == M`. Month M's monitoring run answers "is the frozen model
still healthy on loans originated in M?".

## 2. Prepared loans: `data/prepared/loans.parquet`  (ingest -> model)
One row per loan; exact columns in `common/schemas.PREPARED_COLUMNS`: `loan_id, issue_month (YYYY-MM string),
window, default_flag (Int8), loan_status, loan_amnt, term_months, int_rate (%), installment, grade, sub_grade,
emp_length_years, home_ownership, annual_inc, verification_status, purpose, addr_state, dti, delinq_2yrs, fico,
inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util (%), total_acc, credit_history_years`.
Also `data/prepared/ingest_manifest.json` (row counts, dataset, licence).
Missing values are NaN/NA (no imputation in ingest). `int_rate`/`grade` are LendingClub's own pricing of
risk; the model builder decides whether to use them (document the choice).

## 3. Model artefacts (model -> model.score, monitoring may read)
`data/model/` : `pd_model.joblib` (or equivalent), `metadata.json` containing `model_version`, feature list,
training window, dev AUC/Gini/KS, and the **frozen score-band cut-points** (10 bands from dev PD deciles).
`score = base_points - factor * ln(odds)` style scorecard points (higher = safer); document the PDO.

## 4. Scored vintages: `data/scored/vintage=<YYYY-MM|dev>.parquet`  (model -> monitoring, reporting)
One row per loan: `loan_id, issue_month, pd, score, score_band (1 safest..10 riskiest), default_flag (Int8, NA
unless outcome_observable), outcome_observable (bool), model_version` **plus every column in
`common.config.CSI_FEATURES` copied unchanged from the prepared table** (raw, un-binned, so the monitoring
engine bins CSI itself using dev quantiles).
`vintage=dev.parquet` is the **reference population**: the out-of-time dev sample (issue 2013-07..2013-12) scored
by the model, same schema. Training rows are never in it. Rewriting a vintage overwrites only that file (idempotent).

## 5. Monitoring results: `data/monitoring/results/month=<YYYY-MM>.parquet`  (monitoring -> reporting)
Long format, one row per (month, metric, segment): `month, metric, segment, value, amber_threshold,
red_threshold, direction (higher_worse|lower_worse), rag (GREEN|AMBER|RED|NA), n_obs, model_version`.
`NA` rag = metric not computable (e.g. outcomes not observable). Required metrics (segment `ALL` unless noted):

| metric | segment | meaning |
|---|---|---|
| `volume`, `mean_pd`, `default_rate` | ALL | context (default_rate NA if not observable) |
| `psi_score` | ALL | PSI of the score distribution vs dev reference (10 dev-decile bins) |
| `csi` | feature name | characteristic stability index per CSI feature |
| `auc`, `gini`, `ks` | ALL | discrimination on the vintage (needs observable outcomes) |
| `calibration_ratio` | ALL, `band=<k>` | observed default rate / mean PD |
| `hl_pvalue` | ALL | Hosmer-Lemeshow style calibration test p-value |
| `binomial_pvalue` | `band=<k>` | per-band binomial test of PD vs observed |
| `overall_rag` | ALL | worst RAG over the key metrics (value = 0/1/2 for G/A/R) |

Thresholds are data in `config/thresholds.yaml` (not hard-coded). RAG is derived per row; `overall_rag` is the
headline status shown on the dashboard.

## 6. Reporting outputs (reporting)
* `data/reports/monitoring_report_<YYYY-MM>.html` -- self-contained monthly report (embedded charts).
* `data/tableau/*.csv` -- flat extracts for Tableau Public (CSV, UTF-8, ISO dates `YYYY-MM-01`): at minimum
  `monitoring_results.csv` (the long table + `month_date`), `monthly_summary.csv` (one row per month wide),
  `score_distribution.csv` (month x band share), `csi_by_feature.csv`, `calibration_by_band.csv`.
* `docs/DASHBOARD_SPEC.md` -- worksheets, fields, calculated fields, filters, layout for Tableau Public.

## 7. Entry points (all idempotent per month, safe to re-run)
```
python -m ingest.download            python -m ingest.prepare [--from-sample]     python -m ingest.sample
python -m model.train                python -m model.score --month YYYY-MM
python -m monitoring.run --month YYYY-MM
python -m reporting.report --month YYYY-MM   python -m reporting.extracts
```

## 8. Environment variables
| var | default | use |
|---|---|---|
| `CRM_DATA_DIR` | `./data` | data root (`make demo` uses `data/demo`) |
| `CRM_RAW_DIR` | `$CRM_DATA_DIR/raw` | raw download location |
| `KAGGLE_API_TOKEN` | read from `~/.kaggle/access_token` | only for `ingest.download`; never written anywhere |
| `AIRFLOW_HOME` | `./airflow_home` | set by orchestration scripts |

## 9. Make targets
`setup`, `lint`, `test`, `download`, `ingest`, `sample`, `train`, `score`, `monitor`, `report`, `extracts`,
`e2e` (full real run, all vintages), `demo` (sample fixture, minutes), `airflow-test`, `clean`.
`score|monitor|report` loop over all 36 months via `common.config.monitor_months()`.

## 10. Testing convention
Unit tests live next to the code (`model/test_*.py` etc.) or in `tests/`, run on **the committed fixture**
`tests/fixtures/loans_sample.parquet` (a real stratified sample, ~70 loans per issue month, prepared schema) via
`common.paths.SAMPLE_LOANS`. CI runs `ruff check`, `ruff format --check`, `pytest` only: no downloads, no Airflow.

## 11. Dependencies / ownership
Add a dependency by pinning it in `pyproject.toml` `[project] dependencies` (builders: tell the lead, one
pin per line, no ranges). Python 3.12. Ownership: `ingest/`, `orchestration/`, `common/`, README/RESULTS/WALKTHROUGH =
lead; `model/` = #2; `monitoring/`, `config/thresholds.yaml` = #3; `reporting/`, `docs/DASHBOARD_SPEC.md` = #4.
