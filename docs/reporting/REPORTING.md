# Reporting component (builder #4)

Plain-English notes on what `reporting/` does, the design choices behind it, and
the questions it is likely to draw. Contract: `docs/CONTRACTS.md` sections 6 and
11. Tableau build guide: `docs/DASHBOARD_SPEC.md`.

## 1. What it does

The monitoring engine writes one Parquet file per month holding a long table of
metrics — `data/monitoring/results/month=<YYYY-MM>.parquet`, one row per
`(month, metric, segment)`. Reporting turns several months of those rows into
two things:

1. **A self-contained monthly HTML report** —
   `data/reports/monitoring_report_<YYYY-MM>.html` — the dated, printable record
   kept with the model. Charts are embedded base64 PNGs, so the single file
   opens anywhere with no internet.
2. **Tableau-ready CSV extracts** in `data/tableau/` — the tidy numbers for the
   dashboard the captain builds by following `docs/DASHBOARD_SPEC.md`.

## 2. The flow

```
data/monitoring/results/month=<M>.parquet  (long: month, metric, segment, value, rag, …)
        │  data.load_results(up_to)
        ▼
  long DataFrame + month_date
        │
        ├─ tables.build_monthly_summary  ── wide: one row per month + rag_* columns
        ├─ tables.build_csi_by_feature   ── month × feature CSI
        ├─ tables.build_calibration_by_band ── month × band ratio + p-value
        ├─ tables.build_breaches         ── amber/red rows + messages
        │
        ├─ charts.*  (matplotlib → base64 PNG)
        │        └─ report.render_report → monitoring_report_<M>.html
        │
        └─ data.load_scored_bands / load_dev_band_shares
                 └─ tables.build_score_distribution → score_distribution.csv

extracts.write_extracts → data/tableau/*.csv + data_dictionary.{csv,md}
```

## 3. How to run

```bash
make monitor                                  # builder #3 produces the results files
python -m reporting.report --month 2016-12    # writes data/reports/monitoring_report_2016-12.html
make extracts                                 # writes data/tableau/*.csv
python -m reporting.report --month 2014-06 --out /tmp/report.html   # custom path
```

`CRM_DATA_DIR` selects the data root (see `common/paths.py`). `make report`
loops all monitor months; `make extracts` runs once over every month present.

## 4. Outputs

| Output | Path |
| --- | --- |
| Monthly report | `data/reports/monitoring_report_<YYYY-MM>.html` |
| Long extract | `data/tableau/monitoring_results.csv` (long table + `month_date`) |
| Monthly wide | `data/tableau/monthly_summary.csv` |
| Score distribution | `data/tableau/score_distribution.csv` |
| Feature CSI | `data/tableau/csi_by_feature.csv` |
| Calibration by band | `data/tableau/calibration_by_band.csv` |
| Breach list | `data/tableau/breaches.csv` |
| Data dictionary | `data/tableau/data_dictionary.{csv,md}` |

Month dates are ISO `YYYY-MM-01` so Tableau treats them as dates.

## 5. Design choices (review-ready)

**Reporting renders RAG; it does not invent it.** The monitoring engine already
writes a `rag` value per metric row using `config/thresholds.yaml`. The extracts
carry `overall_rag` and `rag_<metric>` straight through, and Tableau colours by
those columns. One rule, defined once, in one place.

**Tidy CSVs with no colour baked in.** Colour does not survive a CSV reload;
colour is a Tableau calculated field driven by the `rag_*` text column. Keeping
the CSV numeric keeps it reusable by any tool.

**The report is one self-contained file.** Charts are base64 PNGs embedded in the
HTML, so it can be emailed or archived and still render — no CDN, no side files.
That matters for an audit trail: the report is evidence and must not depend on a
network.

**Why matplotlib, not Plotly?** `pyproject.toml` pins matplotlib; adding a
dependency is a cross-team change (contract section 11). Matplotlib also renders
deterministically and headless (`Agg`), which keeps CI green and memory low on
the shared build machine.

**Long input, wide output.** The producer's long table is the right storage shape
(one fact per row, easy to append a metric). But Tableau wants one row per month
for trend lines, so `build_monthly_summary` pivots it. The long file is still
exported unchanged for anyone who wants to slice by metric.

**Graceful on missing data.** A vintage 2016-01..2016-12 has no observable
outcomes yet, so AUC/Gini/KS/default-rate rows are `NA`. The pivots keep those as
blanks rather than errors, and the report shows "—". This mirrors reality: you
monitor population stability from day one and outcome metrics only once loans
mature.

**Score distribution needs the loans, not just the metrics.** Band shares are not
in the results table, so reporting reads `data/scored/vintage=*.parquet` (and the
`dev` reference) for the score-distribution extract. If the vintages are absent
the extract is written empty rather than crashing — the rest of the run still
succeeds.

## 6. Tests

`python -m pytest reporting/tests -q` — 27 tests, no dataset required. A
synthetic data root is generated in a temp dir and pointed at via `CRM_DATA_DIR`
(`reporting/tests/_synth.py`, `conftest.py`), covering the loaders, every table
builder, CSV writing and the data dictionary, and the rendered report
(sections present, charts embedded, only months up to the requested month used).
The whole repository suite runs with `make test`.

## 7. Likely interview questions and answers

**Q1. What is PSI and what value is bad?**
The Population Stability Index measures how far a distribution has shifted from
a reference: it sums `(actual% − expected%) × ln(actual% / expected%)` over bins.
Below 0.10 is negligible, 0.10–0.25 is a shift to investigate, above 0.25 the
population has moved enough to question the model. Here PSI is computed by the
monitoring engine on the score distribution (10 dev-decile bins); CSI is the same
idea applied per input feature, which finds *which* variable moved.

**Q2. Why is RAG computed in Python and only rendered in Tableau?**
So there is a single source of truth. The thresholds live in
`config/thresholds.yaml`; the monitoring engine classifies each metric once and
writes the status into the results. The report and the dashboard both read that
same status, so they can never disagree, and changing a threshold never requires
editing Tableau.

**Q3. Why are some months' Gini/KS/calibration blank?**
Only vintages issued up to 2015-12 have fully matured 36-month outcomes. For
2016 vintages we can still measure population stability (PSI/CSI) but not
discrimination or calibration, so those rows are `NA` and the report shows "—".
Reporting deliberately keeps them blank instead of guessing.

**Q4. How does the calibration check differ from discrimination?**
Discrimination asks "does the model rank correctly?" (Gini/KS/AUC). Calibration
asks "are the predicted probabilities right?" — here the observed default rate
divided by the mean predicted PD per band. A model can rank well and still
predict 5% when the truth is 10%; the calibration ratio catches that.

**Q5. Why two output shapes (long and wide)?**
The long `monitoring_results.csv` is the faithful, extensible record — one fact
per row, appending a metric costs nothing. The wide `monthly_summary.csv` is the
convenience shape for trend charts and KPI tiles. Both come from the same rows,
so they cannot drift apart.

**Q6. What would you add next?**
Trend alerting (flag a metric that has been amber for N months even if no single
month is red), a champion/challenger comparison sheet, and replacing the raw
calibration ratio with a proper statistical test (Hosmer-Lemeshow p-value) as the
primary signal — the `binomial_pvalue` per band is already extracted, so the
dashboard can show it today.
