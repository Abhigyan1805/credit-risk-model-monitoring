# Model Monitoring Engine (`monitoring/`)

This component is the **monitoring engine** for the credit-risk monitoring
system: a tested Python library plus a runner that turns a model's monthly
scored vintages into the long monitoring-results table a bank reviews each
month. It is deliberately pure **numpy / pandas / scipy** so every number is
explainable line by line and it runs on a laptop without a cluster.

It answers the four questions a model-monitoring workstream asks every month:

| Question | Metric | Alert direction |
| --- | --- | --- |
| Has the applicant population moved? | `psi_score`, `csi` per feature | higher is worse |
| Can the model still rank good vs bad? | `auc` (context), `gini`, `ks` | lower is worse |
| Are predicted default rates right? | `hl_pvalue`, `binomial_pvalue` per band, `calibration_ratio` | p lower is worse; ratio two-sided |
| What happened this month overall? | `overall_rag` (0/1/2 for G/A/R) | worst of the above |

## Where it sits

```
model.score -> data/scored/vintage=<month>.parquet
monitoring.run --month <M>  ->  data/monitoring/results/month=<M>.parquet  ->  reporting + Tableau
```

The engine is the statistical core. It does **not** train the PD model
(`model/`) and does **not** draw the dashboard (Tableau, by the captain); it
produces the `RESULT_COLUMNS` table those pieces consume. Entry point:
`python -m monitoring.run --month YYYY-MM` (wired to `make monitor`, which loops
over all 36 vintages). Thresholds live in `config/thresholds.yaml`.

## The metrics in plain English

### Population Stability Index (PSI) and Characteristic Stability Index (CSI)

Both the dev reference and the current month's scores are split into the **same
10 bins** — equal-frequency deciles frozen on the dev reference — and we compare
the share of loans per bin:

```
PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)
```

Both distributions identical → 0; `< 0.10` stable, `0.10–0.25` worth a look,
`> 0.25` material shift. CSI is the same formula for one *characteristic*:
numeric features use dev quantile bins, categorical features compare category by
category over the union of categories. Empty bins are clipped at a tiny epsilon
so the result stays finite.

### Discrimination: AUC, Gini, KS

Computed only on **seasoned vintages** — months whose loans have had time to
default or fully repay (`outcome_observable`), so the outcome is observed.
AUC is the probability a random default is scored riskier than a random
non-default (ties half), via the rank/Mann-Whitney formula. `Gini = 2·AUC − 1`.
KS is the biggest gap between the default and non-default score CDFs.

**The Gini and KS traffic lights are relative to the development baseline, not
absolute.** The real PD model's out-of-time Gini is ~0.33; an absolute floor of
0.40 would flag it red every month. So the lights are driven by *deterioration
from the dev baseline*: `drop = (baseline − value) / baseline`, with green
`< 10%`, amber `10–25%`, red `> 25%`. Low absolute floors (Gini 0.25/0.20, KS
0.15/0.10) are kept only as a catastrophic **secondary** signal. The baseline
Gini/KS are computed from the dev reference at run time; the results table
stores the derived absolute thresholds so a dashboard can draw them as lines.

### Calibration: predicted vs observed

Discrimination can stay healthy while the *level* is wrong.

* **`calibration_ratio`** = observed default rate / mean predicted PD, reported
  overall (`ALL`) and per score band (`band=<k>`). The RAG is **two-sided**:
  severity `|ratio − 1|` vs amber `0.15`, red `0.30`, so predicting too high or
  too low both flag. Band-level ratios are **diagnostic only** (`rag = NA`):
  in low-risk bands the expected default count is tiny so the ratio is unstable;
  the band-level binomial test below is the alerting signal.
* **`binomial_pvalue`** per band — an exact two-sided binomial test of the
  observed defaults vs the band's predicted rate. This is the small-sample
  "Jeffreys-style" view and localises miscalibration.
* **`hl_pvalue`** overall — the Hosmer-Lemeshow statistic
  `Σ (Oᵢ − Eᵢ)² / (nᵢ·pᵢ·(1−pᵢ))` against a chi-square with `bands − 2` d.o.f.
  A small p-value means the model is miscalibrated overall.

## Traffic lights (`config/thresholds.yaml`)

Thresholds are **data, not code**, so the policy is tunable and auditable. Three
modes:

| mode | metric | rule |
| --- | --- | --- |
| `absolute` | `psi_score`, `csi` | compared directly (higher worse) |
| `absolute` | `hl_pvalue`, `binomial_pvalue` | compared directly (lower worse) |
| `relative_drop` | `gini`, `ks` | drop from dev baseline (higher drop worse) + low absolute floor |
| `deviation` | `calibration_ratio` | two-sided `|ratio − 1|` |

`alerts.rag` maps a value (and, for Gini/KS, the baseline) to `GREEN / AMBER /
RED / NA`. `NA` means not computable (e.g. outcomes not observable) and is
ignored when rolling up the headline.

`overall_rag` is the **worst** rag over the model-level metrics — PSI, the worst
CSI, Gini, KS, overall `calibration_ratio` and `hl_pvalue` — stored as 0/1/2 for
green/amber/red. Band-level rows (`band=<k>`) are drill-down detail, deliberately
**excluded** from the headline: with ~10 bands a per-band test carries a
multiple-testing false-positive risk, and `hl_pvalue` already covers overall
calibration.

## Design choices (and why)

* **Frozen dev bins.** Score and feature bins come from the dev reference and are
  reused every month, so each month is measured against *development*, not
  against the previous month. No drift compounding; months are comparable.
* **Relative Gini/KS with floors.** See above — the model shipped at ~0.33 Gini,
  so absolute thresholds alone would be permanently red. Relative deterioration
  is the informative signal; floors catch collapse.
* **Band-level ratio is diagnostic.** Avoids alerting on ratios with near-zero
  expected counts; the binomial test (which respects n) drives band alerts.
* **Immature vintages get `NA`, not omission.** 2016 vintages (outcomes not yet
  observable) still emit every row with a NaN value and `NA` rag, so every month
  has the same schema for reporting.
* **Pure numpy/pandas/scipy.** No heavy dependencies, low memory, no Docker —
  keeps four builders working on one 5 GB machine.

## Output table (`RESULT_COLUMNS`)

One row per `(month, metric, segment)`, validated by
`common.schemas.validate_results`:

`month, metric, segment, value, amber_threshold, red_threshold, direction
(higher_worse | lower_worse), rag (GREEN | AMBER | RED | NA), n_obs,
model_version`.

Segments: `ALL`, a feature name (`csi`), or `band=<k>` (1 safest … 10 riskiest,
the model's frozen score band). Metrics: `volume`, `mean_pd`, `default_rate`,
`psi_score`, `csi`, `auc`, `gini`, `ks`, `calibration_ratio`, `hl_pvalue`,
`binomial_pvalue`, `overall_rag`.

## Running it

```bash
# unit tests (no dataset needed, <1s)
.venv/bin/python -m pytest monitoring -q
# or from the repo root:
make -f monitoring/Makefile test

# real run for one vintage (needs `make train` and `make score` first)
.venv/bin/python -m monitoring.run --month 2014-06

# offline demo on synthetic scored data -> out/monitoring/monitoring_results.csv
.venv/bin/python -m monitoring --months 6
```

The synthetic book (`monitoring/synthetic.py`) drifts in the later months on
purpose, so the demo produces real GREEN/AMBER/RED alerts and is a useful
walkthrough.

## How this maps to the SCMAC PS brief

* **"Automated workflows for model monitoring in Python"** — this engine is the
  monitoring logic the Airflow DAG calls each month.
* **"Population stability, discrimination, calibration"** — all three with
  textbook definitions, unit-tested against hand-computed examples.
* **"Red/amber/green the way banks flag model health"** — explicit,
  config-driven RAG with a single headline verdict and the evidence behind it.
* **Explainability** — every metric is a few lines of readable code and this
  document is written so the captain can defend it in an interview.

## Likely interview questions

**1. What is PSI and what does 0.3 mean?**
PSI compares the share of the population in each score bin between the baseline
and the current month as `(actual−expected)·ln(actual/expected)`. 0 means
identical; 0.3 is a material population shift that makes you question whether the
model's assumptions still hold. We don't act on PSI alone; we escalate it
alongside discrimination and calibration.

**2. Why monitor Gini *drop* rather than the raw Gini?**
Because the question is "has this month degraded relative to validation?" A
Gini of 0.33 is fine if it shipped at 0.33 and alarming if it shipped at 0.60.
Our model's out-of-time Gini is ~0.33, so an absolute floor of 0.40 would be red
every month — hence relative drop (`<10%` green, `10–25%` amber, `>25%` red) with
low absolute floors as a catastrophic backstop.

**3. Why use equal-frequency (decile) bands instead of equal-width?**
Default scores are heavily skewed, so equal-width bins would put almost everyone
in one bin and tell you nothing. Decile bins give each band a similar sample
size and are the model's own frozen score bands, so calibration is reported on
the same bands the scorecard uses.

**4. Why is the band-level calibration ratio `NA` but the binomial test live?**
In a low-risk band the expected default count can be a fraction of a loan, so the
ratio observed/predicted is unstable and would false-alarm. The exact binomial
test accounts for the sample size and is the correct per-band signal; the ratio
is still shown as a colourless diagnostic.

**5. Why do you skip discrimination for recent months?**
Outcomes aren't observed yet — a loan issued last month hasn't had time to
default. Measuring AUC on unseasoned loans would understate the default rate and
give false comfort, so those metrics are emitted as `NA`.

**6. How do you validate the metrics without trusting the library?**
Every metric has a hand-computed test: PSI on a three-bin example (0.09808), AUC
on `[0,0,1,1]`/`[0.1,0.4,0.35,0.8]` (= 0.75), HL on three 100-loan bands
(= 100/9, 1 d.o.f.), plus a brute-force ROC cross-check, known-good/known-bad
calibration cases and the relative-drop boundaries. A test also runs the full
`monitoring.run` path on a temporary data root and re-validates the Parquet.

## Status / contract

Implemented to `docs/CONTRACTS.md` sections 5, 8, 9 and 11: writes
`paths.monitoring_path(month)` in the `RESULT_COLUMNS` schema, validated with
`validate_results`; thresholds in `config/thresholds.yaml` (owned by this
component); entry point `python -m monitoring.run --month`. Any interface gap is
raised as `needs-decision [key=contract]` rather than edited into
`CONTRACTS.md`.
