# Model Monitoring Engine (`monitoring/`)

This component is the **monitoring engine** for the credit-risk monitoring
system: a tested Python library plus a runner that turns a model's monthly
outputs into the evidence a bank reviews each month. It is deliberately kept to
pure **numpy / pandas / scipy** so every number is explainable line by line and
it runs on a laptop without a cluster.

It covers the four questions a model-monitoring workstream asks every month:

| Question | Metric | Alert |
| --- | --- | --- |
| Has the applicant population moved? | PSI (score), CSI (per feature) | green/amber/red |
| Can the model still rank good vs bad? | AUC, Gini, KS | green/amber/red |
| Are predicted default rates right? | Hosmer-Lemeshow, per-band binomial test | green/amber/red |
| What happened this month overall? | one rolled-up RAG verdict + alert rows | red/amber/green |

## Where it sits

```
data / model scores  ->  monitoring/  ->  monitoring-results table  ->  report + Tableau
                          (this piece)
```

The engine is the statistical core. It does **not** train the PD model
(builder #1) and does **not** draw the dashboard (Tableau, by the captain);
it produces the tidy tables those pieces consume.

## The metrics in plain English

### Population Stability Index (PSI) and Characteristic Stability Index (CSI)

We split both the baseline and the current month's scores into the **same** 10
bins (equal-frequency bins frozen from the development sample). For each bin we
compare the share of loans, `expected` vs `actual`:

```
PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)
```

* Both distributions identical -> PSI = 0.
* A small shift -> small positive number. Rule of thumb: `< 0.10` stable,
  `0.10–0.25` worth a look, `> 0.25` material shift.
* The CSI is exactly the same formula applied to one *characteristic*
  (feature). Numeric features use baseline quantile bins; categorical features
  compare category by category with the union of categories.

`monitoring/stability.py` also clips empty bins at a tiny epsilon so a bin that
is populated in one month but empty in another gives a finite, comparable value
rather than infinity.

### Discrimination: AUC, Gini, KS

Computed only on **seasoned vintages** — loans old enough that they have
defaulted or fully repaid, so the outcome is actually observed.

* **AUC** is the probability a random default is scored riskier than a random
  non-default (ties count as half). Computed with the rank/Mann-Whitney formula,
  which is exactly equivalent to the brute-force pairwise definition and is
  O(n log n).
* **Gini** = `2·AUC − 1`; 0 means no better than random, 1 is perfect.
* **KS** is the biggest gap between the score distributions of defaults and
  non-defaults (max |CDF_default − CDF_good|).

We track the **drop from the development baseline**, because a model that
shipped with Gini 0.50 and now reads 0.40 is the thing that matters, not the
absolute level.

### Calibration: predicted vs observed

Discrimination can stay healthy while the *level* is wrong. Calibration checks:

* **Per-band binomial test** — group loans into equal-frequency bands by
  predicted PD; in each band compare observed defaults with the expected count
  (`n × mean predicted PD`) using an exact two-sided binomial test. This is the
  small-sample-friendly "Jeffreys-style" view.
* **Hosmer-Lemeshow** — one overall goodness-of-fit statistic:
  `HL = Σ (O_i − E_i)² / (n_i·p_i·(1−p_i))`, compared with a chi-square with
  `bands − 2` degrees of freedom. Large statistic / small p-value = miscalibrated.

## Traffic-light thresholds (config-driven)

All thresholds live in `monitoring/config/monitoring_thresholds.yaml`, so the
policy can be tuned without touching code. Defaults:

| Metric | Green | Amber | Red |
| --- | --- | --- | --- |
| PSI / CSI | < 0.10 | 0.10 – 0.25 | ≥ 0.25 |
| Gini (relative drop vs baseline) | < 10% | 10 – 20% | ≥ 20% |
| KS (absolute drop vs baseline) | < 0.05 | 0.05 – 0.10 | ≥ 0.10 |
| HL p-value | ≥ 0.05 | 0.01 – 0.05 | < 0.01 |
| Per-band binomial | p ≥ 0.05 | p < 0.05 | p < 0.01 or large deviation |

`alerts.py` maps each value to a band; `overall_rag` takes the **worst**
non-`na` band so one red metric makes the month red.

## Design choices (and why)

* **Frozen baseline bins.** Score and feature bins are computed once from the
  development sample and reused every month, so each month is measured against
  *development*, not against the previous month. This avoids drift compounding
  and makes months comparable.
* **Equal-frequency bands.** Quantile bands give each score band a meaningful
  sample size and are robust to a skewed score distribution; the code splits by
  rank so tied scores stay deterministic.
* **Seasoning guard.** Discrimination/calibration are skipped when outcomes are
  not yet observed (the runner marks `calibration_available = False`), which is
  how real monitoring handles immature vintages.
* **Monthly summary + tidy long table.** The long `monitoring_results` table
  (one row per month/metric) is trivial to slice for dashboards; the wide
  `monthly_summary` is what a human reads.
* **Pure numpy/pandas/scipy.** No heavy dependencies, no Docker, low memory —
  matches the shared-machine constraint and keeps several builders working at
  once.

## Output tables

`run_monitoring(baseline, months, config)` returns four DataFrames:

* **`monitoring_results`** (long): `month, metric_group, metric, feature, value,
  baseline_value, band, alert, threshold_ref, details`.
* **`monthly_summary`** (wide): one row per month with `psi_score`, `psi_band`,
  `csi_max`, `csi_max_feature`, `auc`, `gini`, `gini_rel_drop`, `gini_band`,
  `ks`, `ks_abs_drop`, `ks_band`, `hl_pvalue`, `hl_statistic`, `hl_band`,
  `max_band_deviation`, `worst_band_band`, `overall_rag`, `n_loans`.
* **`calibration_detail`** (long): one row per (month, score band) with `n`,
  `mean_predicted_pd`, `expected_defaults`, `observed_defaults`,
  `observed_default_rate`, `deviation`, `binomial_pvalue`, `band_rag`.
* **`alert_summary`** (long): one row per month with `n_red`, `n_amber`, the
  names of the breached metrics (`red_metrics`, `amber_metrics`) and the
  `overall_rag` verdict — the per-month alert digest.

## Running it

```bash
# unit tests (no dataset needed, ~0.3s)
.venv/bin/python -m pytest monitoring/tests -q
# or from the repo root:
make -f monitoring/Makefile test

# full synthetic demo -> out/monitoring/*.csv
.venv/bin/python -m monitoring --out-dir out/monitoring --months 6
make -f monitoring/Makefile demo
```

The synthetic book (in `monitoring/__main__.py`) drifts in the later months on
purpose, so the demo lights up amber/red alerts and is a useful walkthrough.

## How this maps to the SCMAC PS brief

* **"Automated workflows for model monitoring in Python"** — this engine is the
  monitoring logic the orchestration layer (Airflow) calls each month.
* **"Population stability, discrimination, calibration"** — all three are
  implemented with textbook definitions and unit-tested against hand-computed
  examples.
* **"Red/amber/green the way banks flag model health"** — the RAG policy is
  explicit, configurable and evidence-producing.
* **Explainability** — every metric is a few lines of readable code and this
  document is written so the captain can defend it in an interview.

## Likely interview questions

**1. What is PSI and what does a value of 0.3 mean?**
PSI compares the share of the population in each score bin between the baseline
and the current month, summed as `(actual−expected)·ln(actual/expected)`. 0 means
identical; 0.3 is a material population shift — the kind that makes you ask
whether the model's assumptions still hold. We don't retire the model on PSI
alone; we escalate it alongside discrimination and calibration.

**2. Why monitor Gini *drop* rather than the raw Gini?**
Because the relevant question is "has this month's model degraded relative to
how it performed when it was validated?" A Gini of 0.40 is fine if the model
shipped at 0.41 and alarming if it shipped at 0.60. Relative drop normalises for
different baselines and portfolio mixes.

**3. Why use equal-frequency (quantile) bands instead of equal-width?**
Default scores are heavily skewed — most loans sit in a few low-risk bands — so
equal-width bins would put almost everyone in one bin and tell you nothing.
Quantile bins give every band a similar sample size, so the per-band calibration
test has power.

**4. Why do you skip discrimination for recent months?**
Outcomes aren't observed yet: a loan issued last month hasn't had time to
default. Measuring AUC on unseasoned loans would understate the default rate and
give false comfort. We only score months whose vintage is old enough — exactly
what a bank's monitoring calendar does.

**5. How do you validate the metrics without trusting the library?**
Every metric has a hand-computed test: PSI on a three-bin worked example
(0.09808), AUC on `[0,0,1,1]`/`[0.1,0.4,0.35,0.8]` (= 0.75), HL on three
100-loan bands (= 100/9 with 1 d.o.f.), plus a brute-force ROC cross-check and
known-good/known-bad calibration cases. If someone challenges a number, we point
at the test.

**6. Why Hosmer-Lemeshow *and* a per-band binomial test?**
HL gives one overall number (convenient for a RAG light) but is a single global
test and can miss a single badly-calibrated decile. The per-band binomial test
localises the problem and works with small band counts, so together they tell
you *whether* and *where* calibration has broken.

## Status / contract

This piece was built ahead of the shared skeleton and follows the interfaces in
`docs/CONTRACTS.md` once it landed (columns, env vars, Make targets). Any
needed interface changes are raised as `needs-decision [key=contract]` rather
than edited into `CONTRACTS.md` directly.
