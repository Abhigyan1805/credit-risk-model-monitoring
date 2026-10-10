# Walkthrough: explaining this project without notes

## 1. The one-minute story
A bank that uses a model to decide on loans must keep proving the model still works. I trained a probability-of-default
(PD) model on LendingClub loans issued 2012-13, froze it, and then replayed 36 later months (2014-2016) as if I were the
bank's model-risk analyst: each month the pipeline scores that month's new loans, checks whether the population has
shifted, whether the model still ranks risky borrowers correctly, whether predicted default rates match reality, and
raises green/amber/red alerts and a report. Airflow runs the monthly cycle; the results feed a Tableau dashboard.

## 2. Concepts in plain English
**PD model.** Predicts the probability a borrower will default (here: Charged Off / Default within the loan's life).
Banks use PD in lending decisions, pricing, and (in wholesale/corporate lending) in capital and provisioning models.
Mine is a *scorecard*: bin each variable, give each bin a weight of evidence (WoE = ln(good share / bad share)),
fit a logistic regression on the WoE values, and convert to points (600 points = 50:1 good:bad odds, +20 points
doubles the odds). It is chosen because every driver is explainable, which model validators and regulators need.

**Development vs out-of-time.** Train on 2012-01..2013-06; test on 2013-07..2013-12 (the future relative to training).
Out-of-time is the honest test; random splits flatter a model.

**Vintage.** All loans issued in one month. Monitoring month M = how does the frozen model do on vintage M.

**Default label.** Charged Off/Default = 1, Fully Paid = 0. Still-current loans are not yet resolved, so they are excluded
from outcome metrics. I used 36-month loans only so a vintage issued up to 2015-12 has fully played out by the end of the
data. Vintages in 2016 have no reliable outcomes yet, so only stability is monitored for them.

**PSI (Population Stability Index).** Compares the score distribution now with the reference (dev OOT sample) using the
same 10 bins: PSI = sum (actual% - expected%) x ln(actual% / expected%). Under 0.10 stable, 0.10-0.25 watch, over 0.25
significant shift. It says "the customers look different", not "the model is wrong".
**CSI** is the same formula for one input variable, so it tells you *which* variable moved (here: interest rate).

**AUC / Gini / KS.** Discrimination = can the model separate defaulters from non-defaulters? AUC is the probability a random
defaulter scores riskier than a random non-defaulter; Gini = 2 x AUC - 1; KS = the largest gap between the cumulative
score distributions of defaulters and non-defaulters. My model: OOT AUC 0.666, Gini 0.331, KS 0.247. Gini/KS alerts are
relative to that baseline (a drop of >10% amber, >25% red), because a model that shipped at 0.33 should not be red forever.

**Calibration.** Discrimination can be fine while the levels are wrong. Calibration ratio = observed default rate /
average predicted PD (1.0 = perfect). Hosmer-Lemeshow style test: group loans by predicted risk and test whether
observed defaults match predictions; per-band binomial test does the same band by band. Small p-value = mis-calibrated.

**RAG.** Each metric has amber/red thresholds in `config/thresholds.yaml`; overall status is the worst traffic light.
Banks use RAG so a committee can see model health at a glance and escalate red items.

## 3. What the pipeline found (see docs/RESULTS.md for exact numbers)
* Ranking power held: Gini stayed in 0.31-0.38 across all 24 observable months (dev 0.331).
* Calibration decayed: observed default rate went from 1.02x to 1.54x predicted. The model under-predicts risk for later
  vintages (LendingClub's lending mix/pricing changed; defaults rose), a classic reason to recalibrate.
* Population shifted: score PSI peaked at 0.28 in 2016-01; interest rate CSI was red in 24 months.
* So the overall status is mostly RED, driven by calibration and drift rather than lost ranking power: the recommendation a
  model owner would draw is "recalibrate / redevelop", not "model useless".

## 4. Pipeline and design choices
1. `ingest`: kaggle CLI download, chunked read (200k rows), keep ~25 columns, clean, write Parquet. Never loads 2.2M rows at once.
2. `model.train` / `model.score`: frozen model and frozen score-band cut-points, so bands mean the same thing each month.
3. `monitoring.run --month`: pure pandas/numpy/scipy; writes one long table (month, metric, segment, value, thresholds, RAG).
4. `reporting`: HTML report per month; CSV extracts shaped for Tableau; dashboard spec in docs/DASHBOARD_SPEC.md.
5. Airflow DAG `crm_monthly_monitoring`: ingest -> train-if-needed -> score -> monitor -> report -> extracts; schedule
   `@monthly`, `catchup=True`, `max_active_runs=1`. Idempotent: each task overwrites only its own month's file or skips if
   its output exists, so reruns and backfills are safe.
Design: Parquet files + explicit schemas (docs/CONTRACTS.md) instead of a database: light, inspectable, testable.
Thresholds in YAML, not code, so a validator can change policy without code changes. Fixture-based CI.

## 5. Likely interview questions
**Q: Why PSI and CSI both?** PSI is a single early-warning on the score; CSI localises which input drifted. A PSI breach
without a CSI breach would point to interactions or a data pipeline issue.
**Q: Gini is stable but the status is red. Why?** Ranking and calibration are different properties. Calibration ratio and
the HL test flagged under-prediction; a stable Gini means the order of risk is still right, only the level is off.
**Q: Why so many red calibration alerts? Is the test too strict?** With ~10,000 loans a month the test detects tiny
differences (high power), and loans within a vintage are correlated, which violates the test's independence assumption.
A bank would pair the p-value with an effect-size threshold (the calibration ratio) and judge by materiality. Here
the ratio really moves to 1.5, so the red is deserved in late 2015, but the early-2014 p-value alerts are about sensitivity.
**Q: Why only 36-month loans and why not monitor 2016 outcomes?** Right-censoring: loans from 2016 are not finished by 2018,
and the ones already resolved are mostly early defaults, which would bias default rates upward.
**Q: What is the difference between PIT and TTC PD?** Point-in-time reflects current conditions; through-the-cycle averages
over the cycle. My model is closer to point-in-time (origination attributes, one regime); this is a limitation.
**Q: How would you adapt this to corporate lending?** Replace the loan table with company financial ratios mapped to
agency ratings, use an ordinal/rating-migration model, and monitor rating-grade stability (migration matrices) and
default rates per grade with the same PSI/Gini/calibration engine. Corporate defaults are few (low-default portfolios), so
tests are weaker and expert judgement more important.
**Q: What are the limits of this project?** Public consumer data, a simple model, no champion/challenger decision rule
in production, Airflow run natively with `dags test` rather than a deployed scheduler, no Docker.
**Q: How do you avoid data leakage?** Only origination-time variables; the label comes from the final loan status; the
out-of-time window sits after the training window; score bands are frozen on development data.
**Q: How is it idempotent?** Output paths are keyed by month and overwritten atomically; heavy one-off steps are skipped
when outputs exist; the DAG has `max_active_runs=1` so backfills do not race.
**Q: What would you do with the red status?** Investigate drivers (CSI: interest rate), recalibrate the PD to recent default
rates (intercept shift), consider redevelopment on newer data, and document the decision in the monitoring committee pack.
