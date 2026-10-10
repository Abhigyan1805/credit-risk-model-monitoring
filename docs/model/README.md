# PD model (model/)

## What it is, in plain English
A **probability-of-default (PD) model**: given a loan application, estimate the chance the borrower defaults
(LendingClub: "Charged Off"/"Default"). It is built the way banks build retail scorecards, because a scorecard
is explainable line by line, which regulators and model-risk teams require.

1. **Binning + WoE.** Each feature is cut into bins (numbers by quantile, categories by value, missing = its own bin).
   Each bin gets *weight of evidence* `WoE = ln(%goods / %bads)`: positive = safer than average, negative = riskier.
2. **Information value (IV)** = Σ (%good − %bad) × WoE ranks features. We keep IV ≥ 0.02; IV above 0.5 would be a leakage red flag.
3. **Logistic regression on WoE values** gives PD. We drop features highly correlated with a stronger one (|ρ| > 0.7) and any with a
   "wrong-sign" coefficient (all must be negative, since WoE is good-vs-bad), so every driver has a sensible direction.
4. **Points.** `score = 600 + (20/ln2) × ln(good:bad odds / 50)`: 600 points = 50:1 odds, every +20 points doubles the odds (PDO = 20).
   Each bin's points are in `scorecard_points.csv`; a loan's score is the sum, exactly equal to the logistic model.
5. **Score bands 1–10** (1 safest): fixed PD cut-points = the deciles of the development training PDs (as CONTRACTS.md requires). They are frozen
   so a band means the same thing every month; that is what makes month-by-month monitoring meaningful.
6. **Challenger**: gradient boosting (scikit-learn `HistGradientBoostingClassifier`) on raw features, for comparison only.

**Out-of-time validation (CONTRACTS.md windows):** train on loans issued 2012-01..2013-06, validate on 2013-07..2013-12 (later months
the model never saw). The OOT sample, scored, is `vintage=dev.parquet`, the reference population monitoring compares every later month against.
Monitoring then replays 2014-01..2016-12 month by month. Only loans with a final outcome (Fully Paid / Charged Off / Default) have a default
flag; Current/Late loans are NA, and vintages after 2015-12 are scored with `default_flag` blank because their resolved loans are biased to early defaults.

## Use (paths via `common/paths.py`, data root `CRM_DATA_DIR`)
```
python -m model.train                  # make train: fits, writes data/model/*, data/scored/vintage=dev.parquet
python -m model.score --month 2014-06  # make score loops over all 36 months -> data/scored/vintage=<month>.parquet
pytest model                           # unit tests: synthetic data + committed sample fixture, no download
```
`data/model/`: `pd_model.joblib`, `metadata.json` (version, windows, features, coefficients, band cut-points, train/OOT/challenger metrics),
`scorecard_points.csv` (the readable scorecard), `iv_table.csv`. Scored files follow SCORED_COLUMNS in `common/schemas.py`; the CSI
feature columns are copied raw from the prepared table whether or not the scorecard selected them.

## Design choices
- `grade` and `int_rate` are LendingClub's own risk pricing (another model's output). They are candidates because the contract monitors them
  (`CSI_FEATURES`), but they correlate strongly, so the correlation filter keeps only the stronger; `sub_grade` is not used. Be ready to say
  that a model leaning on grade partly copies LendingClub's underwriting, and that dropping both is a sensible sensitivity test.
- Only information available at origination is used (no payments, recoveries, FICO-last) to avoid leakage.
- Monotonic WoE, small-bin merging and sign checks trade a little AUC for explainability.
- Unseen categories fall to the `OTHER` bin; missing numerics to `MISSING`, so scoring never crashes on new data.

## Results
Real full-run numbers go here after `make e2e` (lead). **Sample-fixture run (4,200-loan stratified real sample, 1,260 train / 420 OOT loans; indicative only, not for the resume):**

| model | train AUC / Gini / KS | OOT AUC / Gini / KS |
|---|---|---|
| WoE scorecard | 0.716 / 0.431 / 0.352 | 0.629 / 0.257 / 0.256 |
| gradient-boosted challenger | 0.782 / 0.565 / 0.422 | 0.622 / 0.244 / 0.257 |

The challenger fits training better but is no better out of time: small-sample overfit, a typical argument for keeping the scorecard.

## Likely interview questions
**1. Why WoE + logistic regression instead of a gradient-boosted model?** Every bin has points, monotonic and signed sensibly, so a decline
can be explained and audited; it is also stable and easy to monitor. The GBM is kept as a challenger to show the AUC given up.
**2. What do IV, WoE, Gini and KS mean?** WoE is ln(%good/%bad) per bin; IV sums it to rank features; Gini = 2·AUC−1 (ranking power);
KS = max gap between cumulative bad and good score distributions.
**3. Why out-of-time rather than a random split?** Real use is predicting future loans; a random split mixes time periods and overstates
performance when the population drifts. OOT shows drop-off from drift, exactly what monitoring tracks.
**4. How are the points computed?** factor = PDO/ln2, offset = 600 − factor·ln(50); score = offset + factor·ln(good:bad odds); split across bins.
**5. How did you define default and handle unresolved loans?** Charged Off/Default = 1, Fully Paid = 0; Current/Late are censored (not observable)
so excluded from training and validation; the flag is filled in later when loans resolve, which is why monitoring discrimination lags.
**6. What if an IV were 0.8?** Suspect leakage (a variable containing the outcome) and investigate before use.
