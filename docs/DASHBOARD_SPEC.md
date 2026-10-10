# Tableau dashboard specification (Tableau Public, ~30–45 minutes)

Step-by-step build guide for the monitoring dashboard, written for a first-time
Tableau user. Everything comes from the CSV extracts that
`python -m reporting.extracts` writes to `data/tableau/`.

Owner: builder #4. Contract: `docs/CONTRACTS.md` section 6. The monitoring
numbers themselves are produced by `monitoring.run`; this guide only *renders*
them, including the RAG status columns Python already computed.

## 0. One-time setup

1. Install **Tableau Public (Desktop)** (free) from <https://public.tableau.com/app/download>
   and create a free Tableau Public account (needed to save/publish).
2. Generate the extracts:
   ```bash
   make monitor      # produces data/monitoring/results/month=*.parquet (all months)
   make extracts     # -> data/tableau/*.csv
   # quick try without the full pipeline: the reporting tests build synthetic data in a temp dir
   ```
3. You now have the files below in `data/tableau/`. Keep `data_dictionary.md` open
   for column meanings.

| File | Grain (one row per…) | Used for |
| --- | --- | --- |
| `monthly_summary.csv` | monitoring month | KPI tiles, PSI/CSI trend, Gini/KS trend |
| `monitoring_results.csv` | month × metric × segment | long-format fallback for any custom view |
| `score_distribution.csv` | month × score band | population-drift view |
| `csi_by_feature.csv` | month × feature | CSI heat-map |
| `calibration_by_band.csv` | month × band | calibration bar chart |
| `breaches.csv` | amber/red metric row | breach list |

**Vocabulary:** a *dimension* is a group-by field (blue pills: `month`,
`feature`); a *measure* is a number (green pills: `psi_score`, `csi`). A *sheet*
is one chart; a *dashboard* is a page of sheets.

## 1. Connect to the data

1. Tableau Public → **Connect → To a File → Text file** → `data/tableau/monthly_summary.csv`.
2. Click **Sheet 1** to begin.
3. **New Data Source → Text file** for each of `score_distribution.csv`,
   `csi_by_feature.csv`, `calibration_by_band.csv`. Switch data source per sheet
   with the selector on the sheet.
4. Ensure `month` is a **String** (blue `Abc`), so it sorts `2014-01, 2014-02, …`.
   (`month_date` is a real ISO date if you prefer a continuous axis.)

Do **not** relate the files to one another; one file per sheet keeps the grain clear.

## 2. Calculated fields (only where needed)

The extracts already carry `overall_rag` and `rag_*` columns, so **colour by
those** (Step 3). These helpers are for charts that need a value rule.

**Is Latest Month?**
```
[month] = { MAX([month]) }
```

**Gini Status** (lower is worse; amber ≤ 0.40, red ≤ 0.30)
```
IF [gini] <= 0.30 THEN "RED"
ELSEIF [gini] <= 0.40 THEN "AMBER"
ELSE "GREEN" END
```

**KS Status** (amber ≤ 0.25, red ≤ 0.20)
```
IF [ks] <= 0.20 THEN "RED"
ELSEIF [ks] <= 0.25 THEN "AMBER"
ELSE "GREEN" END
```

**Calibration Deviation** (the extract stores the ratio; thresholds are on the deviation)
```
ABS([calibration_ratio] - 1)
```

## 3. RAG colour palette

Dragging any `rag_*`/`overall_rag` field to **Colour → Edit Colours…**, set:
`GREEN = #1B8A4B`, `AMBER = #D99100`, `RED = #C0392B`, `NA = #6B7280`. Save the
palette so every sheet matches. (These mirror `reporting.rag.RAG_COLORS`, so the
HTML report and the dashboard use identical colours.)

## 4. Sheet 1 — KPI tiles (latest month)

1. `Is Latest Month?` → **Filters** → True.
2. `Measure Names` → **Columns**; `Measure Values` → **Text** and **Label**; add
   `psi_score`, `csi_max`, `gini`, `ks`, `calibration_ratio` to Measure Values
   (remove the automatic `Number of Records`).
3. `overall_rag` → **Colour**.
4. Right-click a measure → **Format** → 3 decimals. Name the sheet **KPI tiles**.

## 5. Sheet 2 — PSI / CSI trend with RAG bands

1. Data source `monthly_summary.csv`. `month` → **Columns** (Discrete).
2. `Measure Names` → **Colour**, `Measure Values` → **Rows**; keep only
   `psi_score` and `csi_max`.
3. **Analytics → Reference Line → Constant**: `0.10` (amber), `0.25` (red). For
   shaded zones use **Reference Band** `0→0.10`, `0.10→0.25`, `0.25→TOP`.
4. Right-click axis → **Edit Axis → Fixed** 0 … 0.35.
5. Name it **PSI-CSI trend**.

## 6. Sheet 3 — Gini / KS discrimination trend

1. `month` → Columns. `gini` and `ks` → Rows (dual axis: drag `ks` to the right
   edge, right-click → **Dual Axis**, **Synchronise Axis**).
2. Add reference lines from the `config/thresholds.yaml` floors: Gini `0.40`
   (amber) / `0.30` (red), KS `0.25` / `0.20`.
3. Colour the Gini line by `rag_gini` and the KS line by `rag_ks`.
4. Name it **Discrimination trend**.

## 7. Sheet 4 — Calibration by score band

1. Data source `calibration_by_band.csv`. `band` → **Columns** (band 1 = safest).
2. `calibration_ratio` → **Rows** (bars). Add a constant reference line at `1.0`
   (perfect calibration), plus lines at `1.15`/`0.85` (amber) and `1.30`/`0.70` (red).
3. `rag_calibration_ratio` → **Colour**.
4. Filter `month` to the latest month (or drive it from a dashboard filter).
5. Name it **Calibration by band**.

## 8. Sheet 5 — Feature CSI heat-map

1. Data source `csi_by_feature.csv`. `feature` → **Rows**, `month` → **Columns**,
   `csi` → **Colour**, mark type **Square**.
2. **Colour → Edit Colours…**: stepped custom palette `#7FBF8F → #E0B84C → #C0392B`,
   **Advanced → Start 0, End 0.25** so it matches the PSI/CSI bands.
3. Optionally drag `csi` → **Label**. Name it **Feature CSI heat-map**.

## 9. Sheet 6 — Population drift by score band (optional)

1. Data source `score_distribution.csv`. `band` → Columns; `share` and `dev_share`
   → Rows (bars + line via Measure Names/Measure Values, or dual axis).
2. Colour by `share_delta` (diverging) to show which bands gained population.

## 10. Assemble the dashboard

1. **New Dashboard**; size Automatic or Fixed 1200×900.
2. Add a title text box, then place: KPI tiles (top), PSI-CSI trend and
   Discrimination trend (side by side), Calibration and CSI heat-map (below),
   breach list / population drift as needed.
3. Drag `month_date` or `month` to **Filters** and set **Apply to All Using This
   Data Source** so one month selector drives every sheet.
4. Add a colour legend for RAG and a text box stating the rules:
   > PSI/CSI: green &lt; 0.10, amber 0.10–0.25, red ≥ 0.25. Gini/KS: lower-is-worse
   > floors (0.40/0.30 and 0.25/0.20). Calibration ratio: green within ±0.15 of 1.0,
   > red beyond ±0.30. Overall = worst metric.

## 11. Save and publish

**File → Save to Tableau Public As…** → publish → copy the public URL into the
portfolio/CV next to the project.

## 12. Checklist

- [ ] Connected to `monthly_summary.csv` (+ the other three files).
- [ ] `month` kept as String so months order correctly.
- [ ] RAG palette `#1B8A4B` / `#D99100` / `#C0392B` / `#6B7280`.
- [ ] PSI/CSI trend shows 0.10 and 0.25 reference lines/bands.
- [ ] Gini/KS trend shows the 0.40/0.30 and 0.25/0.20 floors.
- [ ] Calibration shows bars + the 1.0 target line, coloured by `rag_calibration_ratio`.
- [ ] CSI heat-map uses a stepped 0→0.25 scale.
- [ ] A month filter is applied to **all** sheets.
- [ ] Published to Tableau Public; URL saved.

## 13. Troubleshooting

- **Month out of order / shown as a date** — change `month` data type to String,
  or switch to `month_date`.
- **Numbers look summed across months** — an aggregation is summing; put `month`
  on the view so each mark is one month, or set the measure to Average/Attribute.
- **Bands sort Z→A** — click the `band` pill → **Sort → Ascending**.
- **Colours differ between sheets** — you edited one sheet only; reuse the saved
  palette, or colour by the same `rag_*` field everywhere.
