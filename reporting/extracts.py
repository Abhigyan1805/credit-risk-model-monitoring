"""Write the Tableau-ready CSV extracts. ``python -m reporting.extracts``

Output: ``common.paths.tableau_dir()`` (``data/tableau/`` by default), UTF-8 CSV
with a header row and ISO ``YYYY-MM-01`` month dates (see ``docs/CONTRACTS.md``
section 6):

* ``monitoring_results.csv`` — the long monitoring table plus ``month_date``.
* ``monthly_summary.csv`` — one row per month, wide, with RAG statuses.
* ``score_distribution.csv`` — population share per score band per month.
* ``csi_by_feature.csv`` — Characteristic Stability Index per feature per month.
* ``calibration_by_band.csv`` — calibration ratio and binomial p-value per band.
* ``breaches.csv`` — every amber/red metric row (convenience for the breach list).
* ``data_dictionary.csv`` / ``data_dictionary.md`` — column-level documentation.

Owner: builder #4. All numbers come from the monitoring results and the scored
vintages; nothing here is hard-coded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from common import paths
from reporting.data import available_months, load_results, load_scored_bands
from reporting.tables import build_extracts

EXTRACT_FILES: dict[str, str] = {
    "monitoring_results": "monitoring_results.csv",
    "monthly_summary": "monthly_summary.csv",
    "score_distribution": "score_distribution.csv",
    "csi_by_feature": "csi_by_feature.csv",
    "calibration_by_band": "calibration_by_band.csv",
    "breaches": "breaches.csv",
}

#: Column-level documentation, surfaced as data_dictionary.{csv,md}.
DATA_DICTIONARY: dict[str, dict[str, tuple[str, str]]] = {
    "monitoring_results": {
        "month": ("string", "Monitoring month, YYYY-MM (the loan vintage)."),
        "month_date": ("date", "month as an ISO date, YYYY-MM-01, for Tableau date axes."),
        "metric": ("string", "Metric name, e.g. psi_score, csi, auc, calibration_ratio, overall_rag."),
        "segment": ("string", "ALL, a feature name (csi) or band=<k> (calibration)."),
        "value": ("decimal", "Metric value."),
        "amber_threshold": ("decimal", "Amber threshold from config/thresholds.yaml."),
        "red_threshold": ("decimal", "Red threshold from config/thresholds.yaml."),
        "direction": ("string", "higher_worse or lower_worse."),
        "rag": ("string", "GREEN / AMBER / RED / NA (NA = not computable)."),
        "n_obs": ("integer", "Accounts observed for the month/metric."),
        "model_version": ("string", "Frozen model version."),
    },
    "monthly_summary": {
        "month": ("string", "Monitoring month, YYYY-MM."),
        "month_date": ("date", "month as ISO date YYYY-MM-01."),
        "volume": ("integer", "Loans originated in the vintage."),
        "mean_pd": ("decimal", "Mean predicted probability of default."),
        "default_rate": ("decimal", "Observed default rate (NA when outcomes not observable)."),
        "psi_score": ("decimal", "PSI of the score distribution vs the dev reference."),
        "csi_max": ("decimal", "Largest feature CSI in the month."),
        "auc": ("decimal", "AUC (NA when outcomes not observable)."),
        "gini": ("decimal", "Gini = 2*AUC - 1."),
        "ks": ("decimal", "Kolmogorov-Smirnov statistic."),
        "calibration_ratio": ("decimal", "Observed default rate / mean PD (ALL segment)."),
        "hl_pvalue": ("decimal", "Hosmer-Lemeshow style calibration p-value."),
        "overall_rag": ("string", "Headline status: worst metric RAG for the month."),
        "rag_psi_score": ("string", "RAG status for psi_score."),
        "rag_csi_max": ("string", "RAG status for the largest feature CSI."),
        "rag_auc": ("string", "RAG status for auc."),
        "rag_gini": ("string", "RAG status for gini."),
        "rag_ks": ("string", "RAG status for ks."),
        "rag_calibration_ratio": ("string", "RAG status for calibration_ratio."),
        "rag_hl_pvalue": ("string", "RAG status for hl_pvalue."),
    },
    "score_distribution": {
        "month": ("string", "Monitoring month, YYYY-MM."),
        "month_date": ("date", "month as ISO date YYYY-MM-01."),
        "score_band": ("integer", "Score band 1 (safest) .. 10 (riskiest)."),
        "count": ("integer", "Loans in the band."),
        "share": ("decimal", "Band share of the month population (0-1)."),
        "dev_share": ("decimal", "Band share in the dev reference population."),
        "share_delta": ("decimal", "share - dev_share; population drift per band."),
    },
    "csi_by_feature": {
        "month": ("string", "Monitoring month, YYYY-MM."),
        "month_date": ("date", "month as ISO date YYYY-MM-01."),
        "feature": ("string", "Input feature name."),
        "csi": ("decimal", "Characteristic Stability Index for the feature."),
        "amber_threshold": ("decimal", "Amber threshold (0.10)."),
        "red_threshold": ("decimal", "Red threshold (0.25)."),
        "rag": ("string", "GREEN / AMBER / RED / NA."),
    },
    "calibration_by_band": {
        "month": ("string", "Monitoring month, YYYY-MM."),
        "month_date": ("date", "month as ISO date YYYY-MM-01."),
        "band": ("integer", "Score band 1 (safest) .. 10 (riskiest)."),
        "calibration_ratio": ("decimal", "Observed default rate / mean PD for the band."),
        "rag_calibration_ratio": ("string", "RAG status for the calibration ratio."),
        "binomial_pvalue": ("decimal", "Per-band binomial test p-value of PD vs observed."),
        "rag_binomial_pvalue": ("string", "RAG status for the binomial p-value."),
    },
    "breaches": {
        "month": ("string", "Monitoring month, YYYY-MM."),
        "month_date": ("date", "month as ISO date YYYY-MM-01."),
        "metric": ("string", "Metric that breached a threshold."),
        "segment": ("string", "ALL, feature name, or band=<k>."),
        "value": ("decimal", "Observed metric value."),
        "amber_threshold": ("decimal", "Amber threshold."),
        "red_threshold": ("decimal", "Red threshold."),
        "direction": ("string", "higher_worse or lower_worse."),
        "rag": ("string", "AMBER or RED."),
        "message": ("string", "Plain-English description of the breach."),
    },
}


def write_dataframe_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as UTF-8 CSV with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
    return path


def write_data_dictionary(out_dir: Path) -> tuple[Path, Path]:
    """Write the data dictionary as CSV and Markdown; return both paths."""
    csv_path = out_dir / "data_dictionary.csv"
    md_path = out_dir / "data_dictionary.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "column", "type", "description"])
        for table, columns in DATA_DICTIONARY.items():
            for column, (dtype, description) in columns.items():
                writer.writerow([table, column, dtype, description])

    lines = ["# Data dictionary", "", "Extracts in `data/tableau/` (see `docs/CONTRACTS.md` section 6).", ""]
    for table, columns in DATA_DICTIONARY.items():
        lines += [f"## `{table}.csv`", "", "| Column | Type | Description |", "| --- | --- | --- |"]
        lines += [f"| `{c}` | {t} | {d} |" for c, (t, d) in columns.items()]
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def write_extracts(
    results: pd.DataFrame,
    bands: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Build and write every extract; return logical name -> path."""
    out_dir = out_dir or paths.tableau_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in build_extracts(results, bands).items():
        written[name] = write_dataframe_csv(df, out_dir / EXTRACT_FILES[name])
    csv_path, md_path = write_data_dictionary(out_dir)
    written["data_dictionary_csv"] = csv_path
    written["data_dictionary_md"] = md_path
    return written


def run(up_to: str | None = None) -> dict[str, Path]:
    """Load all available monitoring results and write the extracts."""
    months = available_months(up_to)
    results = load_results(months)
    bands = load_scored_bands(months)
    return write_extracts(results, bands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reporting.extracts", description="Write Tableau-ready CSV extracts."
    )
    parser.add_argument("--up-to", help="Only include months at or before this YYYY-MM.")
    parser.add_argument("--out-dir", help=f"Output directory (default {paths.tableau_dir()}).")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    months = available_months(args.up_to)
    results = load_results(months)
    bands = load_scored_bands(months)
    out_dir = Path(args.out_dir) if args.out_dir else paths.tableau_dir()
    written = write_extracts(results, bands, out_dir)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
