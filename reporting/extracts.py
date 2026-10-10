"""Write Tableau-ready CSV extracts plus a data dictionary.

The outputs of :func:`write_extracts` are what the reader connects to from
Tableau Public (or any BI tool):

* ``monitoring_fact.csv`` — one row per (period) with all headline metrics and
  their RAG status. The spine of the dashboard.
* ``score_distribution.csv`` — one row per (period, score band).
* ``calibration_by_band.csv`` — one row per (period, score band).
* ``feature_csi.csv`` — one row per (period, feature).
* ``breaches.csv`` — one row per amber/red metric breach.
* ``data_dictionary.csv`` / ``data_dictionary.md`` — column-level documentation.

All files are UTF-8, comma-separated, with a header row.
"""

from __future__ import annotations

import csv
import os
from typing import Any

from reporting.schema import MonitoringResults
from reporting.tables import build_all_tables

#: The four "named" extracts called out in the task, mapped to output filename.
EXTRACT_FILES = {
    "fact": "monitoring_fact.csv",
    "score_distribution": "score_distribution.csv",
    "calibration_by_band": "calibration_by_band.csv",
    "feature_csi": "feature_csi.csv",
    "breaches": "breaches.csv",
}

#: Column-level documentation. ``dtype`` and ``description`` are surfaced in
#: the data dictionary and repeated in docs/reporting/TABLEAU_DASHBOARD.md.
DATA_DICTIONARY: dict[str, dict[str, tuple[str, str]]] = {
    "fact": {
        "period": ("string", "Monitoring month, format YYYY-MM."),
        "n_obs": ("integer", "Number of scored accounts/loans observed in the period."),
        "default_rate": ("decimal", "Observed default rate in the period (0-1)."),
        "psi": ("decimal", "Population Stability Index versus the reference period."),
        "csi": ("decimal", "Largest Characteristic Stability Index across features."),
        "auc": ("decimal", "Area under the ROC curve, a discrimination measure."),
        "gini": ("decimal", "Gini = 2*AUC - 1."),
        "ks": ("decimal", "Kolmogorov-Smirnov statistic, a discrimination measure."),
        "calibration_gap": ("decimal", "Mean absolute |predicted PD - observed default rate|."),
        "rag_overall": ("string", "Worst RAG status across monitored metrics: green/amber/red."),
        "rag_psi": ("string", "RAG status for PSI."),
        "rag_csi": ("string", "RAG status for CSI."),
        "rag_auc": ("string", "RAG status for AUC."),
        "rag_gini": ("string", "RAG status for Gini."),
        "rag_ks": ("string", "RAG status for KS."),
        "rag_calibration": ("string", "RAG status for the calibration gap."),
    },
    "score_distribution": {
        "period": ("string", "Monitoring month, format YYYY-MM."),
        "band": ("string", "Score decile/band label, e.g. 'A' (lowest risk) to 'J'."),
        "score_min": ("decimal", "Lower bound of the PD score band."),
        "score_max": ("decimal", "Upper bound of the PD score band."),
        "count": ("integer", "Accounts/loans scored into the band."),
        "share": ("decimal", "Share of the period population in the band (0-1)."),
        "expected_share": ("decimal", "Reference-period share for the band (0-1)."),
        "share_delta": ("decimal", "share minus expected_share; population drift per band."),
    },
    "calibration_by_band": {
        "period": ("string", "Monitoring month, format YYYY-MM."),
        "band": ("string", "Score decile/band label."),
        "count": ("integer", "Accounts/loans scored into the band."),
        "predicted_pd": ("decimal", "Mean model PD in the band (0-1)."),
        "observed_pd": ("decimal", "Observed default rate in the band (0-1)."),
        "abs_gap": ("decimal", "|observed_pd - predicted_pd|; calibration error for the band."),
        "rag_calibration": ("string", "RAG status for this band's calibration gap."),
    },
    "feature_csi": {
        "period": ("string", "Monitoring month, format YYYY-MM."),
        "feature": ("string", "Input feature name, e.g. 'annual_inc'."),
        "csi": ("decimal", "Characteristic Stability Index for the feature."),
        "rag": ("string", "RAG status for the feature CSI."),
    },
    "breaches": {
        "period": ("string", "Monitoring month the breach was recorded in."),
        "metric": ("string", "Metric that breached its threshold."),
        "value": ("decimal", "Observed metric value."),
        "reference": ("decimal", "Reference value, where applicable."),
        "level": ("string", "Breach severity: amber or red."),
        "message": ("string", "Plain-English description of the breach."),
    },
}


def write_csv(path: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one tidy table to ``path`` as UTF-8 CSV with a header row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _format_cell(row.get(col)) for col in columns})


def _format_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else value


def write_data_dictionary(out_dir: str) -> tuple[str, str]:
    """Write the data dictionary as both CSV and Markdown; return their paths."""
    csv_path = os.path.join(out_dir, "data_dictionary.csv")
    md_path = os.path.join(out_dir, "data_dictionary.md")
    os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["table", "column", "type", "description"])
        for table, columns in DATA_DICTIONARY.items():
            for column, (dtype, description) in columns.items():
                writer.writerow([table, column, dtype, description])

    lines = ["# Data dictionary", ""]
    for table, columns in DATA_DICTIONARY.items():
        lines.append(f"## `{table}`")
        lines.append("")
        lines.append("| Column | Type | Description |")
        lines.append("| --- | --- | --- |")
        for column, (dtype, description) in columns.items():
            lines.append(f"| `{column}` | {dtype} | {description} |")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return csv_path, md_path


def write_extracts(results: MonitoringResults, out_dir: str) -> dict[str, str]:
    """Write every CSV extract plus the data dictionary into ``out_dir``.

    Returns a mapping of logical table name to the written file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    tables = build_all_tables(results)
    written: dict[str, str] = {}
    for name, (columns, rows) in tables.items():
        path = os.path.join(out_dir, EXTRACT_FILES[name])
        write_csv(path, columns, rows)
        written[name] = path
    csv_path, md_path = write_data_dictionary(out_dir)
    written["data_dictionary_csv"] = csv_path
    written["data_dictionary_md"] = md_path
    return written
