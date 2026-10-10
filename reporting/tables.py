"""Build tidy, one-row-per-observation tables from normalized results.

Every table is returned as ``(columns, rows)`` where ``rows`` is a list of
plain dicts and ``columns`` fixes the column order for CSV/DataFrame
consumption. The tables are intentionally "tidy" (one fact per row, no merged
cells, no colour) because that is exactly what Tableau wants — colour is added
in Tableau itself from the ``rag`` columns, so the CSV stays analysis-ready.
"""

from __future__ import annotations

from typing import Any, Optional

from reporting.rag import MONITORED_METRICS, classify_metric, worst_status
from reporting.schema import MonitoringResults

FACT_COLUMNS = [
    "period",
    "n_obs",
    "default_rate",
    "psi",
    "csi",
    "auc",
    "gini",
    "ks",
    "calibration_gap",
    "rag_overall",
    "rag_psi",
    "rag_csi",
    "rag_auc",
    "rag_gini",
    "rag_ks",
    "rag_calibration",
]

SCORE_DIST_COLUMNS = [
    "period",
    "band",
    "score_min",
    "score_max",
    "count",
    "share",
    "expected_share",
    "share_delta",
]

CALIBRATION_COLUMNS = [
    "period",
    "band",
    "count",
    "predicted_pd",
    "observed_pd",
    "abs_gap",
    "rag_calibration",
]

FEATURE_CSI_COLUMNS = ["period", "feature", "csi", "rag"]

BREACH_COLUMNS = ["period", "metric", "value", "reference", "level", "message"]


#: Short RAG column names shown next to each metric in the fact table.
_RAG_COLUMN_NAMES = {
    "psi": "rag_psi",
    "csi": "rag_csi",
    "auc": "rag_auc",
    "gini": "rag_gini",
    "ks": "rag_ks",
    "calibration_gap": "rag_calibration",
}


def _rag_columns(metrics: dict[str, Optional[float]], reference: dict[str, float]) -> dict[str, Optional[str]]:
    return {
        _RAG_COLUMN_NAMES[m]: classify_metric(m, metrics.get(m), reference_value=reference.get(m))
        for m in MONITORED_METRICS
    }


def build_fact_table(results: MonitoringResults) -> tuple[list[str], list[dict[str, Any]]]:
    """One row per monitoring period with every metric and its RAG status."""
    reference = results.reference_metrics
    rows: list[dict[str, Any]] = []
    for period in results.periods_sorted:
        metrics = period.get("metrics", {})
        rag_cols = _rag_columns(metrics, reference)
        row: dict[str, Any] = {
            "period": period["period"],
            "n_obs": period.get("n_obs"),
            "default_rate": period.get("default_rate"),
            "psi": metrics.get("psi"),
            "csi": metrics.get("csi"),
            "auc": metrics.get("auc"),
            "gini": metrics.get("gini"),
            "ks": metrics.get("ks"),
            "calibration_gap": metrics.get("calibration_gap"),
            "rag_overall": period.get("rag") or worst_status(list(rag_cols.values())),
        }
        row.update(rag_cols)
        rows.append(row)
    return list(FACT_COLUMNS), rows


def build_score_distribution_table(
    results: MonitoringResults,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Score-band population shares over time, with the change vs expected."""
    rows: list[dict[str, Any]] = []
    for record in results.score_distribution:
        share = record.get("share")
        expected = record.get("expected_share")
        share_delta = record.get("share_delta")
        if share_delta is None and isinstance(share, (int, float)) and isinstance(
            expected, (int, float)
        ):
            share_delta = share - expected
        rows.append(
            {
                "period": record.get("period"),
                "band": record.get("band"),
                "score_min": record.get("score_min"),
                "score_max": record.get("score_max"),
                "count": record.get("count"),
                "share": share,
                "expected_share": expected,
                "share_delta": share_delta,
            }
        )
    rows.sort(key=lambda r: (str(r["period"]), str(r["band"])))
    return list(SCORE_DIST_COLUMNS), rows


def build_calibration_table(
    results: MonitoringResults,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Predicted vs observed default rate by score band, with the absolute gap."""
    rows: list[dict[str, Any]] = []
    for record in results.calibration_by_band:
        predicted = record.get("predicted_pd")
        observed = record.get("observed_pd")
        gap = record.get("abs_gap")
        if gap is None and isinstance(predicted, (int, float)) and isinstance(
            observed, (int, float)
        ):
            gap = abs(observed - predicted)
        rows.append(
            {
                "period": record.get("period"),
                "band": record.get("band"),
                "count": record.get("count"),
                "predicted_pd": predicted,
                "observed_pd": observed,
                "abs_gap": gap,
                "rag_calibration": classify_metric("calibration_gap", gap),
            }
        )
    rows.sort(key=lambda r: (str(r["period"]), str(r["band"])))
    return list(CALIBRATION_COLUMNS), rows


def build_feature_csi_table(
    results: MonitoringResults,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Characteristic Stability Index per input feature per period."""
    rows: list[dict[str, Any]] = []
    for record in results.feature_csi:
        csi = record.get("csi")
        rows.append(
            {
                "period": record.get("period"),
                "feature": record.get("feature"),
                "csi": csi,
                "rag": classify_metric("csi", csi),
            }
        )
    rows.sort(key=lambda r: (str(r["feature"]), str(r["period"])))
    return list(FEATURE_CSI_COLUMNS), rows


def build_breach_table(
    results: MonitoringResults,
) -> tuple[list[str], list[dict[str, Any]]]:
    """All amber/red metric breaches, oldest first."""
    rows: list[dict[str, Any]] = []
    for breach in results.breaches:
        rows.append({col: breach.get(col) for col in BREACH_COLUMNS})
    rows.sort(key=lambda r: (str(r["period"]), str(r["metric"]), str(r["level"])))
    return list(BREACH_COLUMNS), rows


def build_all_tables(results: MonitoringResults) -> dict[str, tuple[list[str], list[dict[str, Any]]]]:
    """Return every extract table keyed by its stable output name."""
    return {
        "fact": build_fact_table(results),
        "score_distribution": build_score_distribution_table(results),
        "calibration_by_band": build_calibration_table(results),
        "feature_csi": build_feature_csi_table(results),
        "breaches": build_breach_table(results),
    }
