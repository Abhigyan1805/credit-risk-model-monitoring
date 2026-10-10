"""Build the Tableau extracts from the long monitoring-results table.

Each builder returns a tidy pandas DataFrame with a stable column order. The
long table carries one row per ``(month, metric, segment)``; the wide summary
carries one row per month. Keeping "one fact per row" is what makes the files
drop straight into Tableau without reshaping.
"""

from __future__ import annotations

import pandas as pd

from reporting.data import load_dev_band_shares
from reporting.rag import worst_rag

#: Scalar metrics that live on the ``ALL`` segment and become wide columns.
SCALAR_METRICS: tuple[str, ...] = (
    "volume",
    "mean_pd",
    "default_rate",
    "psi_score",
    "auc",
    "gini",
    "ks",
    "calibration_ratio",
    "hl_pvalue",
)

LONG_COLUMNS: list[str] = [
    "month",
    "month_date",
    "metric",
    "segment",
    "value",
    "amber_threshold",
    "red_threshold",
    "direction",
    "rag",
    "n_obs",
    "model_version",
]

SUMMARY_COLUMNS: list[str] = [
    "month",
    "month_date",
    "volume",
    "mean_pd",
    "default_rate",
    "psi_score",
    "csi_max",
    "auc",
    "gini",
    "ks",
    "calibration_ratio",
    "hl_pvalue",
    "overall_rag",
    "rag_psi_score",
    "rag_csi_max",
    "rag_auc",
    "rag_gini",
    "rag_ks",
    "rag_calibration_ratio",
    "rag_hl_pvalue",
]

SCORE_DIST_COLUMNS: list[str] = [
    "month",
    "month_date",
    "score_band",
    "count",
    "share",
    "dev_share",
    "share_delta",
]

CSI_COLUMNS: list[str] = [
    "month",
    "month_date",
    "feature",
    "csi",
    "amber_threshold",
    "red_threshold",
    "rag",
]

CALIBRATION_COLUMNS: list[str] = [
    "month",
    "month_date",
    "band",
    "calibration_ratio",
    "rag_calibration_ratio",
    "binomial_pvalue",
    "rag_binomial_pvalue",
]

BREACH_COLUMNS: list[str] = [
    "month",
    "month_date",
    "metric",
    "segment",
    "value",
    "amber_threshold",
    "red_threshold",
    "direction",
    "rag",
    "message",
]

_OVERALL_VALUE_TO_RAG = {0: "GREEN", 1: "AMBER", 2: "RED"}


def build_long_table(results: pd.DataFrame) -> pd.DataFrame:
    """The monitoring long table plus the ISO ``month_date`` column."""
    return (
        results.loc[:, LONG_COLUMNS].sort_values(["month_date", "metric", "segment"]).reset_index(drop=True)
    )


def build_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    """One row per month with every scalar metric and its RAG status."""
    all_rows = results[results["segment"] == "ALL"]
    value_wide = all_rows.pivot_table(index="month", columns="metric", values="value", aggfunc="first")
    rag_wide = all_rows.pivot_table(index="month", columns="metric", values="rag", aggfunc="first")

    csi_max = results[results["metric"] == "csi"].groupby("month")["value"].max().rename("csi_max")
    csi_rag = (
        results[results["metric"] == "csi"]
        .groupby("month")["rag"]
        .apply(lambda values: worst_rag(list(values)))
        .rename("rag_csi_max")
    )

    months = sorted(results["month"].unique())
    summary = pd.DataFrame(index=pd.Index(months, name="month"))

    for metric in SCALAR_METRICS:
        summary[metric] = value_wide[metric] if metric in value_wide.columns else pd.NA
    summary["csi_max"] = csi_max.reindex(summary.index)

    # Headline status: map the producer's overall_rag code, else worst of the rags.
    overall = all_rows[all_rows["metric"] == "overall_rag"].set_index("month")["value"]
    mapped = overall.map(lambda v: _OVERALL_VALUE_TO_RAG.get(int(v), "NA") if pd.notna(v) else "NA")
    fallback = summary.index.to_series().map(lambda m: worst_rag(_rag_row(rag_wide, m, SCALAR_METRICS)))
    summary["overall_rag"] = [mapped.get(m, fallback.get(m, "NA")) for m in summary.index]

    for metric in SCALAR_METRICS:
        summary[f"rag_{metric}"] = [
            rag_wide.at[m, metric] if metric in rag_wide.columns and m in rag_wide.index else "NA"
            for m in summary.index
        ]
    summary["rag_csi_max"] = [csi_rag.get(m, "NA") for m in summary.index]

    summary = summary.reset_index()
    summary["month_date"] = summary["month"] + "-01"
    return summary.loc[:, SUMMARY_COLUMNS].sort_values("month_date").reset_index(drop=True)


def _rag_row(rag_wide: pd.DataFrame, month: str, metrics: tuple[str, ...]) -> list[str]:
    if month not in rag_wide.index:
        return []
    return [
        rag_wide.at[month, m] for m in metrics if m in rag_wide.columns and pd.notna(rag_wide.at[month, m])
    ]


def build_score_distribution(bands: pd.DataFrame, dev_shares: pd.Series | None = None) -> pd.DataFrame:
    """Population share of each score band per month, versus the dev reference."""
    if bands.empty:
        return pd.DataFrame(columns=SCORE_DIST_COLUMNS)
    counts = bands.groupby(["issue_month", "score_band"]).size().rename("count").reset_index()
    totals = counts.groupby("issue_month")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    counts = counts.rename(columns={"issue_month": "month"})
    if dev_shares is not None:
        counts["dev_share"] = counts["score_band"].map(dev_shares)
    else:
        counts["dev_share"] = pd.NA
    counts["share_delta"] = counts["share"] - counts["dev_share"]
    counts["month_date"] = counts["month"] + "-01"
    return counts.loc[:, SCORE_DIST_COLUMNS].sort_values(["month_date", "score_band"]).reset_index(drop=True)


def build_csi_by_feature(results: pd.DataFrame) -> pd.DataFrame:
    """Per-feature Characteristic Stability Index per month."""
    csi = results[results["metric"] == "csi"].copy()
    if csi.empty:
        return pd.DataFrame(columns=CSI_COLUMNS)
    csi = csi.rename(columns={"segment": "feature", "value": "csi"})
    return csi.loc[:, CSI_COLUMNS].sort_values(["month_date", "feature"]).reset_index(drop=True)


def build_calibration_by_band(results: pd.DataFrame) -> pd.DataFrame:
    """Predicted-to-observed calibration ratio and binomial p-value per band."""
    band_rows = results[results["metric"].isin(["calibration_ratio", "binomial_pvalue"])].copy()
    band_rows = band_rows[band_rows["segment"].str.startswith("band=")]
    if band_rows.empty:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    band_rows["band"] = band_rows["segment"].str.removeprefix("band=").astype(int)
    values = band_rows.pivot_table(index=["month", "band"], columns="metric", values="value", aggfunc="first")
    rags = band_rows.pivot_table(index=["month", "band"], columns="metric", values="rag", aggfunc="first")
    out = pd.DataFrame(index=values.index)
    out["calibration_ratio"] = values.get("calibration_ratio")
    out["rag_calibration_ratio"] = rags.get("calibration_ratio")
    out["binomial_pvalue"] = values.get("binomial_pvalue")
    out["rag_binomial_pvalue"] = rags.get("binomial_pvalue")
    out = out.reset_index()
    out["month_date"] = out["month"] + "-01"
    return out.loc[:, CALIBRATION_COLUMNS].sort_values(["month_date", "band"]).reset_index(drop=True)


def build_breaches(results: pd.DataFrame) -> pd.DataFrame:
    """Every amber/red metric row, with a plain-English message."""
    breaches = results[results["rag"].isin(["AMBER", "RED"])].copy()
    if breaches.empty:
        return pd.DataFrame(columns=BREACH_COLUMNS)
    breaches["message"] = [_breach_message(r) for r in breaches.itertuples()]
    return (
        breaches.loc[:, BREACH_COLUMNS]
        .sort_values(["month_date", "metric", "segment"])
        .reset_index(drop=True)
    )


def _breach_message(row) -> str:
    metric = row.metric
    label = metric.replace("_", " ")
    if metric == "overall_rag":
        return f"Overall model health is {row.rag} in {row.month}."
    where = "" if row.segment == "ALL" else f" ({row.segment})"
    return (
        f"{label}{where} = {row.value:.4g} is {row.rag} "
        f"(thresholds amber {row.amber_threshold}, red {row.red_threshold})."
    )


def build_extracts(results: pd.DataFrame, bands: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Build every Tableau extract as a DataFrame keyed by output name."""
    if bands is None:
        bands = pd.DataFrame(columns=["issue_month", "score_band"])
    return {
        "monitoring_results": build_long_table(results),
        "monthly_summary": build_monthly_summary(results),
        "score_distribution": build_score_distribution(bands, load_dev_band_shares()),
        "csi_by_feature": build_csi_by_feature(results),
        "calibration_by_band": build_calibration_by_band(results),
        "breaches": build_breaches(results),
    }
