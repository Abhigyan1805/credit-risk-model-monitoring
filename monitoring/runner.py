"""Run the monthly monitoring job and emit the monitoring-results tables.

The runner ties the metrics and the threshold rules together:

1. build the *development baseline* from the model's training/out-of-time
   sample -- it fixes the score and feature bins and the baseline Gini/KS;
2. for each month, score the seasoned loans, compute stability,
   discrimination and calibration metrics against that frozen baseline;
3. apply the YAML traffic-light thresholds and roll the bands up into one
   red/amber/green verdict per month.

Outputs
-------
``monitoring_results``
    Tidy long table, one row per (month, metric) pair -- the table a dashboard
    or a monitoring report reads directly.
``monthly_summary``
    Wide table, one row per month with the headline numbers and overall RAG.
``calibration_detail``
    One row per (month, score band) with predicted vs observed default rates.
``alert_summary``
    One row per month listing which metrics breached (red/amber) and the
    rolled-up RAG verdict -- the "alert summary per month".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from monitoring import alerts
from monitoring.calibration import calibration_bands, hosmer_lemeshow
from monitoring.config import ThresholdConfig, load_config
from monitoring.discrimination import auc, gini, ks
from monitoring.stability import (
    csi,
    csi_categorical,
    psi,
    quantile_bin_edges,
)

LONG_COLUMNS = [
    "month",
    "metric_group",
    "metric",
    "feature",
    "value",
    "baseline_value",
    "band",
    "alert",
    "threshold_ref",
    "details",
]


@dataclass
class ModelSnapshot:
    """Model outputs for one sample (the baseline) or one month.

    Attributes
    ----------
    scores:
        Predicted probabilities / scores, one per loan.
    labels:
        Observed default flag (1 = default). Required on the baseline to
        establish baseline discrimination; optional for months that are not
        yet seasoned (discrimination is then skipped).
    features:
        Mapping of feature name -> array of values aligned with ``scores``.
    """

    scores: np.ndarray
    labels: np.ndarray | None = None
    features: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=float)
        if self.labels is not None:
            self.labels = np.asarray(self.labels)
        self.features = {k: np.asarray(v) for k, v in self.features.items()}

    def __len__(self) -> int:
        return int(self.scores.shape[0])


@dataclass
class Baseline:
    """Frozen development baseline used to compare every month against."""

    snapshot: ModelSnapshot
    score_edges: np.ndarray
    numeric_feature_edges: dict[str, np.ndarray]
    feature_kinds: dict[str, str]
    gini: float
    ks: float
    auc: float


def _feature_kind(values: np.ndarray, declared: str | None) -> str:
    if declared in ("numeric", "categorical"):
        return declared
    return "numeric" if np.issubdtype(values.dtype, np.number) else "categorical"


def build_baseline(
    snapshot: ModelSnapshot,
    config: ThresholdConfig,
    feature_types: dict[str, str] | None = None,
) -> Baseline:
    """Freeze bins and baseline discrimination metrics from the dev sample.

    Bins are computed once from the baseline and reused every month so a
    month's PSI/CSI measures movement *relative to development*, not to the
    previous month.
    """
    if snapshot.labels is None:
        raise ValueError("baseline snapshot must include observed labels")
    score_edges = quantile_bin_edges(snapshot.scores, n_bins=config.psi["n_bins"])

    feature_kinds: dict[str, str] = {}
    numeric_edges: dict[str, np.ndarray] = {}
    declared = feature_types or {}
    for name, values in snapshot.features.items():
        kind = _feature_kind(values, declared.get(name))
        feature_kinds[name] = kind
        if kind == "numeric":
            try:
                numeric_edges[name] = quantile_bin_edges(
                    np.asarray(values, dtype=float), n_bins=config.csi["n_bins"]
                )
            except ValueError:
                feature_kinds[name] = "categorical"

    return Baseline(
        snapshot=snapshot,
        score_edges=score_edges,
        numeric_feature_edges=numeric_edges,
        feature_kinds=feature_kinds,
        gini=gini(snapshot.labels, snapshot.scores),
        ks=ks(snapshot.labels, snapshot.scores),
        auc=auc(snapshot.labels, snapshot.scores),
    )


def _row(
    month: str,
    group: str,
    metric: str,
    value: float | None,
    baseline_value: float | None,
    band: str,
    threshold_ref: str,
    feature: str | None = None,
    details: str = "",
) -> dict[str, Any]:
    return {
        "month": month,
        "metric_group": group,
        "metric": metric,
        "feature": feature,
        "value": value,
        "baseline_value": baseline_value,
        "band": band,
        "alert": band == "red",
        "threshold_ref": threshold_ref,
        "details": details,
    }


def _monitor_month(
    month: str,
    snapshot: ModelSnapshot,
    baseline: Baseline,
    config: ThresholdConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame | None]:
    """Compute every metric for one month, returning long rows + wide summary."""
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"month": month, "n_loans": len(snapshot)}

    # --- stability -----------------------------------------------------
    psi_value = psi(
        baseline.snapshot.scores, snapshot.scores, bin_edges=baseline.score_edges
    )
    psi_band = alerts.band_psi(psi_value, config)
    rows.append(
        _row(
            month,
            "stability",
            "psi_score",
            psi_value,
            0.0,
            psi_band,
            "psi.green_below/amber_below",
            details="score distribution vs baseline",
        )
    )
    summary["psi_score"] = psi_value
    summary["psi_band"] = psi_band

    csi_values: dict[str, float] = {}
    for name in baseline.snapshot.features:
        base_feat = baseline.snapshot.features[name]
        curr_feat = snapshot.features.get(name)
        if curr_feat is None:
            continue
        kind = baseline.feature_kinds.get(name, "numeric")
        if kind == "numeric":
            value = csi(
                base_feat,
                curr_feat,
                bin_edges=baseline.numeric_feature_edges.get(name),
            )
        else:
            value = csi_categorical(base_feat, curr_feat)
        band = alerts.band_csi(value, config)
        csi_values[name] = value
        rows.append(
            _row(
                month,
                "stability",
                "csi",
                value,
                0.0,
                band,
                "csi.green_below/amber_below",
                feature=name,
                details=f"{kind} characteristic",
            )
        )
    if csi_values:
        worst_feature = max(csi_values, key=csi_values.get)
        summary["csi_max"] = csi_values[worst_feature]
        summary["csi_max_feature"] = worst_feature
        summary["csi_band"] = alerts.band_csi(csi_values[worst_feature], config)
    bands_so_far = [r["band"] for r in rows]

    # --- discrimination (seasoned months only) -------------------------
    if snapshot.labels is not None and len(np.unique(snapshot.labels)) == 2:
        m_auc = auc(snapshot.labels, snapshot.scores)
        m_gini = gini(snapshot.labels, snapshot.scores)
        m_ks = ks(snapshot.labels, snapshot.scores)
        gini_band = alerts.band_gini_drop(baseline.gini, m_gini, config)
        ks_band = alerts.band_ks_drop(baseline.ks, m_ks, config)
        rows.extend(
            [
                _row(month, "discrimination", "auc", m_auc, baseline.auc, "na",
                     "reference only", details="area under ROC"),
                _row(month, "discrimination", "gini", m_gini, baseline.gini, gini_band,
                     "discrimination.gini", details="2*AUC-1"),
                _row(month, "discrimination", "ks", m_ks, baseline.ks, ks_band,
                     "discrimination.ks", details="max CDF gap"),
            ]
        )
        summary.update(
            {
                "auc": m_auc,
                "gini": m_gini,
                "gini_rel_drop": (baseline.gini - m_gini) / baseline.gini
                if baseline.gini > 0
                else np.nan,
                "gini_band": gini_band,
                "ks": m_ks,
                "ks_abs_drop": baseline.ks - m_ks,
                "ks_band": ks_band,
                "calibration_available": True,
            }
        )
        bands_so_far += [gini_band, ks_band]
    else:
        summary["calibration_available"] = False

    # --- calibration ---------------------------------------------------
    calib_detail = None
    if summary.get("calibration_available"):
        hl = hosmer_lemeshow(
            snapshot.labels, snapshot.scores, n_bands=config.calibration["n_bands"]
        )
        hl_band = alerts.band_hl_pvalue(hl["p_value"], config)
        rows.append(
            _row(
                month,
                "calibration",
                "hl_pvalue",
                hl["p_value"],
                None,
                hl_band,
                "calibration.hl",
                details=f"HL={hl['statistic']:.2f}, dof={hl['dof']}",
            )
        )
        bands = calibration_bands(
            snapshot.labels, snapshot.scores, n_bands=config.calibration["n_bands"]
        )
        bands = bands.copy()
        bands.insert(0, "month", month)
        bands["band_rag"] = [
            alerts.band_calibration_band(
                row.deviation, row.binomial_pvalue, row.n, config
            )
            for row in bands.itertuples()
        ]
        calib_detail = bands
        band_rags = [b for b in bands["band_rag"] if b != "na"]
        worst_band_rag = alerts.overall_rag(band_rags) if band_rags else "green"
        rows.append(
            _row(
                month,
                "calibration",
                "worst_band",
                float(bands["deviation"].abs().max()),
                0.0,
                worst_band_rag,
                "calibration.binomial",
                details="largest |observed-expected| default-rate gap",
            )
        )
        bands_so_far.append(hl_band)
        bands_so_far.append(worst_band_rag)
        summary.update(
            {
                "hl_pvalue": hl["p_value"],
                "hl_statistic": hl["statistic"],
                "hl_band": hl_band,
                "max_band_deviation": float(bands["deviation"].abs().max()),
                "worst_band_band": worst_band_rag,
            }
        )

    summary["overall_rag"] = alerts.overall_rag(bands_so_far)
    return rows, summary, calib_detail


def _metric_label(metric: str, feature: Any) -> str:
    if isinstance(feature, str) and feature:
        return f"{metric}[{feature}]"
    return metric


def build_alert_summary(monitoring_results: pd.DataFrame) -> pd.DataFrame:
    """Per-month alert summary: which metrics breached and the overall RAG.

    Turns the long results table into the one-line-per-month view a monitoring
    report or an email digest leads with.
    """
    rows: list[dict[str, Any]] = []
    for month, group in monitoring_results.groupby("month", sort=False):
        reds = group[group["band"] == "red"]
        ambers = group[group["band"] == "amber"]
        rows.append(
            {
                "month": month,
                "n_red": int(len(reds)),
                "n_amber": int(len(ambers)),
                "red_metrics": "; ".join(
                    _metric_label(m, f) for m, f in zip(reds["metric"], reds["feature"])
                ),
                "amber_metrics": "; ".join(
                    _metric_label(m, f)
                    for m, f in zip(ambers["metric"], ambers["feature"])
                ),
                "overall_rag": alerts.overall_rag(group["band"]),
            }
        )
    return pd.DataFrame(rows)


def run_monitoring(
    baseline_snapshot: ModelSnapshot,
    months: dict[str, ModelSnapshot],
    config: ThresholdConfig | None = None,
    feature_types: dict[str, str] | None = None,
    baseline: Baseline | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full monthly monitoring job.

    Parameters
    ----------
    baseline_snapshot:
        Development-sample model outputs (with labels) that define the
        comparison baseline.
    months:
        Ordered mapping of month label -> :class:`ModelSnapshot`.
    config:
        Threshold config; defaults to the packaged YAML.
    feature_types:
        Optional overrides ``{feature_name: "numeric" | "categorical"}``.
    baseline:
        A pre-built :class:`Baseline`; computed from ``baseline_snapshot``
        when omitted.

    Returns
    -------
    dict with keys ``monitoring_results`` (long), ``monthly_summary`` (wide)
    and ``calibration_detail`` (long, per band).
    """
    config = config or load_config()
    if baseline is None:
        baseline = build_baseline(baseline_snapshot, config, feature_types)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []

    for month, snapshot in months.items():
        rows, summary, calib = _monitor_month(month, snapshot, baseline, config)
        all_rows.extend(rows)
        summaries.append(summary)
        if calib is not None:
            details.append(calib)

    results = pd.DataFrame(all_rows, columns=LONG_COLUMNS)
    monthly = pd.DataFrame(summaries)
    calibration_detail = (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    )
    return {
        "monitoring_results": results,
        "monthly_summary": monthly,
        "calibration_detail": calibration_detail,
        "alert_summary": build_alert_summary(results),
    }
