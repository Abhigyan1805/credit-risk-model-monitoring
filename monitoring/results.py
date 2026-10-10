"""Build the monthly monitoring-results table for one vintage.

Produces exactly the long table defined in ``docs/CONTRACTS.md`` section 5
(``common.schemas.RESULT_COLUMNS``): one row per ``(month, metric, segment)``
with the value, the amber/red thresholds used, the direction, the derived
``GREEN``/``AMBER``/``RED``/``NA`` rag and the observation count.

Metrics (segment ``ALL`` unless noted):

* ``volume``, ``mean_pd``, ``default_rate`` -- context;
* ``psi_score`` -- PSI of the score distribution vs the dev reference;
* ``csi`` (segment = feature) -- stability of each ``CSI_FEATURES`` column;
* ``auc``, ``gini``, ``ks`` -- discrimination, only when outcomes are observable;
* ``calibration_ratio`` (``ALL`` and ``band=<k>``) -- observed / predicted PD;
* ``hl_pvalue`` (``ALL``) -- Hosmer-Lemeshow goodness-of-fit p-value;
* ``binomial_pvalue`` (``band=<k>``) -- per-band exact binomial test;
* ``overall_rag`` (``ALL``) -- worst rag over the key metrics (0/1/2 for G/A/R).

A metric that cannot be computed (e.g. outcomes not yet observable) is emitted
with a NaN value and ``NA`` rag rather than dropped, so every month has the
same shape for reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CSI_FEATURES, MODEL_VERSION
from common.schemas import RESULT_COLUMNS
from monitoring import alerts
from monitoring.calibration import calibration_by_band, hosmer_lemeshow_by_band
from monitoring.config import Thresholds
from monitoring.discrimination import auc, gini, ks
from monitoring.stability import (
    csi,
    csi_categorical,
    psi,
    quantile_bin_edges,
)

# Metrics that drive the headline ``overall_rag``. Band-level rows are drill-down
# detail only: with ~10 bands, a per-band test has a multiple-testing false-positive
# risk, so the headline uses model-level metrics plus the overall HL test.
_HEADLINE_ALL_METRICS = {"psi_score", "gini", "ks", "calibration_ratio", "hl_pvalue"}


def _is_headline(row: dict) -> bool:
    if row["metric"] == "csi":
        return True
    return row["segment"] == "ALL" and row["metric"] in _HEADLINE_ALL_METRICS


def _row(
    month: str,
    metric: str,
    segment: str,
    value: float,
    spec: dict | None,
    direction: str,
    n_obs: int,
    model_version: str,
    rag: str,
    baseline: float | None = None,
) -> dict:
    amber, red = alerts.thresholds_for(spec, baseline)
    return {
        "month": month,
        "metric": metric,
        "segment": segment,
        "value": float(value),
        "amber_threshold": amber,
        "red_threshold": red,
        "direction": direction,
        "rag": rag,
        "n_obs": int(n_obs),
        "model_version": model_version,
    }


def _direction(spec: dict | None) -> str:
    return spec["direction"] if spec else "higher_worse"


def _model_version(month_df: pd.DataFrame) -> str:
    versions = month_df["model_version"].dropna().unique()
    return str(versions[0]) if len(versions) else MODEL_VERSION


def _observable(month_df: pd.DataFrame) -> bool:
    if month_df.empty or "outcome_observable" not in month_df.columns:
        return False
    return bool(month_df["outcome_observable"].fillna(False).all())


def _scored_outcomes(month_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with an observed 0/1 outcome: (default_flag, pd, score_band)."""
    mask = month_df["default_flag"].notna()
    subset = month_df.loc[mask]
    y = subset["default_flag"].astype(float).to_numpy()
    p = subset["pd"].astype(float).to_numpy()
    bands = subset["score_band"].to_numpy()
    return y, p, bands


def _baseline_discrimination(dev_df: pd.DataFrame) -> tuple[float, float]:
    """Dev-reference Gini and KS, used as the relative-deterioration baseline."""
    mask = dev_df["default_flag"].notna()
    y = dev_df.loc[mask, "default_flag"].astype(float).to_numpy()
    p = dev_df.loc[mask, "pd"].astype(float).to_numpy()
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return gini(y, p), ks(y, p)


def _score_psi(dev_df: pd.DataFrame, month_df: pd.DataFrame) -> float:
    dev_scores = dev_df["score"].dropna().to_numpy(dtype=float)
    month_scores = month_df["score"].dropna().to_numpy(dtype=float)
    if dev_scores.size == 0 or month_scores.size == 0:
        return float("nan")
    try:
        edges = quantile_bin_edges(dev_scores, n_bins=10)
        return psi(dev_scores, month_scores, bin_edges=edges)
    except ValueError:
        return float("nan")


def _feature_csi(dev_values: pd.Series, month_values: pd.Series) -> float:
    dev = dev_values.dropna()
    cur = month_values.dropna()
    if dev.empty or cur.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(dev):
        dev = dev.astype(float)
        cur = cur.astype(float)
        try:
            edges = quantile_bin_edges(dev.to_numpy(), n_bins=10)
        except ValueError:
            return float("nan")
        return csi(dev.to_numpy(), cur.to_numpy(), bin_edges=edges)
    return csi_categorical(dev.to_numpy(), cur.to_numpy())


def build_monitoring_results(
    month: str,
    dev_df: pd.DataFrame,
    month_df: pd.DataFrame,
    thresholds: Thresholds,
) -> pd.DataFrame:
    """Build the long monitoring-results table for vintage ``month``."""
    rows: list[dict] = []
    n = int(len(month_df))
    mv = _model_version(month_df) if n else MODEL_VERSION
    observable = _observable(month_df)
    base_gini, base_ks = _baseline_discrimination(dev_df)
    baselines = {"auc": None, "gini": base_gini, "ks": base_ks}

    # --- context -------------------------------------------------------
    rows.append(_row(month, "volume", "ALL", float(n), None, "higher_worse", n, mv, alerts.NA))
    mean_pd = float(month_df["pd"].mean()) if n else float("nan")
    rows.append(_row(month, "mean_pd", "ALL", mean_pd, None, "higher_worse", n, mv, alerts.NA))
    default_rate = float("nan")
    if observable and n:
        default_rate = float(month_df["default_flag"].astype(float).mean())
    rows.append(_row(month, "default_rate", "ALL", default_rate, None, "higher_worse", n, mv, alerts.NA))

    # --- stability -----------------------------------------------------
    psi_spec = thresholds.for_metric("psi_score")
    psi_value = _score_psi(dev_df, month_df)
    rows.append(
        _row(
            month,
            "psi_score",
            "ALL",
            psi_value,
            psi_spec,
            _direction(psi_spec),
            n,
            mv,
            alerts.rag(psi_value, psi_spec),
        )
    )

    csi_spec = thresholds.for_metric("csi")
    for feature in CSI_FEATURES:
        if feature not in dev_df.columns or feature not in month_df.columns:
            continue
        value = _feature_csi(dev_df[feature], month_df[feature])
        rows.append(
            _row(
                month,
                "csi",
                feature,
                value,
                csi_spec,
                _direction(csi_spec),
                n,
                mv,
                alerts.rag(value, csi_spec),
            )
        )

    # --- discrimination + calibration (seasoned vintages only) ---------
    if observable:
        y, p, bands = _scored_outcomes(month_df)
        usable = y.size > 0 and len(np.unique(y)) == 2
    else:
        y = p = bands = np.empty(0)
        usable = False

    for metric_name, fn in (("auc", auc), ("gini", gini), ("ks", ks)):
        spec = thresholds.for_metric(metric_name)
        base = baselines[metric_name]
        value = float(fn(y, p)) if usable else float("nan")
        rows.append(
            _row(
                month,
                metric_name,
                "ALL",
                value,
                spec,
                _direction(spec),
                n,
                mv,
                alerts.rag(value, spec, base),
                baseline=base,
            )
        )

    cal_ratio_spec = thresholds.for_metric("calibration_ratio")
    hl_spec = thresholds.for_metric("hl_pvalue")
    binom_spec = thresholds.for_metric("binomial_pvalue")

    if usable:
        band_frame = calibration_by_band(y, p, bands)
        hl = hosmer_lemeshow_by_band(y, p, bands)

        ratio_all = float(y.sum() / p.sum()) if p.sum() > 0 else float("nan")
        rows.append(
            _row(
                month,
                "calibration_ratio",
                "ALL",
                ratio_all,
                cal_ratio_spec,
                _direction(cal_ratio_spec),
                n,
                mv,
                alerts.rag(ratio_all, cal_ratio_spec),
            )
        )
        for row in band_frame.itertuples():
            segment = f"band={int(row.band)}"
            ratio = (
                row.observed_default_rate / row.mean_predicted_pd
                if row.mean_predicted_pd > 0
                else float("nan")
            )
            rows.append(
                _row(
                    month,
                    "calibration_ratio",
                    segment,
                    ratio,
                    cal_ratio_spec,
                    _direction(cal_ratio_spec),
                    row.n,
                    mv,
                    alerts.NA,
                )
            )
        rows.append(
            _row(
                month,
                "hl_pvalue",
                "ALL",
                hl["p_value"],
                hl_spec,
                _direction(hl_spec),
                n,
                mv,
                alerts.rag(hl["p_value"], hl_spec),
            )
        )
        for row in band_frame.itertuples():
            segment = f"band={int(row.band)}"
            rows.append(
                _row(
                    month,
                    "binomial_pvalue",
                    segment,
                    row.binomial_pvalue,
                    binom_spec,
                    _direction(binom_spec),
                    row.n,
                    mv,
                    alerts.rag(row.binomial_pvalue, binom_spec),
                )
            )
    else:
        for metric_name, spec in (
            ("hl_pvalue", hl_spec),
            ("calibration_ratio", cal_ratio_spec),
        ):
            rows.append(
                _row(month, metric_name, "ALL", float("nan"), spec, _direction(spec), n, mv, alerts.NA)
            )
        band_labels = sorted(pd.Series(month_df["score_band"]).dropna().unique() if n else [])
        for band in band_labels:
            segment = f"band={int(band)}"
            for metric_name, spec in (
                ("calibration_ratio", cal_ratio_spec),
                ("binomial_pvalue", binom_spec),
            ):
                rows.append(
                    _row(month, metric_name, segment, float("nan"), spec, _direction(spec), n, mv, alerts.NA)
                )

    # --- headline ------------------------------------------------------
    overall = alerts.worst_rag(r["rag"] for r in rows if _is_headline(r))
    rows.append(
        _row(month, "overall_rag", "ALL", alerts.rag_score(overall), None, "higher_worse", n, mv, overall)
    )

    result = pd.DataFrame(rows)[list(RESULT_COLUMNS)]
    result["n_obs"] = result["n_obs"].astype("int64")
    for column in ("value", "amber_threshold", "red_threshold"):
        result[column] = result[column].astype("float64")
    return result
