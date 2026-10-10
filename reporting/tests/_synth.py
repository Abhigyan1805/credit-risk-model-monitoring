"""Synthetic monitoring-results + scored data for reporting tests.

Generates Parquet files in the exact contract schemas so reporting can be tested
without running the full pipeline or downloading LendingClub. The series is
deliberately deteriorating so GREEN, AMBER and RED all appear.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common.config import CSI_FEATURES, MODEL_VERSION
from reporting.rag import rag_from_value, worst_rag

MONTHS: list[str] = ["2014-01", "2014-02", "2014-03", "2014-04", "2014-05", "2014-06"]
FEATURES: list[str] = list(CSI_FEATURES)

# (metric, threshold spec) for the scalar ALL metrics.
_DIRECTIONS = {
    "psi_score": ("higher_worse", 0.10, 0.25),
    "gini": ("lower_worse", 0.40, 0.30),
    "ks": ("lower_worse", 0.25, 0.20),
    "calibration_ratio": ("higher_worse", 0.15, 0.30),
    "hl_pvalue": ("lower_worse", 0.05, 0.01),
}
_OVERALL_CODE = {"GREEN": 0, "AMBER": 1, "RED": 2, "NA": 2}


def _row(
    month: str,
    metric: str,
    segment: str,
    value,
    rag: str,
    n_obs: int,
    amber=None,
    red=None,
    direction="higher_worse",
) -> dict:
    return {
        "month": month,
        "metric": metric,
        "segment": segment,
        "value": float(value),
        "amber_threshold": float(amber) if amber is not None else float("nan"),
        "red_threshold": float(red) if red is not None else float("nan"),
        "direction": direction,
        "rag": rag,
        "n_obs": int(n_obs),
        "model_version": MODEL_VERSION,
    }


def monitoring_frame(months: list[str] | None = None) -> pd.DataFrame:
    months = months or MONTHS
    rows: list[dict] = []
    for i, month in enumerate(months):
        n_obs = 1000 + 10 * i
        rag_by_metric: dict[str, str] = {}

        scalar_values = {
            "volume": 1000 + 10 * i,
            "mean_pd": 0.05 + 0.002 * i,
            "default_rate": 0.048 + 0.004 * i,
            "psi_score": 0.03 + 0.05 * i,
            "gini": 0.44 - 0.02 * i,
            "ks": 0.33 - 0.02 * i,
            "calibration_ratio": 1.0 + 0.06 * i,
            "hl_pvalue": 0.20 - 0.045 * i,
        }
        for metric, value in scalar_values.items():
            if metric in _DIRECTIONS:
                direction, amber, red = _DIRECTIONS[metric]
                # calibration_ratio thresholds are deviations from 1.0, not the ratio itself.
                probe = abs(value - 1.0) if metric == "calibration_ratio" else value
                rag = rag_from_value(probe, amber, red, direction)
            else:
                direction, amber, red, rag = "higher_worse", None, None, "NA"
            rag_by_metric[metric] = rag
            rows.append(_row(month, metric, "ALL", value, rag, n_obs, amber, red, direction))

        # auc has no threshold in config/thresholds.yaml -> not RAG-rated.
        auc = 0.72 - 0.012 * i
        rows.append(_row(month, "auc", "ALL", auc, "NA", n_obs))

        # Feature CSI.
        for j, feature in enumerate(FEATURES):
            csi = 0.02 + 0.04 * i + 0.001 * j
            rag = rag_from_value(csi, 0.10, 0.25, "higher_worse")
            rows.append(_row(month, "csi", feature, csi, rag, n_obs, 0.10, 0.25, "higher_worse"))

        # Per-band calibration.
        for band in range(1, 11):
            ratio = 1.0 + 0.01 * band + 0.05 * i
            pval = max(0.0, 0.30 - 0.03 * band - 0.03 * i)
            rows.append(
                _row(
                    month,
                    "calibration_ratio",
                    f"band={band}",
                    ratio,
                    rag_from_value(abs(ratio - 1.0), 0.15, 0.30, "higher_worse"),
                    n_obs,
                    0.15,
                    0.30,
                    "higher_worse",
                )
            )
            rows.append(
                _row(
                    month,
                    "binomial_pvalue",
                    f"band={band}",
                    pval,
                    rag_from_value(pval, 0.05, 0.01, "lower_worse"),
                    n_obs,
                    0.05,
                    0.01,
                    "lower_worse",
                )
            )

        overall = worst_rag(list(rag_by_metric.values()))
        rows.append(
            _row(month, "overall_rag", "ALL", _OVERALL_CODE[overall], overall, n_obs, direction="lower_worse")
        )

    frame = pd.DataFrame(rows)
    frame["value"] = frame["value"].astype("float64")
    frame["n_obs"] = frame["n_obs"].astype("int64")
    return frame


def scored_frame(month: str, n: int = 500) -> pd.DataFrame:
    bands = [((k % 10) + 1) for k in range(n)]
    frame = pd.DataFrame({"issue_month": [month] * n, "score_band": bands})
    for column in ("loan_id", "pd", "score", "outcome_observable", "model_version"):
        frame[column] = "x"
    frame["default_flag"] = pd.array([0] * n, dtype="Int8")
    for feature in FEATURES:
        frame[feature] = 0
    return frame


def write_synthetic_data(root: Path, months: list[str] | None = None) -> Path:
    """Write monitoring results, scored vintages and a dev reference under ``root``."""
    months = months or MONTHS
    results_dir = root / "monitoring" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = monitoring_frame(months)
    for month in months:
        frame[frame["month"] == month].to_parquet(results_dir / f"month={month}.parquet", index=False)

    scored_dir = root / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    for month in months:
        scored_frame(month).to_parquet(scored_dir / f"vintage={month}.parquet", index=False)
    scored_frame("dev", n=800).to_parquet(scored_dir / "vintage=dev.parquet", index=False)
    return root
