"""Canonical in-memory model of the monitoring results the reports consume.

The monitoring stage (a sibling component) is expected to emit a single JSON
document per run. Rather than couple the reporting code to the exact
producer payload, everything funnels through :func:`normalize_results`, which
maps the payload onto the small, explicit shape defined here. When the shared
``docs/CONTRACTS.md`` fixes the producer schema, only this mapping needs to
change.

Canonical payload
-----------------

.. code-block:: json

    {
      "model": {"name": ..., "version": ..., "reference_period": ...,
                 "target": ..., "reference_metrics": {...}},
      "generated_at": "2026-10-12T09:00:00Z",
      "periods": [
        {"period": "2018-02", "n_obs": 1234, "default_rate": 0.05,
         "metrics": {"psi": 0.03, "csi": 0.02, "auc": 0.71, "gini": 0.42,
                     "ks": 0.31, "calibration_gap": 0.01},
         "rag": "green"}
      ],
      "score_distribution": [
        {"period": "2018-02", "band": "A", "score_min": 0.0, "score_max": 0.1,
         "count": 100, "share": 0.10, "expected_share": 0.10}
      ],
      "calibration_by_band": [
        {"period": "2018-02", "band": "A", "count": 100,
         "predicted_pd": 0.01, "observed_pd": 0.012}
      ],
      "feature_csi": [
        {"period": "2018-02", "feature": "annual_inc", "csi": 0.02}
      ],
      "breaches": [
        {"period": "2018-02", "metric": "psi", "value": 0.31,
         "threshold": 0.25, "level": "red", "message": "..."}
      ]
    }

The ``periods.metrics`` block is the spine: ``psi``/``csi`` are stability
metrics; ``auc``/``gini``/``ks`` are discrimination metrics whose RAG depends
on ``model.reference_metrics``; ``calibration_gap`` is the mean absolute
difference between predicted and observed default rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from reporting.rag import DEFAULT_THRESHOLDS, classify_metric, worst_status


@dataclass
class MonitoringResults:
    """Normalized monitoring results ready for table/chart construction."""

    model: dict[str, Any]
    periods: list[dict[str, Any]]
    score_distribution: list[dict[str, Any]] = field(default_factory=list)
    calibration_by_band: list[dict[str, Any]] = field(default_factory=list)
    feature_csi: list[dict[str, Any]] = field(default_factory=list)
    breaches: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""
    thresholds: dict[str, dict] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    # -- convenience accessors -------------------------------------------------
    @property
    def reference_metrics(self) -> dict[str, float]:
        return dict(self.model.get("reference_metrics", {}))

    @property
    def periods_sorted(self) -> list[dict[str, Any]]:
        return sorted(self.periods, key=lambda p: str(p.get("period", "")))

    @property
    def latest_period(self) -> Optional[dict[str, Any]]:
        ordered = self.periods_sorted
        return ordered[-1] if ordered else None

    def metric_series(self, metric: str) -> list[tuple[str, Optional[float]]]:
        """Return ``[(period, value), ...]`` for one metric, oldest first."""
        out: list[tuple[str, Optional[float]]] = []
        for p in self.periods_sorted:
            out.append((str(p.get("period")), p.get("metrics", {}).get(metric)))
        return out


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_results(payload: dict[str, Any]) -> MonitoringResults:
    """Validate and normalize a raw monitoring-results payload.

    Raises ``ValueError`` if the payload has no periods, since a report with no
    periods would silently look "all green".
    """
    if not isinstance(payload, dict):
        raise ValueError("Monitoring results must be a JSON object")

    model = dict(payload.get("model", {}))
    reference = {
        k: _as_float(v) for k, v in model.get("reference_metrics", {}).items()
    }
    model["reference_metrics"] = {k: v for k, v in reference.items() if v is not None}

    raw_periods = payload.get("periods", [])
    if not raw_periods:
        raise ValueError("Monitoring results contain no periods")

    periods: list[dict[str, Any]] = []
    for raw in raw_periods:
        metrics = {
            k: _as_float(v) for k, v in dict(raw.get("metrics", {})).items()
        }
        periods.append(
            {
                "period": str(raw.get("period")),
                "n_obs": _as_int(raw.get("n_obs")),
                "default_rate": _as_float(raw.get("default_rate")),
                "metrics": metrics,
                "rag": raw.get("rag") or _period_rag(metrics, model["reference_metrics"]),
            }
        )

    return MonitoringResults(
        model=model,
        periods=periods,
        score_distribution=[
            _normalize_record(r) for r in payload.get("score_distribution", [])
        ],
        calibration_by_band=[
            _normalize_record(r) for r in payload.get("calibration_by_band", [])
        ],
        feature_csi=[_normalize_record(r) for r in payload.get("feature_csi", [])],
        breaches=[dict(b) for b in payload.get("breaches", [])],
        generated_at=str(payload.get("generated_at", "")),
        thresholds=dict(payload.get("thresholds") or DEFAULT_THRESHOLDS),
    )


def load_results(path: str) -> MonitoringResults:
    """Load and normalize a monitoring-results JSON file from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        return normalize_results(json.load(fh))


def _period_rag(metrics: dict[str, Optional[float]], reference: dict[str, float]) -> Optional[str]:
    statuses = [
        classify_metric(m, metrics.get(m), reference_value=reference.get(m))
        for m in ("psi", "csi", "auc", "gini", "ks", "calibration_gap")
    ]
    return worst_status(statuses)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Coerce numeric-looking fields to floats, leaving other fields as-is."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        if key in ("period", "band", "feature"):
            out[key] = str(value)
        elif key == "count":
            out[key] = _as_int(value)
        else:
            coerced = _as_float(value)
            out[key] = coerced if coerced is not None else value
    return out
