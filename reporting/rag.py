"""Red/Amber/Green (RAG) classification for model-monitoring metrics.

Banks do not treat every metric the same way. Two families are used here:

* **Stability metrics** (``psi``, ``csi``) are "lower is better" and are
  classified against absolute thresholds.
* **Performance / calibration metrics** (``auc``, ``gini``, ``ks``,
  ``calibration_gap``) are classified by how far they have moved from the
  model's development ("reference") value, because a level that is fine for
  one model can be poor for another.

The thresholds below are deliberately shipped as one readable dictionary so a
reviewer can argue with a number rather than reverse-engineer it. The defaults
follow common practice in credit-risk model monitoring:

======================  ============  ============
Metric                  Amber         Red
======================  ============  ============
PSI / CSI               >= 0.10       >= 0.25
AUC / Gini / KS drop    > 0.02        > 0.05
Calibration gap         > 0.02        > 0.05
======================  ============  ============

All classifications are ``"green"``, ``"amber"``, ``"red"`` or ``None`` when
the metric is missing.
"""

from __future__ import annotations

from typing import Optional

# Canonical ordering used whenever statuses must be combined into one status.
RAG_ORDER = {"green": 0, "amber": 1, "red": 2}

#: Default monitoring thresholds. Keys are metric names; ``kind`` decides how
#: the metric is interpreted and is the only place the two families differ.
DEFAULT_THRESHOLDS: dict[str, dict] = {
    # Absolute, lower-is-better.
    "psi": {"kind": "absolute", "amber": 0.10, "red": 0.25},
    "csi": {"kind": "absolute", "amber": 0.10, "red": 0.25},
    # Improvement is fine; deterioration (a drop, or a gap) is what matters.
    "auc": {"kind": "drop", "amber": 0.02, "red": 0.05},
    "gini": {"kind": "drop", "amber": 0.02, "red": 0.05},
    "ks": {"kind": "drop", "amber": 0.02, "red": 0.05},
    "calibration_gap": {"kind": "increase", "amber": 0.02, "red": 0.05},
}

#: Metrics that are shown as RAG-coloured tiles in the report, in display order.
MONITORED_METRICS: tuple[str, ...] = ("psi", "csi", "auc", "gini", "ks", "calibration_gap")

#: Human-readable labels and units for each metric.
METRIC_LABELS: dict[str, str] = {
    "psi": "Population Stability Index (PSI)",
    "csi": "Characteristic Stability Index (CSI)",
    "auc": "AUC (discrimination)",
    "gini": "Gini (discrimination)",
    "ks": "Kolmogorov-Smirnov (discrimination)",
    "calibration_gap": "Calibration gap (mean |predicted - observed|)",
}


def classify_metric(
    metric: str,
    value: Optional[float],
    reference_value: Optional[float] = None,
    thresholds: Optional[dict] = None,
) -> Optional[str]:
    """Return ``"green"`` / ``"amber"`` / ``"red"`` for a single metric.

    Parameters
    ----------
    metric:
        Name of the metric (must be a key in the threshold table).
    value:
        The observed value for the monitoring period.
    reference_value:
        The model-development value, required for ``kind="drop"`` metrics so
        the deterioration can be measured against it.
    thresholds:
        Override for :data:`DEFAULT_THRESHOLDS`.

    ``None`` is returned when the metric cannot be classified (missing value,
    missing reference for a ``drop`` metric, or an unknown metric name).
    """
    if value is None:
        return None
    table = (thresholds or DEFAULT_THRESHOLDS).get(metric)
    if table is None:
        return None

    kind = table["kind"]
    if kind == "absolute":
        magnitude = abs(value)
        if magnitude >= table["red"]:
            return "red"
        if magnitude >= table["amber"]:
            return "amber"
        return "green"

    if kind == "drop":
        if reference_value is None:
            return None
        drop = reference_value - value  # positive => deteriorated
        if drop > table["red"]:
            return "red"
        if drop > table["amber"]:
            return "amber"
        return "green"

    if kind == "increase":
        if value > table["red"]:
            return "red"
        if value > table["amber"]:
            return "amber"
        return "green"

    raise ValueError(f"Unknown threshold kind: {kind!r}")


def worst_status(statuses: list[Optional[str]]) -> Optional[str]:
    """Combine several statuses into the single worst status.

    Missing (``None``) statuses are ignored. If every status is missing the
    result is ``None``.
    """
    present = [s for s in statuses if s is not None]
    if not present:
        return None
    return max(present, key=lambda s: RAG_ORDER[s])


def status_for_color(status: Optional[str]) -> str:
    """Map a status to a hex colour used by the report and dashboard."""
    return {"green": "#1b8a4b", "amber": "#d99100", "red": "#c0392b"}.get(
        status or "", "#6b7280"
    )


def classify_breaches(
    metrics: dict[str, Optional[float]],
    reference: Optional[dict[str, float]] = None,
    thresholds: Optional[dict] = None,
) -> list[dict]:
    """Return a list of breach records for a period's metric dictionary.

    A "breach" is any metric classified ``amber`` or ``red``. Each record is a
    plain dict so it can be written straight to CSV or JSON.
    """
    reference = reference or {}
    out: list[dict] = []
    for metric in MONITORED_METRICS:
        value = metrics.get(metric)
        ref = reference.get(metric)
        status = classify_metric(
            metric, value, reference_value=ref, thresholds=thresholds
        )
        if status in ("amber", "red"):
            out.append(
                {
                    "metric": metric,
                    "value": value,
                    "reference": ref,
                    "level": status,
                    "message": _breach_message(metric, value, ref, status),
                }
            )
    return out


def _breach_message(
    metric: str, value: Optional[float], reference: Optional[float], level: str
) -> str:
    label = METRIC_LABELS.get(metric, metric)
    if reference is not None and metric in ("auc", "gini", "ks"):
        return (
            f"{label} fell to {value:.3f} from reference {reference:.3f} "
            f"({level.upper()})"
        )
    return f"{label} = {value:.3f} ({level.upper()})"
