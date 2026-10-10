"""Traffic-light (GREEN / AMBER / RED / NA) alerting rules.

One generic function maps a metric value and its YAML rule to a rag label, so
the whole alerting policy is auditable in a few lines. ``NA`` means the metric
could not be computed (e.g. outcomes not yet observable) and is ignored when
rolling up to the headline ``overall_rag``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

GREEN = "GREEN"
AMBER = "AMBER"
RED = "RED"
NA = "NA"

_ORDER = {GREEN: 0, AMBER: 1, RED: 2}
_SCORE = {GREEN: 0.0, AMBER: 1.0, RED: 2.0}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _absolute(value: float, amber: float, red: float, direction: str) -> str:
    if direction == "higher_worse":
        if value < amber:
            return GREEN
        if value < red:
            return AMBER
        return RED
    if value > amber:
        return GREEN
    if value > red:
        return AMBER
    return RED


def _primary(value: float, spec: dict[str, Any], baseline: float | None) -> str:
    if spec.get("mode") == "relative_drop":
        if _is_missing(baseline) or float(baseline) <= 0:
            return NA
        drop = (float(baseline) - value) / float(baseline)
        if drop < spec["rel_amber"]:
            return GREEN
        if drop < spec["rel_red"]:
            return AMBER
        return RED
    if "amber_dev" in spec:  # calibration_ratio: two-sided around 1.0
        severity = abs(value - 1.0)
        if severity < spec["amber_dev"]:
            return GREEN
        if severity < spec["red_dev"]:
            return AMBER
        return RED
    return _absolute(value, float(spec["amber"]), float(spec["red"]), spec["direction"])


def rag(value: Any, spec: dict[str, Any] | None, baseline: float | None = None) -> str:
    """Map ``value`` to a rag label using ``spec`` from the threshold config.

    ``baseline`` is the development-baseline metric value, required for
    ``relative_drop`` metrics (Gini / KS). ``spec`` is ``None`` for metrics
    without a threshold (e.g. ``auc``), which yields ``NA``.
    """
    if spec is None or _is_missing(value):
        return NA
    value = float(value)
    primary = _primary(value, spec, baseline)
    if "abs_amber" in spec and "abs_red" in spec:
        floor = _absolute(value, float(spec["abs_amber"]), float(spec["abs_red"]), "lower_worse")
        return worst_rag([primary, floor])
    return primary


def thresholds_for(spec: dict[str, Any] | None, baseline: float | None = None) -> tuple[float, float]:
    """The (amber, red) thresholds to store in the results table (NaN if none).

    For ``relative_drop`` metrics the thresholds are the absolute levels derived
    from the baseline (``baseline * (1 - rel)``) so they share the value's scale.
    """
    if not spec:
        return (float("nan"), float("nan"))
    if "amber_dev" in spec:
        return (float(spec["amber_dev"]), float(spec["red_dev"]))
    if spec.get("mode") == "relative_drop":
        if _is_missing(baseline) or float(baseline) <= 0:
            return (float("nan"), float("nan"))
        base = float(baseline)
        return (base * (1.0 - spec["rel_amber"]), base * (1.0 - spec["rel_red"]))
    return (float(spec["amber"]), float(spec["red"]))


def worst_rag(rags: Iterable[str]) -> str:
    """Worst non-``NA`` rag among ``rags`` (``NA`` if none are assessable)."""
    worst: str | None = None
    for band in rags:
        if band not in _ORDER:
            continue
        if worst is None or _ORDER[band] > _ORDER[worst]:
            worst = band
    return worst if worst is not None else NA


def rag_score(band: str) -> float:
    """Numeric rag for the ``overall_rag`` row: 0/1/2 for G/A/R, NaN for NA."""
    return _SCORE.get(band, float("nan"))
