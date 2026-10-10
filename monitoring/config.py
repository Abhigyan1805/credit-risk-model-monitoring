"""Load the monitoring thresholds from ``config/thresholds.yaml``.

Thresholds are data, not code: the engine reads this file at run time so the
red/amber/green policy can be tuned (and audited) without a code change. The
loader validates the shape up front so a typo fails loudly rather than silently
producing wrong alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.yaml"

STANDARD_METRICS = ("psi_score", "csi", "gini", "ks", "hl_pvalue", "binomial_pvalue")
DIRECTIONS = ("higher_worse", "lower_worse")


@dataclass(frozen=True)
class Thresholds:
    """Validated threshold mapping: metric name -> rule spec."""

    raw: dict[str, Any]

    def for_metric(self, metric: str) -> dict[str, Any] | None:
        """Rule spec for ``metric`` (or ``None`` if the metric has no threshold)."""
        return self.raw.get(metric)


def _check_spec(metric: str, spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"thresholds: {metric} must be a mapping")
    direction = spec.get("direction")
    if direction not in DIRECTIONS:
        raise ValueError(f"thresholds: {metric}.direction must be one of {DIRECTIONS}")
    if metric == "calibration_ratio":
        if "amber_dev" not in spec or "red_dev" not in spec:
            raise ValueError("thresholds: calibration_ratio needs amber_dev and red_dev")
        if not spec["amber_dev"] < spec["red_dev"]:
            raise ValueError("thresholds: calibration_ratio amber_dev must be < red_dev")
        return
    mode = spec.get("mode", "absolute")
    if mode == "relative_drop":
        for key in ("rel_amber", "rel_red"):
            if key not in spec:
                raise ValueError(f"thresholds: {metric} needs {key} for relative_drop")
        if not spec["rel_amber"] < spec["rel_red"]:
            raise ValueError(f"thresholds: {metric} needs rel_amber < rel_red")
        if ("abs_amber" in spec) != ("abs_red" in spec):
            raise ValueError(f"thresholds: {metric} needs both abs_amber and abs_red")
        if "abs_amber" in spec and not spec["abs_red"] < spec["abs_amber"]:
            raise ValueError(f"thresholds: {metric} needs abs_red < abs_amber")
        return
    if mode != "absolute":
        raise ValueError(f"thresholds: {metric}.mode must be absolute or relative_drop")
    for key in ("amber", "red"):
        if key not in spec:
            raise ValueError(f"thresholds: {metric} needs {key}")
    amber, red = float(spec["amber"]), float(spec["red"])
    if direction == "higher_worse" and not amber < red:
        raise ValueError(f"thresholds: {metric} needs amber < red for higher_worse")
    if direction == "lower_worse" and not red < amber:
        raise ValueError(f"thresholds: {metric} needs red < amber for lower_worse")


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Read and validate ``config/thresholds.yaml`` (path overridable for tests)."""
    config_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"thresholds at {config_path} is not a mapping")
    missing = [m for m in (*STANDARD_METRICS, "calibration_ratio") if m not in raw]
    if missing:
        raise ValueError(f"thresholds missing required metrics: {missing}")
    for metric, spec in raw.items():
        _check_spec(metric, spec)
    return Thresholds(raw=raw)
