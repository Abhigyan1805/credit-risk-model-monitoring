"""Load and validate the YAML monitoring-threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "monitoring_thresholds.yaml"


@dataclass(frozen=True)
class ThresholdConfig:
    """Validated view over the threshold YAML.

    Kept as plain nested dicts (``psi``, ``csi``, ``discrimination``,
    ``calibration``) with a few convenience accessors so callers can read
    exactly the keys documented in the YAML.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def psi(self) -> dict[str, Any]:
        return self.raw["psi"]

    @property
    def csi(self) -> dict[str, Any]:
        return self.raw["csi"]

    @property
    def discrimination(self) -> dict[str, Any]:
        return self.raw["discrimination"]

    @property
    def calibration(self) -> dict[str, Any]:
        return self.raw["calibration"]


_REQUIRED_TOP_LEVEL = ("psi", "csi", "discrimination", "calibration")


def load_config(path: str | Path | None = None) -> ThresholdConfig:
    """Read the threshold YAML and check the required sections exist.

    ``path`` defaults to the packaged ``config/monitoring_thresholds.yaml``.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"threshold config at {config_path} is not a mapping")
    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in raw]
    if missing:
        raise ValueError(f"threshold config missing required sections: {missing}")

    psi = raw["psi"]
    if psi["green_below"] >= psi["amber_below"]:
        raise ValueError("psi.green_below must be smaller than psi.amber_below")
    hl = raw["calibration"]["hl"]
    if hl["amber_pvalue_at_or_above"] >= hl["green_pvalue_at_or_above"]:
        raise ValueError(
            "calibration.hl.amber_pvalue_at_or_above must be smaller than "
            "calibration.hl.green_pvalue_at_or_above"
        )
    return ThresholdConfig(raw=raw)
