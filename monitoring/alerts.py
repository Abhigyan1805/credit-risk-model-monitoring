"""Traffic-light (red / amber / green) alerting rules.

Each ``band_*`` function turns one metric value into a ``"green"``, ``"amber"``,
``"red"`` or ``"na"`` label using thresholds from the YAML config. Keeping the
rules in one tiny module makes the alerting policy auditable and easy to tune:
a reviewer can read the five functions and know exactly why a month lit up.
"""

from __future__ import annotations

from typing import Iterable

from monitoring.config import ThresholdConfig

RAG_ORDER = {"green": 0, "amber": 1, "red": 2}


def band_psi(value: float, config: ThresholdConfig) -> str:
    """PSI / CSI band: ``< green_below`` green, ``< amber_below`` amber, else red."""
    section = config.psi
    if value < section["green_below"]:
        return "green"
    if value < section["amber_below"]:
        return "amber"
    return "red"


def band_csi(value: float, config: ThresholdConfig) -> str:
    """CSI band (same shape as PSI but its own tunable thresholds)."""
    section = config.csi
    if value < section["green_below"]:
        return "green"
    if value < section["amber_below"]:
        return "amber"
    return "red"


def band_gini_drop(baseline_gini: float, current_gini: float, config: ThresholdConfig) -> str:
    """Gini band from the *relative* drop versus the development baseline.

    ``rel_drop = (baseline - current) / baseline``. A model whose Gini improves
    (negative drop) is green; the thresholds widen the band as discrimination
    decays.
    """
    if baseline_gini <= 0:
        return "na"
    section = config.discrimination["gini"]
    rel_drop = (baseline_gini - current_gini) / baseline_gini
    if rel_drop < section["rel_drop_green_below"]:
        return "green"
    if rel_drop < section["rel_drop_amber_below"]:
        return "amber"
    return "red"


def band_ks_drop(baseline_ks: float, current_ks: float, config: ThresholdConfig) -> str:
    """KS band from the *absolute* drop versus the development baseline."""
    section = config.discrimination["ks"]
    abs_drop = baseline_ks - current_ks
    if abs_drop < section["abs_drop_green_below"]:
        return "green"
    if abs_drop < section["abs_drop_amber_below"]:
        return "amber"
    return "red"


def band_hl_pvalue(p_value: float, config: ThresholdConfig) -> str:
    """Overall calibration band from the Hosmer-Lemeshow p-value.

    A high p-value means observed and expected agree, so green is the *upper*
    tail here -- the opposite direction to the stability metrics.
    """
    section = config.calibration["hl"]
    if p_value >= section["green_pvalue_at_or_above"]:
        return "green"
    if p_value >= section["amber_pvalue_at_or_above"]:
        return "amber"
    return "red"


def band_calibration_band(
    deviation: float,
    p_value: float,
    n: int,
    config: ThresholdConfig,
) -> str:
    """Band for a single score band from its binomial p-value and deviation.

    Bands smaller than ``min_band_size`` are not alerted on (too noisy). The
    deviation guard fires red/amber even when the binomial test is not yet
    significant, to catch a band drifting well away from its predicted rate.
    """
    section = config.calibration
    if n < section["binomial"]["min_band_size"]:
        return "na"
    if p_value < section["binomial"]["alpha_red"]:
        return "red"
    if abs(deviation) >= section["max_band_deviation_red"]:
        return "red"
    if p_value < section["binomial"]["alpha_green"]:
        return "amber"
    if abs(deviation) >= section["max_band_deviation_amber"]:
        return "amber"
    return "green"


def overall_rag(bands: Iterable[str]) -> str:
    """Worst non-``na`` band, so one red metric makes the month red."""
    worst = "green"
    seen = False
    for band in bands:
        if band not in RAG_ORDER:
            continue
        seen = True
        if RAG_ORDER[band] > RAG_ORDER[worst]:
            worst = band
    return worst if seen else "na"
