"""Hand-made monitoring results (contract schema) for checker tests: one month and the month before."""

from __future__ import annotations

import math

import pandas as pd

from commentary.facts import MonthFacts, build_facts
from common.config import CSI_FEATURES, MODEL_VERSION

NAN = math.nan
# metric -> (value, rag, amber, red, direction)
_SCALARS_MAR = {
    "volume": (10234, "NA", NAN, NAN, "higher_worse"),
    "mean_pd": (0.118, "NA", NAN, NAN, "higher_worse"),
    "default_rate": (0.163, "NA", NAN, NAN, "higher_worse"),
    "psi_score": (0.1834, "AMBER", 0.10, 0.25, "higher_worse"),
    "auc": (0.6562, "NA", NAN, NAN, "higher_worse"),
    "gini": (0.3124, "GREEN", 0.2979, 0.24825, "lower_worse"),
    "ks": (0.2361, "GREEN", 0.2223, 0.18525, "lower_worse"),
    "calibration_ratio": (1.3812, "RED", 0.15, 0.30, "higher_worse"),
    "hl_pvalue": (0.00012, "RED", 0.05, 0.01, "lower_worse"),
}
_SCALARS_FEB = {
    "volume": (9876, "NA", NAN, NAN, "higher_worse"),
    "mean_pd": (0.116, "NA", NAN, NAN, "higher_worse"),
    "default_rate": (0.151, "NA", NAN, NAN, "higher_worse"),
    "psi_score": (0.1511, "AMBER", 0.10, 0.25, "higher_worse"),
    "auc": (0.6625, "NA", NAN, NAN, "higher_worse"),
    "gini": (0.3250, "GREEN", 0.2979, 0.24825, "lower_worse"),
    "ks": (0.2302, "GREEN", 0.2223, 0.18525, "lower_worse"),
    "calibration_ratio": (1.2903, "AMBER", 0.15, 0.30, "higher_worse"),
    "hl_pvalue": (0.0031, "RED", 0.05, 0.01, "lower_worse"),
}
_CSI_MAR = {"int_rate": (0.3105, "RED"), "grade": (0.1207, "AMBER"), "fico": (0.0412, "GREEN")}
_CSI_FEB = {"int_rate": (0.2701, "RED"), "grade": (0.1402, "AMBER"), "fico": (0.0388, "GREEN")}
_BANDS_MAR = {8: (0.0021, "RED"), 9: (0.031, "AMBER")}


def _row(month, metric, segment, value, rag, amber=NAN, red=NAN, direction="higher_worse"):
    return {
        "month": month,
        "metric": metric,
        "segment": segment,
        "value": float(value),
        "amber_threshold": float(amber),
        "red_threshold": float(red),
        "direction": direction,
        "rag": rag,
        "n_obs": 1000,
        "model_version": MODEL_VERSION,
    }


def results_frame(month: str, scalars: dict, csi: dict, bands: dict, overall: str) -> pd.DataFrame:
    rows = [_row(month, m, "ALL", v, r, a, rd, d) for m, (v, r, a, rd, d) in scalars.items()]
    for feature in CSI_FEATURES:
        value, rag = csi.get(feature, (0.012, "GREEN"))
        rows.append(_row(month, "csi", feature, value, rag, 0.10, 0.25))
    for band in range(1, 11):
        value, rag = bands.get(band, (0.4, "GREEN"))
        rows.append(_row(month, "binomial_pvalue", f"band={band}", value, rag, 0.05, 0.01, "lower_worse"))
        rows.append(_row(month, "calibration_ratio", f"band={band}", 1.0 + band / 50, "NA", 0.15, 0.30))
    rows.append(_row(month, "overall_rag", "ALL", {"GREEN": 0, "AMBER": 1, "RED": 2}[overall], overall))
    frame = pd.DataFrame(rows)
    for col in ("month", "metric", "segment", "direction", "rag", "model_version"):
        frame[col] = frame[col].astype("string")
    return frame


def march() -> MonthFacts:
    """2015-03: overall RED; PSI AMBER and rising, calibration RED, int_rate CSI RED."""
    cur = results_frame("2015-03", _SCALARS_MAR, _CSI_MAR, _BANDS_MAR, "RED")
    prev = results_frame("2015-02", _SCALARS_FEB, _CSI_FEB, {}, "RED")
    return build_facts(cur, prev)


def first_month() -> MonthFacts:
    """2014-01: no previous month."""
    return build_facts(results_frame("2014-01", _SCALARS_MAR, _CSI_MAR, _BANDS_MAR, "RED"))


def immature() -> MonthFacts:
    """2016-04: outcomes not observable, so Gini/KS/AUC/calibration are NaN with rag NA."""
    na = {m: (NAN, "NA", a, r, d) for m, (_, _, a, r, d) in _SCALARS_MAR.items()}
    na["volume"] = (11020, "NA", NAN, NAN, "higher_worse")
    na["mean_pd"] = (0.121, "NA", NAN, NAN, "higher_worse")
    na["psi_score"] = (0.2811, "RED", 0.10, 0.25, "higher_worse")
    prev = dict(na, psi_score=(0.2402, "AMBER", 0.10, 0.25, "higher_worse"))
    nan_bands = {b: (NAN, "NA") for b in range(1, 11)}
    return build_facts(
        results_frame("2016-04", na, _CSI_MAR, nan_bands, "RED"),
        results_frame("2016-03", prev, _CSI_FEB, nan_bands, "RED"),
    )


GOOD_MARCH = """What changed: In 2015-03 the overall status is RED. Score PSI rose to 0.183 from 0.151 in 2015-02 and stays AMBER. Gini fell to 0.312 but remains GREEN, and KS is 0.236 (GREEN). The calibration ratio deteriorated to 1.381 and is now RED, up from AMBER last month.

Alerts: The calibration ratio and the Hosmer-Lemeshow p-value (0.00012) are RED, score PSI is AMBER, and the interest rate CSI of 0.311 is RED. The binomial test flags bands 8 and 9.

Likely drivers: The shift in interest rate (CSI 0.311, RED) and grade (CSI 0.121, AMBER) is the likely driver of the score drift; FICO is stable.

Recommended action: Review PD calibration with the model owner, since observed defaults (default rate 16.3%) exceed the mean PD of 0.118."""
