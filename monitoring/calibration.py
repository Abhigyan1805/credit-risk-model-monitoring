"""Calibration: does the model's predicted default rate match reality?

Discrimination metrics can look healthy while the *level* of predictions is
wrong -- a model can rank borrowers well yet predict 2% defaults when 5%
actually default. Calibration checks compare the predicted probability with
the observed default rate, overall and band by band:

* ``calibration_bands`` -- groups loans into score bands (equal-frequency by
  predicted PD) and reports n, mean predicted PD, observed default rate and an
  exact two-sided binomial p-value per band;
* ``hosmer_lemeshow`` -- the classic goodness-of-fit statistic, the sum of
  chi-square standardised (observed - expected) differences across the bands,
  compared to a chi-square distribution with ``n_bands - 2`` degrees of freedom.

The per-band binomial test is the "Jeffreys-style" exact view of calibration
recommended for small samples: it answers whether the observed number of
defaults in a band is compatible with the predicted rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if y.size == 0:
        raise ValueError("inputs are empty")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("y_prob must be probabilities in [0, 1]")
    return y, p


def _equal_frequency_groups(y_prob: np.ndarray, n_bands: int) -> list[np.ndarray]:
    """Indices split into ``n_bands`` equal-frequency groups by predicted PD."""
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    if n_bands > y_prob.size:
        raise ValueError("n_bands cannot exceed the number of observations")
    order = np.argsort(y_prob, kind="mergesort")
    return [idx for idx in np.array_split(order, n_bands) if idx.size > 0]


def _binomial_pvalue(observed: int, n: int, p: float) -> float:
    """Two-sided exact binomial p-value, tolerant of p exactly 0 or 1."""
    if n == 0:
        return float("nan")
    if p <= 0.0 or p >= 1.0:
        expected = n * p
        return 1.0 if np.isclose(observed, expected) else 0.0
    return float(binomtest(observed, n, p).pvalue)


def calibration_bands(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Predicted vs observed default rate per equal-frequency score band.

    Returns a tidy DataFrame with one row per band: the loan count, mean
    predicted PD, expected and observed default counts/rates, the signed
    deviation and the exact two-sided binomial p-value for that band.
    """
    y, p = _validate(y_true, y_prob)
    rows = []
    for band, idx in enumerate(_equal_frequency_groups(p, n_bands)):
        n = int(idx.size)
        observed = int(y[idx].sum())
        mean_pred = float(p[idx].mean())
        expected = n * mean_pred
        rows.append(
            {
                "band": band,
                "n": n,
                "mean_predicted_pd": mean_pred,
                "expected_defaults": expected,
                "observed_defaults": observed,
                "observed_default_rate": observed / n,
                "deviation": observed / n - mean_pred,
                "binomial_pvalue": _binomial_pvalue(observed, n, mean_pred),
            }
        )
    return pd.DataFrame(rows)


def hosmer_lemeshow(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bands: int = 10,
) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test for calibration.

    ``HL = sum_i (O_i - E_i)^2 / (n_i * p_i * (1 - p_i))`` with ``E_i = n_i p_i``
    and ``p_i`` the mean predicted PD in band ``i``. A large statistic (small
    p-value) means the model is miscalibrated. Degrees of freedom are
    ``n_bands - 2``; the test needs at least 3 bands.
    """
    y, p = _validate(y_true, y_prob)
    groups = _equal_frequency_groups(p, n_bands)
    if len(groups) < 3:
        raise ValueError("Hosmer-Lemeshow needs at least 3 non-empty bands")

    statistic = 0.0
    rows = []
    for band, idx in enumerate(groups):
        n = int(idx.size)
        observed = int(y[idx].sum())
        mean_pred = float(p[idx].mean())
        expected = n * mean_pred
        denom = n * mean_pred * (1.0 - mean_pred)
        if denom <= 0:
            raise ValueError("a band has mean predicted PD of 0 or 1; cannot test")
        statistic += (observed - expected) ** 2 / denom
        rows.append(
            {
                "band": band,
                "n": n,
                "mean_predicted_pd": mean_pred,
                "expected_defaults": expected,
                "observed_defaults": observed,
                "observed_default_rate": observed / n,
                "chi2_component": (observed - expected) ** 2 / denom,
            }
        )

    dof = len(groups) - 2
    p_value = float(chi2.sf(statistic, dof))
    return {
        "statistic": float(statistic),
        "dof": int(dof),
        "p_value": p_value,
        "n_bands": len(groups),
        "bands": pd.DataFrame(rows),
    }


def _groups_from_labels(labels: np.ndarray) -> list[tuple[object, np.ndarray]]:
    """Split indices by label, preserving the sorted label order."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    return [(label, order[sorted_labels == label]) for label in np.unique(sorted_labels)]


def _band_frame(
    y: np.ndarray,
    p: np.ndarray,
    groups: list[tuple[object, np.ndarray]],
) -> pd.DataFrame:
    """One row per (label, index) group with counts, rates and test components."""
    rows = []
    for label, idx in groups:
        n = int(idx.size)
        if n == 0:
            continue
        observed = int(y[idx].sum())
        mean_pred = float(p[idx].mean())
        expected = n * mean_pred
        denom = n * mean_pred * (1.0 - mean_pred)
        rows.append(
            {
                "band": label,
                "n": n,
                "mean_predicted_pd": mean_pred,
                "expected_defaults": expected,
                "observed_defaults": observed,
                "observed_default_rate": observed / n,
                "deviation": observed / n - mean_pred,
                "binomial_pvalue": _binomial_pvalue(observed, n, mean_pred),
                "chi2_component": (observed - expected) ** 2 / denom if denom > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_by_band(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bands: np.ndarray,
) -> pd.DataFrame:
    """Predicted vs observed calibration using *pre-defined* band labels.

    ``bands`` is the model's frozen ``score_band`` (1 safest .. 10 riskiest), so
    calibration is reported on the same bands the scorecard uses rather than on
    bands re-derived each month. Returns one row per band present.
    """
    y, p = _validate(y_true, y_prob)
    bands = np.asarray(bands)
    if bands.shape != y.shape:
        raise ValueError("bands must be aligned with y_true")
    return _band_frame(y, p, _groups_from_labels(bands))


def hosmer_lemeshow_by_band(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bands: np.ndarray,
) -> dict:
    """Hosmer-Lemeshow statistic using pre-defined band labels.

    Degrees of freedom are ``n_non-empty_bands - 2``. Returns ``NA`` values
    (NaN) when fewer than three bands are usable.
    """
    y, p = _validate(y_true, y_prob)
    bands = np.asarray(bands)
    if bands.shape != y.shape:
        raise ValueError("bands must be aligned with y_true")
    frame = _band_frame(y, p, _groups_from_labels(bands))
    usable = frame[frame["chi2_component"].notna()]
    if len(usable) < 3:
        return {
            "statistic": float("nan"),
            "dof": 0,
            "p_value": float("nan"),
            "n_bands": int(len(usable)),
            "bands": frame,
        }
    statistic = float(usable["chi2_component"].sum())
    dof = int(len(usable) - 2)
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": float(chi2.sf(statistic, dof)),
        "n_bands": int(len(usable)),
        "bands": frame,
    }
