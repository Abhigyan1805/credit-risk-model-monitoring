"""Population stability index (PSI) and characteristic stability index (CSI).

The PSI measures how far a *current* distribution has moved away from a
*reference* (development baseline) distribution, bin by bin::

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

where each term is the proportion of records falling in bin ``i``. A value of
zero means the two distributions are identical; rule-of-thumb bands used across
the industry are ``< 0.10`` stable (green), ``0.10 - 0.25`` some shift (amber)
and ``> 0.25`` major shift (red).

The CSI is the same statistic applied to a single model characteristic
(feature): numeric features are binned on baseline quantiles, categorical
features are compared category by category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_EPSILON = 1e-6


def psi_from_proportions(
    expected: np.ndarray,
    actual: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI from two aligned vectors of bin *proportions* (each summing to 1).

    Proportions are clipped at ``epsilon`` before the logarithm so a bin that
    is present in one distribution but empty in the other cannot produce an
    infinite value -- the standard practical fix.
    """
    e = np.clip(np.asarray(expected, dtype=float), epsilon, None)
    a = np.clip(np.asarray(actual, dtype=float), epsilon, None)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same number of bins")
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_counts(
    expected_counts: np.ndarray,
    actual_counts: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI from aligned bin *counts*; counts are normalised internally."""
    expected_counts = np.asarray(expected_counts, dtype=float)
    actual_counts = np.asarray(actual_counts, dtype=float)
    if expected_counts.sum() <= 0 or actual_counts.sum() <= 0:
        raise ValueError("expected and actual counts must each sum to a positive value")
    return psi_from_proportions(
        expected_counts / expected_counts.sum(),
        actual_counts / actual_counts.sum(),
        epsilon=epsilon,
    )


def quantile_bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin edges from ``n_bins`` equal-frequency quantiles of ``reference``.

    The outer edges are widened to +/- infinity so that current values outside
    the baseline range still land in the end bins. Duplicate quantiles (e.g. a
    feature with many equal values) are collapsed, which may yield fewer than
    ``n_bins`` bins.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.size == 0:
        raise ValueError("reference sample is empty")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, qs))
    if edges.size < 2:
        raise ValueError("reference sample has no spread; cannot form bins")
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    bin_edges: np.ndarray | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI between two raw score/value samples.

    Binning uses equal-frequency quantiles of ``expected`` (the baseline). Pass
    ``bin_edges`` to reuse frozen baseline edges, which is what monthly
    monitoring does so month-to-month values stay comparable.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if bin_edges is None:
        bin_edges = quantile_bin_edges(expected, n_bins=n_bins)
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    return psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)


def csi(
    baseline_values: np.ndarray,
    current_values: np.ndarray,
    n_bins: int = 10,
    bin_edges: np.ndarray | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Characteristic stability index for one *numeric* feature.

    This is PSI specialised to a single characteristic; the baseline values
    define the bins unless ``bin_edges`` (frozen at development time) is given.
    """
    return psi(
        baseline_values,
        current_values,
        n_bins=n_bins,
        bin_edges=bin_edges,
        epsilon=epsilon,
    )


def csi_categorical(
    baseline_values: np.ndarray | pd.Series,
    current_values: np.ndarray | pd.Series,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """CSI for one *categorical* feature, compared category by category.

    The union of categories seen in either sample forms the bins so a category
    that appears only in the current month is counted rather than silently
    dropped.
    """
    base = pd.Series(np.asarray(baseline_values))
    curr = pd.Series(np.asarray(current_values))
    categories = pd.Index(base.unique()).union(pd.Index(curr.unique()))
    expected_counts = base.value_counts().reindex(categories, fill_value=0).to_numpy()
    actual_counts = curr.value_counts().reindex(categories, fill_value=0).to_numpy()
    return psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)
