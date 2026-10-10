"""Discrimination metrics: AUC, Gini and the Kolmogorov-Smirnov statistic.

All three answer "how well does the model rank good vs bad borrowers?". They
are computed on seasoned vintages only -- months whose loans have had time to
default or fully repay -- so the observed outcome is known.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _check_binary(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must be binary (0 = good, 1 = default)")
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("y_true must contain both classes")
    return y


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U / rank formula.

    ``AUC = (sum of ranks of defaults - n_pos*(n_pos+1)/2) / (n_pos * n_neg)``

    Tied scores receive the average rank (``scipy.stats.rankdata``), so the
    result matches the probability that a random default is scored above a
    random non-default, with ties counted as half.
    """
    y = _check_binary(y_true)
    s = np.asarray(y_score, dtype=float)
    if s.shape != y.shape:
        raise ValueError("y_true and y_score must have the same length")
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    ranks = rankdata(s)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient, the normalised AUC: ``Gini = 2*AUC - 1``.

    Ranges from 0 (no discrimination, the diagonal) to 1 (perfect ranking).
    """
    return 2.0 * auc(y_true, y_score) - 1.0


def ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic = max gap between the score CDFs.

    Sorts by score and tracks the cumulative share of defaults versus
    non-defaults; the largest absolute difference is the KS. 0 means the two
    score distributions are identical, 1 means perfectly separated.
    """
    y = _check_binary(y_true)
    s = np.asarray(y_score, dtype=float)
    if s.shape != y.shape:
        raise ValueError("y_true and y_score must have the same length")
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    n_pos = float((y_sorted == 1).sum())
    n_neg = float((y_sorted == 0).sum())
    cum_pos = np.cumsum(y_sorted == 1) / n_pos
    cum_neg = np.cumsum(y_sorted == 0) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))
