"""Discrimination metrics used for development and monitoring: AUC, Gini, KS.

All functions take y_true (1 = default) and a risk score where HIGHER = RISKIER
(i.e. the PD), and return a float (NaN if only one class is present).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y_true, pd_hat) -> float:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, pd_hat))


def gini(y_true, pd_hat) -> float:
    """Gini = 2*AUC - 1 (the 'accuracy ratio' banks quote)."""
    return 2 * auc(y_true, pd_hat) - 1


def ks(y_true, pd_hat) -> float:
    """Kolmogorov-Smirnov: max gap between cumulative bad and good distributions."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return float(np.max(tpr - fpr))


def summary(y_true, pd_hat) -> dict:
    return {
        "auc": auc(y_true, pd_hat),
        "gini": gini(y_true, pd_hat),
        "ks": ks(y_true, pd_hat),
        "n": int(len(y_true)),
        "default_rate": float(np.mean(y_true)),
    }
