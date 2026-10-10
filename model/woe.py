"""Weight-of-evidence (WoE) binning and information value (IV).

Plain-English idea: chop each feature into a few bins, and replace each bin by
    WoE = ln( share of all goods in the bin / share of all bads in the bin ).
Positive WoE = safer than average, negative = riskier. A logistic regression on
WoE values is then a "scorecard": every bin earns points. IV summarises how well
a whole feature separates goods from bads:
    IV = sum over bins of (good% - bad%) * WoE
Rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
>0.5 suspiciously strong (check for leakage).

Numeric features: quantile bins, tiny bins merged, then adjacent bins merged
until WoE is monotonic in the feature (banks require a sensible, explainable
direction). Missing values get their own bin. Categorical features: one bin per
category, rare categories pooled into "OTHER". A 0.5 smoothing term avoids
log(0) for bins with no goods or bads.
"""

import numpy as np
import pandas as pd

MISSING = "MISSING"
OTHER = "OTHER"


def _woe_iv(good, bad, tot_good, tot_bad):
    g = (good + 0.5) / (tot_good + 0.5)
    b = (bad + 0.5) / (tot_bad + 0.5)
    woe = np.log(g / b)
    return woe, (g - b) * woe


class WoEBinner:
    """Fit bins + WoE for one feature. `y` is 1 for bad (default)."""

    def __init__(self, name, numeric=True, max_bins=8, min_bin_frac=0.03, monotonic=True):
        self.name, self.numeric = name, numeric
        self.max_bins, self.min_bin_frac, self.monotonic = max_bins, min_bin_frac, monotonic

    # ---- fitting -------------------------------------------------------
    def fit(self, x: pd.Series, y):
        y = np.asarray(y)
        self.tot_bad_, self.tot_good_ = y.sum(), len(y) - y.sum()
        if self.numeric:
            self._fit_numeric(pd.to_numeric(x, errors="coerce"), y)
        else:
            self._fit_categorical(x.astype("object"), y)
        return self

    def _stats(self, bins, y):
        df = pd.DataFrame({"bin": bins, "y": y})
        t = df.groupby("bin", observed=True)["y"].agg(["size", "sum"])
        t.columns = ["n", "bad"]
        t["good"] = t["n"] - t["bad"]
        return t

    def _fit_numeric(self, x, y):
        nn = x.notna().values
        xv = x[nn]
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
        edges = list(qs)
        n = len(y)
        while True:
            idx = np.searchsorted(edges, xv.values, side="right")
            t = self._stats(idx, y[nn])
            # all bins incl. empty-edge cases are indexed 0..len(edges)
            t = t.reindex(range(len(edges) + 1), fill_value=0)
            woe, _ = _woe_iv(t["good"].values, t["bad"].values, self.tot_good_, self.tot_bad_)
            small = t["n"].values < self.min_bin_frac * n
            drop = None
            if len(edges) > 0 and small.any():
                i = int(np.argmin(np.where(small, t["n"].values, np.inf)))
                k = len(edges)  # bins 0..k; edge j separates bin j and j+1
                if i == 0:
                    drop = 0
                elif i == k:
                    drop = k - 1
                else:  # merge with the smaller neighbour
                    drop = i - 1 if t["n"].values[i - 1] <= t["n"].values[i + 1] else i
            elif self.monotonic and len(edges) > 1:
                d = np.diff(woe)
                sign = 1 if woe[-1] >= woe[0] else -1
                bad_steps = np.where(d * sign < 0)[0]
                if len(bad_steps):
                    drop = int(bad_steps[np.argmin(np.abs(d[bad_steps]))])
            if drop is None:
                break
            edges.pop(drop)
        self.edges_ = edges
        bins = self._assign_numeric(x)
        self._finalise(bins, y)

    def _assign_numeric(self, x):
        out = np.full(len(x), MISSING, dtype=object)
        nn = x.notna().values
        idx = np.searchsorted(self.edges_, x[nn].values, side="right")
        out[nn] = [self._label(i) for i in idx]
        return out

    def _label(self, i):
        lo = "-inf" if i == 0 else f"{self.edges_[i - 1]:g}"
        hi = "inf" if i == len(self.edges_) else f"{self.edges_[i]:g}"
        return f"({lo}, {hi}]"

    def _fit_categorical(self, x, y):
        x = x.fillna(MISSING)
        freq = x.value_counts(normalize=True)
        self.keep_ = set(freq[freq >= self.min_bin_frac].index)
        self._finalise(self._assign_cat(x), y)

    def _assign_cat(self, x):
        x = x.astype("object").fillna(MISSING)
        return np.where(x.isin(self.keep_), x, OTHER)

    def _finalise(self, bins, y):
        t = self._stats(bins, y)
        t["woe"], t["iv_part"] = _woe_iv(t["good"].values, t["bad"].values, self.tot_good_, self.tot_bad_)
        t["bad_rate"] = t["bad"] / t["n"]
        if self.numeric:  # natural bin order, MISSING last
            order = [self._label(i) for i in range(len(self.edges_) + 1)] + [MISSING]
            t = t.reindex([o for o in order if o in t.index])
        self.table_ = t
        self.iv_ = float(t["iv_part"].sum())

    # ---- applying ------------------------------------------------------
    def bin_labels(self, x: pd.Series):
        if self.numeric:
            return self._assign_numeric(pd.to_numeric(x, errors="coerce"))
        return self._assign_cat(x)

    def transform(self, x: pd.Series) -> np.ndarray:
        """Map raw values to WoE. Unseen categories fall in OTHER (or 0 if none)."""
        lab = pd.Series(self.bin_labels(x))
        w = self.table_["woe"]
        return lab.map(w).fillna(0.0).values


def fit_binners(X: pd.DataFrame, y, numeric_cols, categorical_cols, **kw):
    binners = {}
    for c in numeric_cols:
        binners[c] = WoEBinner(c, numeric=True, **kw).fit(X[c], y)
    for c in categorical_cols:
        binners[c] = WoEBinner(c, numeric=False, **kw).fit(X[c], y)
    return binners


def iv_table(binners) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": list(binners), "iv": [b.iv_ for b in binners.values()]})
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
