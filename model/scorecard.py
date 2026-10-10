"""Logistic-regression scorecard built on WoE-binned features.

Pipeline (what a bank scorecard developer does):
 1. Bin every candidate feature and compute WoE / IV   (woe.py)
 2. Keep features with IV >= min_iv                      (weak ones add noise)
 3. Drop one of any pair of features whose WoE values correlate > max_corr
    (keeps the lower-IV one out; avoids unstable, double-counted coefficients)
 4. Fit logistic regression  logit(PD) = b0 + sum_i b_i * WoE_i
    and drop any feature whose coefficient has the "wrong" sign (WoE is
    good-vs-bad, so every b_i must be negative) until all signs are sensible.
 5. Convert to points. With odds = good:bad,
        score = offset + factor * ln(odds),  factor = PDO / ln 2
    so +PDO points = doubling of good:bad odds. Defaults: 600 points at 50:1
    odds, PDO = 20. Each bin's points = -(b_i * WoE_bin + b0/n) * factor + offset/n,
    and the loan's score is the sum of its bin points (identical to the
    logistic model, just easier to read and audit).
 6. Score bands 1 (safest) .. 10 (riskiest): fixed PD cut-points (the dev PD
    deciles) taken once from the development training data, so a band means the
    same thing in every later month - this is what monitoring relies on.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import schema as S
from .woe import fit_binners, iv_table

N_BANDS = 10  # CONTRACTS.md: 10 bands from dev PD deciles, 1 = safest ... 10 = riskiest


class Scorecard:
    def __init__(
        self,
        numeric=None,
        categorical=None,
        min_iv=0.02,
        max_corr=0.7,
        C=1.0,
        base_score=600,
        base_odds=50.0,
        pdo=20.0,
    ):
        self.numeric = list(numeric if numeric is not None else S.NUMERIC_FEATURES)
        self.categorical = list(categorical if categorical is not None else S.CATEGORICAL_FEATURES)
        self.min_iv, self.max_corr, self.C = min_iv, max_corr, C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    # ---- fit -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        num = [c for c in self.numeric if c in X.columns]
        cat = [c for c in self.categorical if c in X.columns]
        self.binners_ = fit_binners(X, y, num, cat)
        self.iv_ = iv_table(self.binners_)

        cand = list(self.iv_.loc[self.iv_["iv"] >= self.min_iv, "feature"])
        W = pd.DataFrame({c: self.binners_[c].transform(X[c]) for c in cand})
        cand = self._drop_correlated(W, cand)
        while True:
            lr = LogisticRegression(C=self.C, max_iter=500).fit(W[cand], y)
            wrong = [c for c, b in zip(cand, lr.coef_[0], strict=True) if b > 0]
            if not wrong or len(cand) == 1:
                break
            drop = min(wrong, key=lambda c: self.binners_[c].iv_)
            cand.remove(drop)
        self.features_, self.lr_ = cand, lr
        self.coef_ = dict(zip(cand, lr.coef_[0], strict=True))
        self.intercept_ = float(lr.intercept_[0])

        pds = self.predict_pd(X)
        self.band_cuts_ = [float(c) for c in np.quantile(pds, np.linspace(0.1, 0.9, N_BANDS - 1))]
        return self

    def _drop_correlated(self, W, cand):
        corr = W[cand].corr().abs()
        keep = list(cand)  # cand is sorted by IV descending
        for i, a in enumerate(cand):
            if a not in keep:
                continue
            for b in cand[i + 1 :]:
                if b in keep and corr.loc[a, b] > self.max_corr:
                    keep.remove(b)
        return keep

    # ---- predict -------------------------------------------------------
    def _woe_matrix(self, X):
        return pd.DataFrame({c: self.binners_[c].transform(X[c]) for c in self.features_}, index=X.index)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self._woe_matrix(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points = offset + factor * ln(good:bad odds) = offset - factor * logit(PD)."""
        W = self._woe_matrix(X)
        logit = self.intercept_ + sum(self.coef_[c] * W[c] for c in self.features_)
        return (self.offset - self.factor * logit).values

    def band(self, pd_hat) -> np.ndarray:
        """Band 1 (safest) .. 10 (riskiest) from the frozen dev PD decile cut-points."""
        return np.searchsorted(self.band_cuts_, np.asarray(pd_hat), side="right").astype("int64") + 1

    # ---- explainability -----------------------------------------------
    def points_table(self) -> pd.DataFrame:
        """One row per (feature, bin) with WoE, bad rate and scorecard points."""
        n = len(self.features_)
        rows = []
        for c in self.features_:
            t = self.binners_[c].table_
            for label, r in t.iterrows():
                pts = -(self.coef_[c] * r["woe"] + self.intercept_ / n) * self.factor + self.offset / n
                rows.append(
                    {
                        "feature": c,
                        "bin": label,
                        "n": int(r["n"]),
                        "bad_rate": r["bad_rate"],
                        "woe": r["woe"],
                        "points": pts,
                    }
                )
        return pd.DataFrame(rows)
