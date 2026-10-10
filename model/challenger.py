"""Gradient-boosted challenger (HistGradientBoostingClassifier) on the raw features.

Only used for comparison: it shows how much discrimination the simple, explainable
scorecard gives up. Banks usually keep the scorecard as the champion because every
decision can be traced to bin points; the challenger is a benchmark.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder


class Challenger:
    def __init__(self, numeric, categorical, seed=0):
        self.numeric, self.categorical, self.seed = list(numeric), list(categorical), seed

    def _matrix(self, X):
        num = X[self.numeric].apply(pd.to_numeric, errors="coerce")
        cat = pd.DataFrame(
            self.enc_.transform(X[self.categorical].astype(str)), columns=self.categorical, index=X.index
        )
        return pd.concat([num, cat], axis=1)

    def fit(self, X, y):
        self.numeric = [c for c in self.numeric if c in X.columns]
        self.categorical = [c for c in self.categorical if c in X.columns]
        self.enc_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
        ).fit(X[self.categorical].astype(str))
        M = self._matrix(X)
        mask = [False] * len(self.numeric) + [True] * len(self.categorical)
        self.model_ = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=100,
            min_samples_leaf=200,
            l2_regularization=1.0,
            categorical_features=mask,
            random_state=self.seed,
        ).fit(M, np.asarray(y).astype(int))
        return self

    def predict_pd(self, X):
        return self.model_.predict_proba(self._matrix(X))[:, 1]
