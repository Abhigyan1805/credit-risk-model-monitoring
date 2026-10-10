"""Score one monthly vintage with the frozen model: ``python -m model.score --month YYYY-MM``

Scores loans with issue_month == month and writes ``paths.scored_path(month)`` (SCORED_COLUMNS):
loan_id, issue_month, pd, score, score_band (1 safest..10 riskiest), default_flag, outcome_observable,
model_version, plus every ``CSI_FEATURES`` column copied unchanged from the prepared table (raw, un-binned).
default_flag is filled only when ``config.outcome_observable(month)``; otherwise NA (late vintages are
biased towards early defaults, so their outcomes must not be used). Idempotent: overwrites only that file.
"""

import argparse
import json

import joblib
import pandas as pd

from common import config, paths
from model.data import add_derived, window


def score_frame(df: pd.DataFrame, model, outcome_observable: bool) -> pd.DataFrame:
    """Score a frame of prepared loans (derived features added here)."""
    d = add_derived(df)
    pd_hat = model.predict_pd(d)
    if outcome_observable:
        flag = df["default_flag"].astype("Int8").values
    else:
        flag = pd.array([pd.NA] * len(df), dtype="Int8")
    out = pd.DataFrame(
        {
            "loan_id": df["loan_id"].astype("string").values,
            "issue_month": df["issue_month"].astype("string").values,
            "pd": pd_hat,
            "score": model.score(d),
            "score_band": model.band(pd_hat),
            "default_flag": flag,
            "outcome_observable": outcome_observable,
            "model_version": config.MODEL_VERSION,
        }
    )
    for c in config.CSI_FEATURES:
        out[c] = df[c].values
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", required=True, help="vintage YYYY-MM")
    month = p.parse_args().month
    model = joblib.load(paths.model_dir() / "pd_model.joblib")
    meta = json.loads((paths.model_dir() / "metadata.json").read_text())
    assert meta["model_version"] == config.MODEL_VERSION
    df = pd.read_parquet(paths.prepared_loans())
    vint = window(df, month, month)
    if vint.empty:
        raise SystemExit(f"no loans issued in {month}")
    out = score_frame(vint, model, config.outcome_observable(month))
    out.to_parquet(paths.ensure_parent(paths.scored_path(month)), index=False)
    print(f"scored {len(out)} loans for {month} -> {paths.scored_path(month)}")


if __name__ == "__main__":
    main()
