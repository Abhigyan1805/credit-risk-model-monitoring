"""Train the PD scorecard and write the model, metadata and the dev reference scoring.

    python -m model.train

Reads ``paths.prepared_loans()``. Training sample = dev-window loans issued DEV_START..TRAIN_END with a
known outcome. Out-of-time (OOT) validation = dev-window loans issued AFTER TRAIN_END (later months than
anything the model saw). Writes under ``paths.model_dir()``:
  pd_model.joblib        the fitted Scorecard (binning + coefficients + frozen band cut-points)
  metadata.json          model_version, features, windows, coefficients, band cut-points, AUC/Gini/KS
                         (train, out-of-time and the gradient-boosted challenger)
  scorecard_points.csv   per-bin WoE / bad rate / points table (the human-readable scorecard)
  iv_table.csv           information value of every candidate feature
and ``paths.scored_path("dev")``: the OOT dev sample scored by the model = monitoring reference population.
"""

import argparse
import json
from datetime import UTC, datetime

import joblib
import pandas as pd

from common import config, paths
from model.challenger import Challenger
from model.data import add_derived, observed, window
from model.metrics import summary
from model.score import score_frame
from model.scorecard import Scorecard


def _eval(model, df) -> dict:
    d = observed(df)
    return summary(d["default_flag"].astype(int).values, model.predict_pd(d))


def train(df: pd.DataFrame, challenger: bool = True, seed: int = 0):
    """Fit on the training window; returns (scorecard, metadata, oot_frame, train_frame)."""
    df = add_derived(df)
    dev = window(df, config.DEV_START, config.DEV_END)
    tr = observed(window(dev, config.DEV_START, config.TRAIN_END))
    oot = window(dev, _next_month(config.TRAIN_END), config.DEV_END)

    sc = Scorecard().fit(tr, tr["default_flag"].astype(int))
    metrics = {"train": _eval(sc, tr), "out_of_time": _eval(sc, oot)}
    if challenger:
        ch = Challenger(sc.numeric, sc.categorical, seed).fit(tr, tr["default_flag"].astype(int))
        metrics["challenger_train"] = _eval(ch, tr)
        metrics["challenger_out_of_time"] = _eval(ch, oot)
    meta = {
        "model_version": config.MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_window": [config.DEV_START, config.TRAIN_END],
        "oot_window": [_next_month(config.TRAIN_END), config.DEV_END],
        "features": sc.features_,
        "coefficients": sc.coef_,
        "intercept": sc.intercept_,
        "score_scaling": {"base_score": 600, "base_odds_good_to_bad": 50, "pdo": 20},
        "band_cut_points_pd": sc.band_cuts_,
        "metrics": metrics,
    }
    return sc, meta, oot, tr


def _next_month(m: str) -> str:
    return (pd.Period(m, "M") + 1).strftime("%Y-%m")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = pd.read_parquet(paths.prepared_loans())
    sc, meta, oot, _ = train(df)
    out = paths.model_dir()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc, out / "pd_model.joblib")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    sc.points_table().to_csv(out / "scorecard_points.csv", index=False)
    sc.iv_.to_csv(out / "iv_table.csv", index=False)
    ref = score_frame(oot, sc, outcome_observable=True)
    ref.to_parquet(paths.ensure_parent(paths.scored_path("dev")), index=False)
    print(json.dumps(meta["metrics"], indent=2))
    print(f"model -> {out}; dev reference ({len(ref)} loans) -> {paths.scored_path('dev')}")


if __name__ == "__main__":
    main()
