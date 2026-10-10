"""Unit tests (synthetic data + the committed sample fixture; no dataset download)."""

import numpy as np
import pandas as pd
import pytest

from common import config, paths, schemas
from model import metrics, score, train
from model.data import add_derived
from model.synthetic import make_loans
from model.woe import WoEBinner


@pytest.fixture(scope="module")
def loans():
    return make_loans(n=12000, months=60, seed=1)


def test_woe_sign_monotone_and_iv():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=4000))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(x * 1.5 - 2)))).astype(int)  # high x => bad
    b = WoEBinner("x").fit(x, y)
    assert b.iv_ > 0.1
    assert np.all(np.diff(b.table_["woe"].values) <= 1e-9)
    assert b.transform(x).shape == (4000,)


def test_woe_missing_and_unseen_category():
    x = pd.Series(["a"] * 500 + ["b"] * 500 + [None] * 100)
    y = np.r_[np.zeros(450), np.ones(50), np.zeros(300), np.ones(200), np.zeros(80), np.ones(20)]
    b = WoEBinner("c", numeric=False).fit(x, y)
    assert "MISSING" in b.table_.index
    assert np.isfinite(b.transform(pd.Series(["zzz"]))).all()


def test_metrics_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1])
    assert metrics.gini(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.ks(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert np.isnan(metrics.auc([0, 0], [0.1, 0.2]))


def test_train_on_synthetic(loans):
    sc, meta, oot, _ = train.train(loans)
    m = meta["metrics"]
    assert m["train"]["auc"] > 0.65 and m["out_of_time"]["auc"] > 0.62
    assert (np.array(list(sc.coef_.values())) < 0).all()  # every driver has a sensible sign
    assert len(sc.band_cuts_) == 9 and sc.band_cuts_ == sorted(sc.band_cuts_)
    assert set(m) >= {"challenger_train", "challenger_out_of_time"}


def test_points_equal_model_score_and_odds(loans):
    sc, _, oot, _ = train.train(loans, challenger=False)
    d = add_derived(oot).head(300)
    pts = sc.points_table().set_index(["feature", "bin"])["points"]
    manual = np.zeros(len(d))
    for c in sc.features_:
        manual += np.array([pts.loc[(c, lab)] for lab in sc.binners_[c].bin_labels(d[c])])
    assert np.allclose(manual, sc.score(d))
    odds = (1 - sc.predict_pd(d)) / sc.predict_pd(d)
    assert np.allclose(sc.score(d), sc.offset + sc.factor * np.log(odds), atol=1e-6)


def test_bands_monotone_in_pd(loans):
    sc, _, oot, _ = train.train(loans, challenger=False)
    pd_hat = sc.predict_pd(add_derived(oot))
    b = sc.band(pd_hat)
    assert b.min() >= 1 and b.max() <= 10
    order = np.argsort(pd_hat)
    assert np.all(np.diff(b[order]) >= 0)


def test_score_frame_matches_contract(loans):
    sc, _, oot, _ = train.train(loans, challenger=False)
    out = score.score_frame(oot, sc, outcome_observable=True)
    schemas.validate_scored(out)
    assert out["pd"].between(0, 1).all() and out["default_flag"].notna().any()
    hidden = score.score_frame(oot, sc, outcome_observable=False)
    schemas.validate_scored(hidden)
    assert hidden["default_flag"].isna().all() and not hidden["outcome_observable"].any()


def test_end_to_end_on_sample_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("CRM_DATA_DIR", str(tmp_path))
    df = pd.read_parquet(paths.SAMPLE_LOANS)
    paths.ensure_parent(paths.prepared_loans())
    df.to_parquet(paths.prepared_loans())
    monkeypatch.setattr("sys.argv", ["x"])
    train.main()
    assert (paths.model_dir() / "pd_model.joblib").exists()
    dev = pd.read_parquet(paths.scored_path("dev"))
    schemas.validate_scored(dev)
    assert dev["issue_month"].min() > config.TRAIN_END
    for month, observable in [("2014-06", True), ("2016-03", False)]:
        monkeypatch.setattr("sys.argv", ["x", "--month", month])
        score.main()
        s = pd.read_parquet(paths.scored_path(month))
        schemas.validate_scored(s)
        assert (s["issue_month"] == month).all()
        assert s["default_flag"].notna().any() == observable
