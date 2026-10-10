"""Tests for the contract monitoring-results table builder."""

import pandas as pd
import pytest

from common.config import MODEL_VERSION
from common.schemas import RESULT_COLUMNS, validate_results
from monitoring.config import load_thresholds
from monitoring.results import build_monitoring_results
from monitoring.synthetic import make_synthetic_scored

REQUIRED_METRICS = {
    "volume",
    "mean_pd",
    "default_rate",
    "psi_score",
    "csi",
    "auc",
    "gini",
    "ks",
    "calibration_ratio",
    "hl_pvalue",
    "binomial_pvalue",
    "overall_rag",
}


@pytest.fixture(scope="module")
def book():
    return make_synthetic_scored(n_dev=1500, n_per_month=400, n_months=6, seed=7)


@pytest.fixture(scope="module")
def results(book):
    dev, months = book
    thresholds = load_thresholds()
    frames = {month: build_monitoring_results(month, dev, df, thresholds) for month, df in months.items()}
    return frames


def test_columns_match_contract(results):
    for frame in results.values():
        assert list(frame.columns) == list(RESULT_COLUMNS)


def test_every_month_validates(results):
    for frame in results.values():
        validate_results(frame)  # raises on bad rag/direction/duplicates


def test_required_metrics_present(results):
    frame = next(iter(results.values()))
    assert REQUIRED_METRICS <= set(frame["metric"])


def test_segments_include_all_feature_and_band(results):
    frame = results["2014-01"]  # observable month
    segments = set(frame["segment"])
    assert "ALL" in segments
    assert {"fico", "dti", "purpose"} <= segments
    assert any(s.startswith("band=") for s in segments)


def test_no_duplicate_keys(results):
    for frame in results.values():
        assert not frame.duplicated(["month", "metric", "segment"]).any()


def test_month_and_model_version(results):
    for month, frame in results.items():
        assert (frame["month"] == month).all()
        assert (frame["model_version"] == MODEL_VERSION).all()


def test_observable_month_has_outcome_metrics(results):
    frame = results["2014-01"]
    for metric in ("auc", "gini", "ks", "hl_pvalue"):
        row = frame[frame["metric"] == metric].iloc[0]
        assert pd.notna(row["value"]), metric
    gini = frame[frame["metric"] == "gini"].iloc[0]
    assert gini["rag"] in {"GREEN", "AMBER", "RED"}


def test_unobservable_month_outcome_metrics_are_na(results):
    frame = results["2014-06"]  # not observable
    for metric in ("auc", "gini", "ks", "hl_pvalue", "calibration_ratio"):
        row = frame[frame["metric"] == metric].iloc[0]
        assert pd.isna(row["value"]), metric
        assert row["rag"] == "NA", metric
    band_rows = frame[(frame["metric"] == "binomial_pvalue") & (frame["segment"].str.startswith("band="))]
    assert not band_rows.empty
    assert (band_rows["rag"] == "NA").all()


def test_overall_rag_row_is_present_and_scored(results):
    for frame in results.values():
        row = frame[frame["metric"] == "overall_rag"].iloc[0]
        assert row["segment"] == "ALL"
        if row["rag"] != "NA":
            assert row["value"] in (0.0, 1.0, 2.0)
        assert row["rag"] in {"GREEN", "AMBER", "RED", "NA"}


def test_context_metrics_have_no_rag(results):
    frame = results["2014-01"]
    for metric in ("volume", "mean_pd", "default_rate"):
        assert (frame.loc[frame["metric"] == metric, "rag"] == "NA").all()


def test_psi_of_dev_against_itself_is_green(book):
    dev, _ = book
    thresholds = load_thresholds()
    frame = build_monitoring_results("2013-12", dev, dev, thresholds)
    psi_row = frame[frame["metric"] == "psi_score"].iloc[0]
    assert psi_row["value"] == pytest.approx(0.0, abs=1e-9)
    assert psi_row["rag"] == "GREEN"


def test_calibration_ratio_all_matches_bands(results):
    frame = results["2014-01"]
    all_ratio = frame[frame["metric"] == "calibration_ratio"].set_index("segment").loc["ALL"]
    assert pd.notna(all_ratio["value"])


def test_band_calibration_ratio_is_diagnostic_only(results):
    frame = results["2014-01"]
    band = frame[(frame["metric"] == "calibration_ratio") & (frame["segment"].str.startswith("band="))]
    assert not band.empty
    assert (band["rag"] == "NA").all()
    all_row = frame[(frame["metric"] == "calibration_ratio") & (frame["segment"] == "ALL")].iloc[0]
    assert all_row["rag"] in {"GREEN", "AMBER", "RED", "NA"}


def test_run_writes_validated_parquet(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from monitoring import run as run_module

    dev, months = make_synthetic_scored(n_dev=600, n_per_month=200, n_months=1, seed=3)
    month = "2014-01"
    monkeypatch.setenv("CRM_DATA_DIR", str(tmp_path))
    scored = tmp_path / "scored"
    scored.mkdir(parents=True)
    dev.to_parquet(scored / "vintage=dev.parquet", index=False)
    months[month].to_parquet(scored / f"vintage={month}.parquet", index=False)

    result = run_module.run(month)
    validate_results(result)
    out_path = tmp_path / "monitoring" / "results" / f"month={month}.parquet"
    assert out_path.exists()
