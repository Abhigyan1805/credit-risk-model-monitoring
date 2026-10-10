"""End-to-end tests for the monitoring runner on synthetic data."""

import numpy as np
import pytest

from monitoring.__main__ import make_synthetic_book
from monitoring.config import load_config
from monitoring.runner import (
    LONG_COLUMNS,
    ModelSnapshot,
    build_baseline,
    run_monitoring,
)


@pytest.fixture(scope="module")
def book():
    return make_synthetic_book(n_baseline=1500, n_per_month=400, n_months=6, seed=7)


@pytest.fixture(scope="module")
def outputs(book):
    baseline, months = book
    return run_monitoring(baseline, months)


def test_output_keys(outputs):
    assert set(outputs) == {
        "monitoring_results",
        "monthly_summary",
        "calibration_detail",
        "alert_summary",
    }


def test_alert_summary_one_row_per_month(book, outputs):
    _, months = book
    summary = outputs["alert_summary"]
    assert len(summary) == len(months)
    assert list(summary.columns) == [
        "month",
        "n_red",
        "n_amber",
        "red_metrics",
        "amber_metrics",
        "overall_rag",
    ]
    assert summary["overall_rag"].tolist() == outputs["monthly_summary"]["overall_rag"].tolist()


def test_alert_summary_matches_breach_counts(outputs):
    results = outputs["monitoring_results"]
    alert = outputs["alert_summary"].set_index("month")
    for month, group in results.groupby("month"):
        assert alert.loc[month, "n_red"] == int((group["band"] == "red").sum())
        assert alert.loc[month, "n_amber"] == int((group["band"] == "amber").sum())


def test_monitoring_results_columns(outputs):
    assert list(outputs["monitoring_results"].columns) == LONG_COLUMNS


def test_one_summary_row_per_month(book, outputs):
    _, months = book
    summary = outputs["monthly_summary"]
    assert len(summary) == len(months)
    assert summary["month"].tolist() == list(months.keys())


def test_overall_rag_is_valid(outputs):
    summary = outputs["monthly_summary"]
    assert set(summary["overall_rag"]).issubset({"green", "amber", "red", "na"})


def test_expected_metrics_present(outputs):
    results = outputs["monitoring_results"]
    metrics = set(results["metric"])
    assert {"psi_score", "csi", "auc", "gini", "ks", "hl_pvalue", "worst_band"} <= metrics
    groups = set(results["metric_group"])
    assert groups == {"stability", "discrimination", "calibration"}


def test_calibration_detail_has_bands_per_month(book, outputs):
    _, months = book
    detail = outputs["calibration_detail"]
    assert not detail.empty
    assert set(detail["month"]) == set(months)
    assert {"band", "n", "mean_predicted_pd", "observed_default_rate", "band_rag"} <= set(detail.columns)


def test_drift_raises_psi_over_time(book, outputs):
    # later synthetic months drift more, so PSI must be larger than month 1
    summary = outputs["monthly_summary"].set_index("month")
    first = summary.loc["2024-01", "psi_score"]
    last = summary.loc["2024-06", "psi_score"]
    assert last > first


def test_alert_boolean_matches_band(outputs):
    results = outputs["monitoring_results"]
    reds = results[results["band"] == "red"]
    assert (reds["alert"]).all()
    non_reds = results[results["band"] != "red"]
    assert not non_reds["alert"].any()


def test_unseasoned_month_skips_discrimination():
    rng = np.random.default_rng(0)
    baseline = ModelSnapshot(
        scores=rng.uniform(0, 1, 800),
        labels=rng.binomial(1, 0.2, 800),
        features={"x": rng.normal(size=800)},
    )
    # a month with no labels (not yet seasoned)
    month = ModelSnapshot(scores=rng.uniform(0, 1, 200), labels=None, features={"x": rng.normal(size=200)})
    outputs = run_monitoring(baseline, {"2025-01": month})
    results = outputs["monitoring_results"]
    assert not (results["metric"] == "gini").any()
    assert outputs["monthly_summary"].loc[0, "calibration_available"] == False  # noqa: E712


def test_build_baseline_freezes_edges(book):
    baseline_snapshot, _ = book
    config = load_config()
    baseline = build_baseline(baseline_snapshot, config)
    assert baseline.score_edges[0] == -np.inf
    assert baseline.gini == pytest.approx(2 * baseline.auc - 1)
    assert "purpose" in baseline.feature_kinds
    assert baseline.feature_kinds["purpose"] == "categorical"
    assert baseline.feature_kinds["fico"] == "numeric"


def test_categorical_feature_produces_csi(outputs):
    results = outputs["monitoring_results"]
    csi_features = set(results.loc[results["metric"] == "csi", "feature"])
    assert {"fico", "dti", "loan_amnt", "purpose"} <= csi_features
