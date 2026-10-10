"""Tests for the Tableau extract builders and the data loaders."""

import pytest

from reporting.data import (
    available_months,
    load_dev_band_shares,
    load_results,
    load_scored_bands,
    month_date,
)
from reporting.tables import (
    LONG_COLUMNS,
    build_breaches,
    build_calibration_by_band,
    build_csi_by_feature,
    build_extracts,
    build_long_table,
    build_monthly_summary,
    build_score_distribution,
)
from reporting.tests import _synth


def test_month_date():
    assert month_date("2014-06") == "2014-06-01"


def test_available_months_and_load(crm_data):
    months = available_months()
    assert months == _synth.MONTHS
    assert load_results(up_to="2014-03")["month"].unique().tolist() == _synth.MONTHS[:3]


def test_load_results_validates_and_adds_month_date(results):
    assert "month_date" in results.columns
    assert results["month_date"].iloc[0].endswith("-01")


def test_long_table_columns(results):
    table = build_long_table(results)
    assert list(table.columns) == LONG_COLUMNS
    assert table["month_date"].str.endswith("-01").all()


def test_monthly_summary(results):
    summary = build_monthly_summary(results)
    assert len(summary) == len(_synth.MONTHS)
    assert summary["overall_rag"].iloc[-1] == "RED"  # series deteriorates
    assert {"psi_score", "csi_max", "gini", "ks", "calibration_ratio", "hl_pvalue"} <= set(summary.columns)
    assert summary["month_date"].str.endswith("-01").all()


def test_score_distribution_shares_sum_to_one(results):
    bands = load_scored_bands(_synth.MONTHS)
    dist = build_score_distribution(bands, load_dev_band_shares())
    for _, group in dist.groupby("month"):
        assert group["share"].sum() == pytest.approx(1.0)
    assert dist["share_delta"].notna().all()


def test_csi_by_feature(results):
    csi = build_csi_by_feature(results)
    assert set(csi["feature"]) == set(_synth.FEATURES)
    assert set(csi["month"]) == set(_synth.MONTHS)


def test_calibration_by_band(results):
    calib = build_calibration_by_band(results)
    assert sorted(calib["band"].unique().tolist()) == list(range(1, 11))
    assert {"calibration_ratio", "binomial_pvalue", "rag_calibration_ratio"} <= set(calib.columns)


def test_breaches_only_amber_red(results):
    breaches = build_breaches(results)
    assert not breaches.empty
    assert set(breaches["rag"]) <= {"AMBER", "RED"}
    assert breaches["message"].str.len().gt(0).all()


def test_build_extracts_keys(results):
    bands = load_scored_bands(_synth.MONTHS)
    extracts = build_extracts(results, bands)
    assert set(extracts) == {
        "monitoring_results",
        "monthly_summary",
        "score_distribution",
        "csi_by_feature",
        "calibration_by_band",
        "breaches",
    }
    assert all(not df.empty for df in extracts.values())
