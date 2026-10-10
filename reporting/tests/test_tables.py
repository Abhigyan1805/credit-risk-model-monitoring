"""Tests for the table builders and schema normalization."""

import json
import os

import pytest

from reporting.schema import MonitoringResults, normalize_results
from reporting.tables import (
    build_all_tables,
    build_calibration_table,
    build_fact_table,
    build_feature_csi_table,
    build_score_distribution_table,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_monitoring_results.json"
)


@pytest.fixture()
def results() -> MonitoringResults:
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return normalize_results(json.load(fh))


def test_normalize_rejects_empty_periods():
    with pytest.raises(ValueError):
        normalize_results({"periods": []})


def test_normalize_computes_period_rag(results):
    # First synthetic month is healthy; the last is clearly red.
    assert results.periods_sorted[0]["rag"] == "green"
    assert results.periods_sorted[-1]["rag"] == "red"


def test_normalize_ignores_non_numeric_reference():
    normalized = normalize_results(
        {
            "model": {"reference_metrics": {"auc": "0.7", "gini": "bad"}},
            "periods": [{"period": "2020-01", "metrics": {"psi": 0.01}}],
        }
    )
    assert normalized.reference_metrics == {"auc": 0.7}


def test_fact_table_shape(results):
    columns, rows = build_fact_table(results)
    assert columns[0] == "period"
    assert len(rows) == len(results.periods)
    assert set(rows[0]) == set(columns)
    # Chronological order.
    assert rows[0]["period"] < rows[-1]["period"]
    assert rows[-1]["rag_overall"] == "red"
    assert rows[0]["rag_overall"] == "green"


def test_score_distribution_share_delta(results):
    _, rows = build_score_distribution_table(results)
    assert rows
    for row in rows:
        assert row["share_delta"] == pytest.approx(row["share"] - row["expected_share"], abs=1e-6)


def test_calibration_gap_derived(results):
    _, rows = build_calibration_table(results)
    assert rows
    for row in rows:
        expected = abs(row["observed_pd"] - row["predicted_pd"])
        assert row["abs_gap"] == pytest.approx(expected, abs=1e-6)


def test_feature_csi_has_every_feature_every_period(results):
    _, rows = build_feature_csi_table(results)
    features = {r["feature"] for r in rows}
    periods = {r["period"] for r in rows}
    assert len(rows) == len(features) * len(periods)
    assert all(r["rag"] in {"green", "amber", "red"} for r in rows)


def test_build_all_tables_keys(results):
    tables = build_all_tables(results)
    assert set(tables) == {"fact", "score_distribution", "calibration_by_band", "feature_csi", "breaches"}
    for columns, rows in tables.values():
        assert isinstance(columns, list) and columns
        for row in rows:
            assert set(row) == set(columns)
