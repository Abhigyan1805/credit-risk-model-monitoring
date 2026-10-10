"""Tests for RAG classification logic."""

import pytest

from reporting.rag import (
    classify_breaches,
    classify_metric,
    worst_status,
)


def test_psi_absolute_thresholds():
    assert classify_metric("psi", 0.05) == "green"
    assert classify_metric("psi", 0.10) == "amber"
    assert classify_metric("psi", 0.20) == "amber"
    assert classify_metric("psi", 0.25) == "red"
    assert classify_metric("psi", 0.99) == "red"


def test_missing_value_is_none():
    assert classify_metric("psi", None) is None
    assert classify_metric("unknown_metric", 0.5) is None


def test_auc_drop_relative_to_reference():
    ref = 0.72
    assert classify_metric("auc", 0.72, reference_value=ref) == "green"
    assert classify_metric("auc", 0.719, reference_value=ref) == "green"
    assert classify_metric("auc", 0.70, reference_value=ref) == "amber"
    assert classify_metric("auc", 0.66, reference_value=ref) == "red"
    # An improvement is never a breach.
    assert classify_metric("auc", 0.80, reference_value=ref) == "green"


def test_drop_metric_without_reference_is_unclassified():
    assert classify_metric("auc", 0.70) is None


def test_calibration_gap_is_increase_worse():
    assert classify_metric("calibration_gap", 0.01) == "green"
    assert classify_metric("calibration_gap", 0.03) == "amber"
    assert classify_metric("calibration_gap", 0.06) == "red"


def test_worst_status_ignores_missing():
    assert worst_status([None, None]) is None
    assert worst_status(["green", None, "amber"]) == "amber"
    assert worst_status(["green", "red", "amber"]) == "red"


def test_classify_breaches_reports_only_deterioration():
    metrics = {"psi": 0.30, "csi": 0.05, "auc": 0.65, "gini": 0.30, "ks": 0.30, "calibration_gap": 0.01}
    reference = {"auc": 0.72, "gini": 0.44, "ks": 0.335}
    breaches = classify_breaches(metrics, reference)
    levels = {b["metric"]: b["level"] for b in breaches}
    assert levels["psi"] == "red"
    assert levels["auc"] == "red"
    assert "csi" not in levels  # 0.05 is green
    assert "calibration_gap" not in levels
    assert all(b["message"] for b in breaches)


def test_classify_breaches_empty_when_all_healthy():
    metrics = {"psi": 0.02, "csi": 0.02, "auc": 0.72, "gini": 0.44, "ks": 0.335, "calibration_gap": 0.005}
    assert classify_breaches(metrics, {"auc": 0.72, "gini": 0.44, "ks": 0.335}) == []
