"""Tests for the traffic-light rag rules."""

import pytest

from monitoring.alerts import (
    AMBER,
    GREEN,
    NA,
    RED,
    rag,
    rag_score,
    thresholds_for,
    worst_rag,
)

HIGHER = {"amber": 0.10, "red": 0.25, "direction": "higher_worse"}
LOWER = {"amber": 0.05, "red": 0.01, "direction": "lower_worse"}
CAL = {"amber_dev": 0.15, "red_dev": 0.30, "direction": "higher_worse"}
REL = {"mode": "relative_drop", "rel_amber": 0.10, "rel_red": 0.25, "direction": "lower_worse"}
REL_ABS = {**REL, "abs_amber": 0.25, "abs_red": 0.20}


def test_higher_worse_boundaries():
    assert rag(0.05, HIGHER) == GREEN
    assert rag(0.099, HIGHER) == GREEN
    assert rag(0.10, HIGHER) == AMBER
    assert rag(0.249, HIGHER) == AMBER
    assert rag(0.25, HIGHER) == RED
    assert rag(0.9, HIGHER) == RED


def test_lower_worse_boundaries():
    assert rag(0.50, LOWER) == GREEN
    assert rag(0.05, LOWER) == AMBER  # not > amber
    assert rag(0.02, LOWER) == AMBER  # > red
    assert rag(0.01, LOWER) == RED  # not > red
    assert rag(0.001, LOWER) == RED


def test_calibration_ratio_is_two_sided():
    assert rag(1.0, CAL) == GREEN
    assert rag(1.10, CAL) == GREEN
    assert rag(0.90, CAL) == GREEN
    assert rag(1.20, CAL) == AMBER
    assert rag(0.80, CAL) == AMBER
    assert rag(1.35, CAL) == RED
    assert rag(0.60, CAL) == RED


def test_missing_values_and_specs_are_na():
    assert rag(float("nan"), HIGHER) == NA
    assert rag(None, HIGHER) == NA
    assert rag(0.9, None) == NA


def test_relative_drop_bands():
    baseline = 0.33
    assert rag(0.33, REL, baseline) == GREEN  # no drop
    assert rag(0.31, REL, baseline) == GREEN  # ~6% drop
    assert rag(0.28, REL, baseline) == AMBER  # ~15% drop
    assert rag(0.24, REL, baseline) == RED  # ~27% drop
    assert rag(0.40, REL, baseline) == GREEN  # improvement
    assert rag(0.30, REL, None) == NA  # baseline required


def test_relative_drop_absolute_floor_is_secondary():
    baseline = 0.33
    # relative drop is zero but the absolute level breaches the catastrophic floor
    assert rag(0.22, REL_ABS, 0.22) == AMBER
    # both a large relative drop and below the red floor -> red
    assert rag(0.18, REL_ABS, baseline) == RED


def test_thresholds_for_relative_drop_uses_baseline():
    amber, red = thresholds_for(REL, 0.33)
    assert amber == pytest.approx(0.33 * 0.90)
    assert red == pytest.approx(0.33 * 0.75)


def test_thresholds_for():
    assert thresholds_for(HIGHER) == (0.10, 0.25)
    assert thresholds_for(CAL) == (0.15, 0.30)
    amber, red = thresholds_for(None)
    assert amber != amber and red != red  # both NaN


def test_worst_rag():
    assert worst_rag([GREEN, AMBER, RED]) == RED
    assert worst_rag([GREEN, AMBER]) == AMBER
    assert worst_rag([GREEN, GREEN]) == GREEN
    assert worst_rag([NA, GREEN, NA]) == GREEN
    assert worst_rag([NA, NA]) == NA
    assert worst_rag([]) == NA


def test_rag_score():
    assert rag_score(GREEN) == 0.0
    assert rag_score(AMBER) == 1.0
    assert rag_score(RED) == 2.0
    assert rag_score(NA) != rag_score(NA)  # NaN
