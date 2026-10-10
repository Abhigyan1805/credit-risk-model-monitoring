"""Tests for the YAML-configurable traffic-light rules."""

import pytest

from monitoring.alerts import (
    band_calibration_band,
    band_csi,
    band_gini_drop,
    band_hl_pvalue,
    band_ks_drop,
    band_psi,
    overall_rag,
)
from monitoring.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_default_config_loads(config):
    assert config.psi["green_below"] == pytest.approx(0.10)
    assert config.psi["amber_below"] == pytest.approx(0.25)


def test_psi_bands(config):
    assert band_psi(0.05, config) == "green"
    assert band_psi(0.099, config) == "green"
    assert band_psi(0.10, config) == "amber"
    assert band_psi(0.24, config) == "amber"
    assert band_psi(0.25, config) == "red"
    assert band_psi(0.9, config) == "red"


def test_csi_bands_match_psi_defaults(config):
    assert band_csi(0.05, config) == "green"
    assert band_csi(0.15, config) == "amber"
    assert band_csi(0.30, config) == "red"


def test_gini_drop_relative_bands(config):
    baseline = 0.50
    assert band_gini_drop(baseline, 0.50, config) == "green"   # 0% drop
    assert band_gini_drop(baseline, 0.46, config) == "green"   # 8% drop
    assert band_gini_drop(baseline, 0.42, config) == "amber"   # 16% drop
    assert band_gini_drop(baseline, 0.38, config) == "red"     # 24% drop
    assert band_gini_drop(baseline, 0.60, config) == "green"   # improvement


def test_gini_drop_na_when_baseline_nonpositive(config):
    assert band_gini_drop(0.0, 0.2, config) == "na"


def test_ks_drop_absolute_bands(config):
    baseline = 0.40
    assert band_ks_drop(baseline, 0.40, config) == "green"   # 0 drop
    assert band_ks_drop(baseline, 0.36, config) == "green"   # 0.04 drop
    assert band_ks_drop(baseline, 0.34, config) == "amber"   # 0.06 drop
    assert band_ks_drop(baseline, 0.25, config) == "red"     # 0.15 drop


def test_hl_pvalue_bands_are_inverted(config):
    assert band_hl_pvalue(0.50, config) == "green"
    assert band_hl_pvalue(0.05, config) == "green"
    assert band_hl_pvalue(0.02, config) == "amber"
    assert band_hl_pvalue(0.01, config) == "amber"
    assert band_hl_pvalue(0.001, config) == "red"


def test_calibration_band_small_sample_ignored(config):
    # n below min_band_size -> not alerted
    assert band_calibration_band(0.5, 0.001, n=5, config=config) == "na"


def test_calibration_band_significant_is_red(config):
    assert band_calibration_band(0.02, 0.0001, n=200, config=config) == "red"


def test_calibration_band_borderline_is_amber(config):
    assert band_calibration_band(0.005, 0.03, n=200, config=config) == "amber"


def test_calibration_band_deviation_guard(config):
    # p-value not significant but the deviation is large -> flagged
    assert band_calibration_band(0.12, 0.5, n=200, config=config) == "red"


def test_overall_rag_takes_worst():
    assert overall_rag(["green", "amber", "red"]) == "red"
    assert overall_rag(["green", "amber"]) == "amber"
    assert overall_rag(["green", "green"]) == "green"


def test_overall_rag_ignores_na():
    assert overall_rag(["na", "green", "na"]) == "green"
    assert overall_rag(["na", "na"]) == "na"
