"""Tests for the YAML threshold loader (config/thresholds.yaml)."""

import pytest
import yaml

from monitoring.config import DEFAULT_THRESHOLDS_PATH, load_thresholds

REQUIRED = ("psi_score", "csi", "gini", "ks", "hl_pvalue", "binomial_pvalue", "calibration_ratio")


def _base_raw() -> dict:
    return yaml.safe_load(DEFAULT_THRESHOLDS_PATH.read_text())


def test_default_thresholds_load():
    thresholds = load_thresholds()
    assert thresholds.for_metric("psi_score") == {
        "amber": 0.10,
        "red": 0.25,
        "direction": "higher_worse",
    }
    gini = thresholds.for_metric("gini")
    assert gini["mode"] == "relative_drop"
    assert gini["direction"] == "lower_worse"
    assert gini["rel_amber"] == 0.10 and gini["rel_red"] == 0.25
    assert thresholds.for_metric("calibration_ratio")["amber_dev"] == 0.15
    assert thresholds.for_metric("auc") is None


def test_missing_metric_raises(tmp_path):
    raw = _base_raw()
    del raw["psi_score"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="missing required metrics"):
        load_thresholds(path)


def test_bad_direction_raises(tmp_path):
    raw = _base_raw()
    raw["csi"]["direction"] = "sideways"
    path = tmp_path / "bad_direction.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="direction"):
        load_thresholds(path)


def test_higher_worse_ordering_raises(tmp_path):
    raw = _base_raw()
    raw["psi_score"] = {"amber": 0.30, "red": 0.25, "direction": "higher_worse"}
    path = tmp_path / "bad_order.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="amber < red"):
        load_thresholds(path)


def test_lower_worse_ordering_raises(tmp_path):
    raw = _base_raw()
    raw["gini"] = {"amber": 0.30, "red": 0.40, "direction": "lower_worse"}
    path = tmp_path / "bad_lower.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="red < amber"):
        load_thresholds(path)


def test_relative_drop_ordering_raises(tmp_path):
    raw = _base_raw()
    raw["gini"] = {
        "mode": "relative_drop",
        "rel_amber": 0.30,
        "rel_red": 0.25,
        "direction": "lower_worse",
    }
    path = tmp_path / "bad_rel.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="rel_amber < rel_red"):
        load_thresholds(path)


def test_relative_drop_absolute_floor_ordering_raises(tmp_path):
    raw = _base_raw()
    raw["ks"] = {
        "mode": "relative_drop",
        "rel_amber": 0.10,
        "rel_red": 0.25,
        "abs_amber": 0.10,
        "abs_red": 0.20,
        "direction": "lower_worse",
    }
    path = tmp_path / "bad_floor.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="abs_red < abs_amber"):
        load_thresholds(path)


def test_relative_drop_needs_both_floors(tmp_path):
    raw = _base_raw()
    raw["ks"] = {
        "mode": "relative_drop",
        "rel_amber": 0.10,
        "rel_red": 0.25,
        "abs_amber": 0.15,
        "direction": "lower_worse",
    }
    path = tmp_path / "one_floor.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="both abs_amber and abs_red"):
        load_thresholds(path)


def test_calibration_dev_ordering_raises(tmp_path):
    raw = _base_raw()
    raw["calibration_ratio"] = {"amber_dev": 0.40, "red_dev": 0.30, "direction": "higher_worse"}
    path = tmp_path / "bad_cal.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="amber_dev must be < red_dev"):
        load_thresholds(path)
