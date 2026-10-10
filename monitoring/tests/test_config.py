"""Validation tests for the threshold config loader."""

import yaml
import pytest

from monitoring.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_path_exists():
    assert DEFAULT_CONFIG_PATH.exists()


def test_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"psi": {"green_below": 0.1, "amber_below": 0.25}}))
    with pytest.raises(ValueError, match="missing required sections"):
        load_config(bad)


def test_inverted_psi_thresholds_raise(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["psi"]["green_below"] = 0.30
    raw["psi"]["amber_below"] = 0.25
    bad = tmp_path / "inverted.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="green_below"):
        load_config(bad)


def test_inverted_hl_thresholds_raise(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["calibration"]["hl"]["amber_pvalue_at_or_above"] = 0.9
    bad = tmp_path / "hl.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="hl"):
        load_config(bad)
