"""Shared fixtures: a populated synthetic data root for reporting tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reporting.data import load_results
from reporting.tests import _synth


@pytest.fixture()
def crm_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``CRM_DATA_DIR`` at a fresh synthetic data root and return it."""
    root = tmp_path / "data"
    monkeypatch.setenv("CRM_DATA_DIR", str(root))
    monkeypatch.setenv("CRM_RAW_DIR", str(root / "raw"))
    _synth.write_synthetic_data(root)
    return root


@pytest.fixture()
def results(crm_data: Path) -> pd.DataFrame:
    """The synthetic monitoring results as the long DataFrame."""
    return load_results()
