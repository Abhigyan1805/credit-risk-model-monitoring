"""Filesystem layout. Every component resolves paths through here (see docs/CONTRACTS.md)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_LOANS = REPO_ROOT / "tests" / "fixtures" / "loans_sample.parquet"


def data_dir() -> Path:
    """Root of all generated data. Override with CRM_DATA_DIR (``make demo`` uses data/demo)."""
    return Path(os.environ.get("CRM_DATA_DIR", REPO_ROOT / "data")).resolve()


def raw_dir() -> Path:
    return Path(os.environ.get("CRM_RAW_DIR", data_dir() / "raw"))


def prepared_loans() -> Path:
    return data_dir() / "prepared" / "loans.parquet"


def model_dir() -> Path:
    return data_dir() / "model"


def scored_path(vintage: str) -> Path:
    """``vintage`` is ``YYYY-MM`` or the literal ``dev``."""
    return data_dir() / "scored" / f"vintage={vintage}.parquet"


def monitoring_path(month: str) -> Path:
    return data_dir() / "monitoring" / "results" / f"month={month}.parquet"


def monitoring_dir() -> Path:
    return data_dir() / "monitoring" / "results"


def commentary_path(month: str) -> Path:
    return data_dir() / "commentary" / f"month={month}.json"


def reports_dir() -> Path:
    return data_dir() / "reports"


def tableau_dir() -> Path:
    return data_dir() / "tableau"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
