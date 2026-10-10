"""Read monitoring results and scored vintages from the Parquet contract.

Paths are resolved only through :mod:`common.paths` (see ``docs/CONTRACTS.md``
sections 4-5). Nothing here writes data.

The monitoring results are a long table: one row per
``(month, metric, segment)``. The scored vintages are one row per loan and are
only needed for the score-distribution extract (and the per-band counts).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from common import paths
from common.config import monitor_months
from common.schemas import validate_results


def month_date(month: str) -> str:
    """Convert ``"YYYY-MM"`` to the ISO date ``"YYYY-MM-01"`` used in extracts."""
    return f"{month}-01"


def _months_on_disk() -> list[str]:
    directory: Path = paths.monitoring_dir()
    if not directory.exists():
        return []
    return sorted(
        p.name.removeprefix("month=").removesuffix(".parquet") for p in directory.glob("month=*.parquet")
    )


def available_months(up_to: str | None = None) -> list[str]:
    """Months that have a monitoring-results file, oldest first.

    Falls back to the configured monitoring window only to raise a helpful
    error when nothing has been produced yet. ``up_to`` (``"YYYY-MM"``) keeps
    only months at or before that month, which is how the monthly report shows
    the trend without leaking future vintages.
    """
    months = _months_on_disk()
    if up_to is not None:
        months = [m for m in months if m <= up_to]
    if not months:
        expected = "none found on disk"
        raise FileNotFoundError(
            f"No monitoring results under {paths.monitoring_dir()} "
            f"(up_to={up_to}, expected months {monitor_months()[0]}..{monitor_months()[-1]}; {expected}). "
            "Run `make monitor` first, or point CRM_DATA_DIR at a populated data root."
        )
    return months


def load_results(
    months: Iterable[str] | None = None,
    up_to: str | None = None,
) -> pd.DataFrame:
    """Load and validate monitoring results for several months.

    Returns a long DataFrame in the contract schema (``RESULT_COLUMNS``) with an
    added ``month_date`` column, sorted by ``(month_date, metric, segment)``.
    Each file is validated so a malformed producer file fails loudly here rather
    than silently producing a wrong report.
    """
    if months is None:
        months = available_months(up_to)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for month in months:
        path = paths.monitoring_path(month)
        if not path.exists():
            missing.append(month)
            continue
        frame = pd.read_parquet(path)
        validate_results(frame)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No monitoring results found for months {list(months)}")
    if missing:
        # A partial window is still useful (e.g. mid-run); the report notes the range.
        print(f"[reporting] warning: missing monitoring results for {missing}")
    results = pd.concat(frames, ignore_index=True)
    results["month_date"] = results["month"].map(month_date)
    return results.sort_values(["month_date", "metric", "segment"]).reset_index(drop=True)


def load_scored_bands(months: Iterable[str]) -> pd.DataFrame:
    """Load ``(issue_month, score_band)`` from each scored vintage, if present.

    Returns an empty DataFrame with the right columns when no scored vintages
    exist, so callers can degrade gracefully instead of crashing.
    """
    frames: list[pd.DataFrame] = []
    for month in months:
        path = paths.scored_path(month)
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["issue_month", "score_band"]))
    if not frames:
        return pd.DataFrame(
            {"issue_month": pd.Series(dtype="string"), "score_band": pd.Series(dtype="int64")}
        )
    return pd.concat(frames, ignore_index=True)


def load_dev_band_shares() -> pd.Series | None:
    """Reference share of each score band, from the ``dev`` scored vintage.

    ``None`` when the dev vintage has not been scored yet.
    """
    path = paths.scored_path("dev")
    if not path.exists():
        return None
    bands = pd.read_parquet(path, columns=["score_band"])["score_band"]
    if bands.empty:
        return None
    return bands.value_counts(normalize=True).sort_index()
