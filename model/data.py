"""Helpers to select model rows from the prepared loans table (see docs/CONTRACTS.md section 2)."""

import numpy as np
import pandas as pd


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived feature: loan size relative to income (blank if income is 0/missing)."""
    df = df.copy()
    income = df["annual_inc"].astype("float64").replace(0, np.nan)
    df["loan_to_income"] = df["loan_amnt"].astype("float64") / income
    # pandas string/object columns -> plain object so the binners treat missing uniformly
    for c in df.select_dtypes(include=["string", "category"]).columns:
        df[c] = df[c].astype("object")
    return df


def window(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Rows with issue_month in [start, end] (inclusive 'YYYY-MM' strings)."""
    m = df["issue_month"].astype(str)
    keep = pd.Series(True, index=df.index)
    if start:
        keep &= m >= start
    if end:
        keep &= m <= end
    return df[keep]


def observed(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose default outcome is known (default_flag not NA)."""
    return df[df["default_flag"].notna()]
