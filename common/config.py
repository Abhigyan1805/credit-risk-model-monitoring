"""Windows and constants shared by all components (the single source of truth)."""

from __future__ import annotations

import pandas as pd

# Scope: 36-month loans only, so a vintage issued up to OUTCOME_CUTOFF_MONTH has fully matured
# by the end of the data (2018-12) and its resolved outcomes are not biased by right-censoring.
TERM_MONTHS = 36
DEV_START, DEV_END = "2012-01", "2013-12"  # development window (train + out-of-time valid)
TRAIN_END = "2013-06"  # train: DEV_START..TRAIN_END, OOT valid: after
MONITOR_START, MONITOR_END = "2014-01", "2016-12"  # monthly monitoring vintages
OUTCOME_CUTOFF_MONTH = "2015-12"  # last vintage whose outcomes are observable
DEFAULT_STATUSES = ("Charged Off", "Default")
GOOD_STATUSES = ("Fully Paid",)
MODEL_VERSION = "pd_v1"

# Features whose drift (CSI) is monitored: the model builder must use a superset of these.
CSI_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico",
    "revol_util",
    "credit_history_years",
    "emp_length_years",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
]
RAG_VALUES = ("GREEN", "AMBER", "RED", "NA")


def month_range(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def monitor_months() -> list[str]:
    return month_range(MONITOR_START, MONITOR_END)


def outcome_observable(month: str) -> bool:
    return month <= OUTCOME_CUTOFF_MONTH
