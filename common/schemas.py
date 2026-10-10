"""Column contracts for every Parquet interface. Mirror of docs/CONTRACTS.md; tests enforce it."""

from __future__ import annotations

import pandas as pd

from common.config import CSI_FEATURES, RAG_VALUES

# data/prepared/loans.parquet - one row per loan (ingest -> model)
PREPARED_COLUMNS: dict[str, str] = {
    "loan_id": "string",
    "issue_month": "string",  # YYYY-MM
    "window": "string",  # dev | monitor | other
    "default_flag": "Int8",  # 1 Charged Off/Default, 0 Fully Paid, <NA> if unresolved
    "loan_status": "string",  # raw LendingClub status
    "loan_amnt": "float64",
    "term_months": "int64",
    "int_rate": "float64",  # percent, e.g. 13.5
    "installment": "float64",
    "grade": "string",  # A..G
    "sub_grade": "string",  # A1..G5
    "emp_length_years": "float64",  # 0..10, NaN unknown
    "home_ownership": "string",
    "annual_inc": "float64",
    "verification_status": "string",
    "purpose": "string",
    "addr_state": "string",
    "dti": "float64",
    "delinq_2yrs": "float64",
    "fico": "float64",  # mean of fico_range_low/high
    "inq_last_6mths": "float64",
    "open_acc": "float64",
    "pub_rec": "float64",
    "revol_bal": "float64",
    "revol_util": "float64",  # percent
    "total_acc": "float64",
    "credit_history_years": "float64",  # issue date - earliest credit line
}

# data/scored/vintage=<YYYY-MM|dev>.parquet - one row per loan (model -> monitoring)
SCORED_CORE_COLUMNS: dict[str, str] = {
    "loan_id": "string",
    "issue_month": "string",  # YYYY-MM (the vintage); "dev" rows keep their real issue month
    "pd": "float64",  # predicted probability of default, (0,1)
    "score": "float64",  # points: higher = safer
    "score_band": "int64",  # 1 (safest) .. 10 (riskiest); cut-points frozen on dev
    "default_flag": "Int8",  # actual outcome, <NA> unless outcome_observable
    "outcome_observable": "bool",
    "model_version": "string",
}
# SCORED also carries every column in CSI_FEATURES, unchanged from the prepared table.
SCORED_COLUMNS = {**SCORED_CORE_COLUMNS, **{f: "object" for f in CSI_FEATURES}}

# data/monitoring/results/month=<YYYY-MM>.parquet - long format (monitoring -> reporting)
RESULT_COLUMNS: dict[str, str] = {
    "month": "string",
    "metric": "string",  # psi_score, csi, auc, gini, ks, calibration_ratio, hl_pvalue, ...
    "segment": "string",  # ALL, a feature name (csi), a band (band=3), grade=B ...
    "value": "float64",
    "amber_threshold": "float64",
    "red_threshold": "float64",
    "direction": "string",  # higher_worse | lower_worse
    "rag": "string",  # GREEN | AMBER | RED | NA
    "n_obs": "int64",
    "model_version": "string",
}


def check_columns(df: pd.DataFrame, spec: dict[str, str], name: str, extra_ok: bool = True) -> None:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if not extra_ok:
        extra = [c for c in df.columns if c not in spec]
        if extra:
            raise ValueError(f"{name}: unexpected columns {extra}")


def validate_prepared(df: pd.DataFrame) -> None:
    check_columns(df, PREPARED_COLUMNS, "prepared")
    if df["loan_id"].duplicated().any():
        raise ValueError("prepared: duplicate loan_id")
    bad = df.loc[~df["default_flag"].isna(), "default_flag"].isin([0, 1])
    if not bad.all():
        raise ValueError("prepared: default_flag must be 0/1/NA")


def validate_scored(df: pd.DataFrame) -> None:
    check_columns(df, {**SCORED_CORE_COLUMNS, **dict.fromkeys(CSI_FEATURES, "object")}, "scored")
    if not df["pd"].between(0, 1).all():
        raise ValueError("scored: pd outside [0,1]")
    if df.loc[~df["outcome_observable"], "default_flag"].notna().any():
        raise ValueError("scored: default_flag must be NA when outcome_observable is False")


def validate_results(df: pd.DataFrame) -> None:
    check_columns(df, RESULT_COLUMNS, "results", extra_ok=False)
    if not df["rag"].isin(RAG_VALUES).all():
        raise ValueError(f"results: rag must be one of {RAG_VALUES}")
    if not df["direction"].isin(["higher_worse", "lower_worse"]).all():
        raise ValueError("results: bad direction")
    if df.duplicated(["month", "metric", "segment"]).any():
        raise ValueError("results: duplicate (month, metric, segment)")
