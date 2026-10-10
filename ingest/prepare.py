"""Prepare the loan table. ``python -m ingest.prepare [--from-sample]``

Streams the raw accepted-loans file in chunks (never loads it whole), keeps only needed columns,
and writes ``data/prepared/loans.parquet`` (schema: common.schemas.PREPARED_COLUMNS) plus
``ingest_manifest.json`` with the real row counts.

Scope and label (see docs/CONTRACTS.md):
* 36-month loans issued 2012-01..2016-12 (window = dev for 2012-2013, monitor for 2014-2016).
* default_flag = 1 for Charged Off / Default, 0 for Fully Paid, NA for anything unresolved
  (Current, Late, In Grace Period, ...), which is excluded from outcome metrics.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common import config
from common.paths import SAMPLE_LOANS, ensure_parent, prepared_loans, raw_dir
from common.schemas import PREPARED_COLUMNS, validate_prepared
from ingest.download import ACCEPTED

RAW_COLUMNS = [
    "id",
    "issue_d",
    "loan_status",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "earliest_cr_line",
]
STR_COLS = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state"]
NUM_COLS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]
CHUNK = 200_000


def clean_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw LendingClub rows -> prepared schema (only in-scope rows)."""
    df = pd.DataFrame(index=raw.index)
    issue = pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce")
    term = pd.to_numeric(raw["term"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    df["loan_id"] = raw["id"].astype("string")
    df["issue_month"] = issue.dt.strftime("%Y-%m").astype("string")
    df["loan_status"] = raw["loan_status"].astype("string")
    df["term_months"] = term
    in_scope = (
        (term == config.TERM_MONTHS).fillna(False).to_numpy(bool)
        & (df["issue_month"] >= config.DEV_START).fillna(False).to_numpy(bool)
        & (df["issue_month"] <= config.MONITOR_END).fillna(False).to_numpy(bool)
    )
    keep = pd.Series(in_scope, index=df.index)
    df, raw, issue = df[keep].copy(), raw[keep], issue[keep]
    status = df["loan_status"]
    flag = pd.Series(pd.NA, index=df.index, dtype="Int8")
    flag[status.isin(config.DEFAULT_STATUSES)] = 1
    flag[status.isin(config.GOOD_STATUSES)] = 0
    df["default_flag"] = flag
    df["window"] = (df["issue_month"] <= config.DEV_END).map({True: "dev", False: "monitor"})
    df["window"] = df["window"].astype("string")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(raw[c], errors="coerce")
    for c in STR_COLS:
        df[c] = raw[c].astype("string").str.strip()
    df["emp_length_years"] = pd.to_numeric(
        raw["emp_length"].astype("string").str.replace("< 1", "0", regex=False).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    df["fico"] = (pd.to_numeric(raw["fico_range_low"]) + pd.to_numeric(raw["fico_range_high"])) / 2
    earliest = pd.to_datetime(raw["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_years"] = (issue - earliest).dt.days / 365.25
    df["term_months"] = df["term_months"].astype("int64")
    return df[list(PREPARED_COLUMNS)].astype(
        {k: v for k, v in PREPARED_COLUMNS.items() if v != "Int8"}, errors="ignore"
    )


def prepare_full() -> dict:
    src = raw_dir() / ACCEPTED
    parts, scanned = [], 0
    for chunk in pd.read_csv(src, usecols=RAW_COLUMNS, chunksize=CHUNK, dtype=str, low_memory=False):
        scanned += len(chunk)
        parts.append(clean_chunk(chunk))
    df = pd.concat(parts, ignore_index=True)
    return write(df, scanned)


def write(df: pd.DataFrame, rows_scanned: int) -> dict:
    df = df.drop_duplicates("loan_id").reset_index(drop=True)
    validate_prepared(df)
    out = ensure_parent(prepared_loans())
    df.to_parquet(out, index=False)
    obs = df[df["issue_month"] <= config.OUTCOME_CUTOFF_MONTH]
    manifest = {
        "dataset": "wordsforthewise/lending-club (LendingClub accepted loans 2007-2018Q4)",
        "licence": "CC0-1.0 (Kaggle dataset metadata)",
        "raw_rows_scanned": rows_scanned,
        "prepared_rows": len(df),
        "rows_by_window": df["window"].value_counts().to_dict(),
        "resolved_rows": int(df["default_flag"].notna().sum()),
        "default_rate_dev": float(df.loc[df["window"] == "dev", "default_flag"].mean()),
        "default_rate_observable_monitor": float(obs.loc[obs["window"] == "monitor", "default_flag"].mean()),
        "status_counts": df["loan_status"].value_counts().to_dict(),
        "scope": f"{config.TERM_MONTHS}-month loans issued {config.DEV_START}..{config.MONITOR_END}",
    }
    (out.parent / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-sample", action="store_true", help="use the committed fixture")
    args = ap.parse_args()
    if args.from_sample:
        df = pd.read_parquet(SAMPLE_LOANS)
        write(df, len(df))
    else:
        prepare_full()


if __name__ == "__main__":
    main()
