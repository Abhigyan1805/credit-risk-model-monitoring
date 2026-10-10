import pandas as pd

from common import config
from common.schemas import validate_prepared
from ingest.prepare import clean_chunk

RAW = {
    "id": ["1", "2", "3", "4", "5"],
    "issue_d": ["Mar-2012", "Jun-2014", "Jun-2014", "Jan-2013", "Jan-2013"],
    "loan_status": ["Fully Paid", "Charged Off", "Current", "Fully Paid", "Fully Paid"],
    "loan_amnt": ["1000"] * 5,
    "term": [" 36 months", " 36 months", " 36 months", " 60 months", " 36 months"],
    "int_rate": ["10.5"] * 5,
    "installment": ["30"] * 5,
    "grade": ["A", "B", "C", "A", "A"],
    "sub_grade": ["A1", "B2", "C3", "A1", "A1"],
    "emp_length": ["< 1 year", "10+ years", None, "3 years", "3 years"],
    "home_ownership": ["RENT"] * 5,
    "annual_inc": ["50000"] * 5,
    "verification_status": ["Verified"] * 5,
    "purpose": ["car"] * 5,
    "addr_state": ["CA"] * 5,
    "dti": ["12.1"] * 5,
    "delinq_2yrs": ["0"] * 5,
    "fico_range_low": ["700"] * 5,
    "fico_range_high": ["704"] * 5,
    "inq_last_6mths": ["1"] * 5,
    "open_acc": ["5"] * 5,
    "pub_rec": ["0"] * 5,
    "revol_bal": ["900"] * 5,
    "revol_util": ["40"] * 5,
    "total_acc": ["9"] * 5,
    "earliest_cr_line": ["Mar-2002", "Jun-2004", "Jun-2004", "Jan-2003", "Jan-2003"],
}


def test_clean_chunk_scope_label_and_schema():
    out = clean_chunk(pd.DataFrame(RAW))
    assert list(out["loan_id"]) == ["1", "2", "3", "5"]  # 60-month loan dropped
    flags = out["default_flag"]
    assert flags.iloc[[0, 1, 3]].tolist() == [0, 1, 0] and pd.isna(flags.iloc[2])
    assert list(out["window"]) == ["dev", "monitor", "monitor", "dev"]
    assert out.loc[out["loan_id"] == "1", "emp_length_years"].iloc[0] == 0
    assert out["fico"].iloc[0] == 702
    assert abs(out["credit_history_years"].iloc[0] - 10.0) < 0.1
    validate_prepared(out)


def test_windows_are_consistent():
    months = config.month_range(config.DEV_START, config.MONITOR_END)
    assert months[0] == "2012-01" and months[-1] == "2016-12"
    assert config.DEV_END < config.MONITOR_START
    assert config.outcome_observable("2015-12") and not config.outcome_observable("2016-01")
