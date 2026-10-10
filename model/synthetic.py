"""Synthetic loans in the PREPARED schema (common/schemas.py) for tests only - not real data.

Default probability is driven by int_rate, grade, dti, income, revol_util and purpose plus noise, so a
scorecard should recover real signal. `drift` shifts int_rate upward over time to test monitoring code.
"""

import numpy as np
import pandas as pd


def make_loans(n=6000, start="2012-01", months=60, seed=0, drift=0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.integers(0, months, n)
    issue_month = [(pd.Period(start, "M") + int(i)).strftime("%Y-%m") for i in m]
    t = m / max(months - 1, 1)
    int_rate = np.clip(rng.normal(13 + 4 * drift * t, 4, n), 5, 28)
    dti = np.clip(rng.gamma(4, 5, n), 0, 60)
    income = np.exp(rng.normal(11.1, 0.5, n))
    revol = np.clip(rng.normal(50, 25, n), 0, 120)
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"],
                         n, p=[0.5, 0.2, 0.1, 0.05, 0.15])  # fmt: skip
    z = (-3.0 + 0.15 * (int_rate - 13) + 0.03 * (dti - 20) - 0.6 * (np.log(income) - 11.1)
         + 0.01 * (revol - 50) + 0.6 * (purpose == "small_business") + rng.normal(0, 0.3, n))  # fmt: skip
    bad = rng.random(n) < 1 / (1 + np.exp(-z))
    flag = pd.array(np.where(bad, 1, 0), dtype="Int8")
    flag[rng.random(n) < 0.05] = pd.NA
    grade = pd.cut(int_rate, [0, 8, 11, 14, 17, 20, 24, 100], labels=list("ABCDEFG")).astype(str)
    df = pd.DataFrame({
        "loan_id": [str(i) for i in range(n)], "issue_month": issue_month, "default_flag": flag,
        "loan_amnt": np.round(rng.gamma(4, 3500, n), -2) + 1000, "int_rate": int_rate,
        "annual_inc": income, "dti": dti, "revol_util": revol, "grade": grade,
        "fico": np.clip(rng.normal(690, 30, n), 620, 850),
        "credit_history_years": rng.gamma(5, 3, n), "emp_length_years": rng.integers(0, 11, n).astype(float),
        "delinq_2yrs": rng.poisson(0.3, n).astype(float), "inq_last_6mths": rng.poisson(0.8, n).astype(float),
        "open_acc": rng.poisson(11, n).astype(float), "pub_rec": rng.poisson(0.15, n).astype(float),
        "revol_bal": np.exp(rng.normal(9, 1, n)), "total_acc": rng.poisson(25, n).astype(float),
        "purpose": purpose,
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.45, 0.1]),
        "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
    })  # fmt: skip
    df.loc[rng.random(n) < 0.03, "revol_util"] = np.nan
    return df
