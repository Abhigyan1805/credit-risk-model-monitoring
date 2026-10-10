"""Synthetic scored vintages for tests and the offline demo.

Builds DataFrames in the scored-vintage schema (``common.schemas``) so the
monitoring engine can be exercised without the full LendingClub data or the
trained model. Later months drift on purpose, so the demo produces real
GREEN/AMBER/RED alerts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import CSI_FEATURES, MODEL_VERSION

_NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico",
    "revol_util",
    "credit_history_years",
    "emp_length_years",
]
_CATEGORICAL_FEATURES = ["grade", "home_ownership", "verification_status", "purpose"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _draw(rng: np.random.Generator, n: int, drift: float) -> dict[str, np.ndarray]:
    """Draw model features and a true default probability for ``n`` loans."""
    fico = np.clip(rng.normal(700 - 25 * drift, 40, n), 580, 820)
    dti = np.clip(rng.normal(18 + 4 * drift, 6, n), 1, 40)
    int_rate = np.clip(rng.normal(13 + 3 * drift, 3, n), 5, 30)
    loan_amnt = np.clip(rng.normal(15000 + 2000 * drift, 5000, n), 1000, 40000)
    annual_inc = np.clip(rng.lognormal(11.0, 0.5, n), 12000, 600000)
    revol_util = np.clip(rng.normal(50 + 8 * drift, 20, n), 0, 150)
    credit_history_years = np.clip(rng.normal(15, 5, n), 0, 50)
    emp_length_years = rng.choice(
        np.append(np.arange(0, 11, dtype=float), np.nan),
        n,
        p=np.append(np.full(11, 0.08), 0.12),
    )

    shift = float(np.clip(drift, 0, 1))
    grade_choices = np.array(list("ABCDEFG"))
    grade_probs = np.array([0.20, 0.25, 0.22, 0.15, 0.09, 0.06, 0.03])
    grade_probs = grade_probs * (1 - 0.3 * shift) + np.roll(grade_probs, 1) * 0.3 * shift
    grade = rng.choice(grade_choices, n, p=grade_probs / grade_probs.sum())
    home_ownership = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], n, p=[0.5, 0.35, 0.14, 0.01])
    verification_status = rng.choice(["Not Verified", "Source Verified", "Verified"], n, p=[0.4, 0.3, 0.3])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "other"],
        n,
        p=[0.45, 0.30, 0.15, 0.10],
    )

    grade_risk = np.array([(ord(g) - ord("A")) * 0.25 for g in grade])
    logit = (
        -1.2 - 0.9 * drift - 0.020 * (fico - 700) + 0.060 * (dti - 18) + 0.080 * (int_rate - 13) + grade_risk
    )
    pd_true = _sigmoid(logit)
    score = 700.0 - 30.0 * np.log(pd_true / (1.0 - pd_true)) + rng.normal(0, 15, n)

    return {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico": fico,
        "revol_util": revol_util,
        "credit_history_years": credit_history_years,
        "emp_length_years": emp_length_years,
        "grade": grade,
        "home_ownership": home_ownership,
        "verification_status": verification_status,
        "purpose": purpose,
        "pd": pd_true,
        "score": score,
    }


def _score_band(pd_values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(pd_values, cuts) + 1, 1, 10).astype(int)


def _frame(
    rng: np.random.Generator,
    n: int,
    issue_months: list[str],
    drift: float,
    observable: bool,
    cuts: np.ndarray,
    loan_id_start: int,
) -> pd.DataFrame:
    data = _draw(rng, n, drift)
    issue_month = rng.choice(issue_months, n)
    pd_values = data["pd"]
    default_flag = (
        pd.array(rng.binomial(1, pd_values), dtype="Int8")
        if observable
        else pd.array([pd.NA] * n, dtype="Int8")
    )
    frame = pd.DataFrame(
        {
            "loan_id": [f"L{loan_id_start + i:07d}" for i in range(n)],
            "issue_month": issue_month,
            "pd": pd_values,
            "score": data["score"],
            "score_band": _score_band(pd_values, cuts),
            "default_flag": default_flag,
            "outcome_observable": observable,
            "model_version": MODEL_VERSION,
        }
    )
    for feature in CSI_FEATURES:
        frame[feature] = data[feature]
    return frame


def make_synthetic_scored(
    n_dev: int = 2000,
    n_per_month: int = 500,
    n_months: int = 6,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return ``(dev_reference, {month: vintage})`` synthetic scored frames.

    The dev reference spans the out-of-time months 2013-07..2013-12; monitor
    months run 2014-01.. and drift increasingly from month 4. The last two
    months are marked not observable to exercise the ``NA`` outcome path.
    """
    rng = np.random.default_rng(seed)
    dev_months = [f"2013-{m:02d}" for m in range(7, 13)]
    dev_data = _draw(rng, n_dev, drift=0.0)
    cuts = np.quantile(dev_data["pd"], np.linspace(0, 1, 11))[1:-1]

    dev_df = _frame(rng, n_dev, dev_months, drift=0.0, observable=True, cuts=cuts, loan_id_start=0)

    months: dict[str, pd.DataFrame] = {}
    for idx in range(1, n_months + 1):
        drift = 0.0 if idx <= 3 else 0.2 * (idx - 3)
        month = f"2014-{idx:02d}"
        observable = idx <= n_months - 2
        months[month] = _frame(
            rng,
            n_per_month,
            [month],
            drift=drift,
            observable=observable,
            cuts=cuts,
            loan_id_start=1_000_000 + idx * 100_000,
        )
    return dev_df, months
