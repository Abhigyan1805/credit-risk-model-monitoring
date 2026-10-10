"""Model feature lists. Column names of the interfaces come from ``common.schemas`` / ``common.config``.

Candidate features are application-time (origination) information only: nothing known after the loan
starts (no payments or recoveries), to avoid leakage. The candidate set is a superset of
``common.config.CSI_FEATURES`` as CONTRACTS.md requires; WoE/IV selection then decides what enters the
final scorecard. Features that are not selected are still copied into the scored file for CSI.

``grade`` and ``int_rate`` are LendingClub's own pricing of risk (the output of another credit model).
They are allowed as candidates because the contract monitors them, but they tend to dominate; the
correlation filter keeps only the stronger of the two, and the docs discuss this choice.
"""

NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "fico", "revol_util", "credit_history_years",
    "emp_length_years", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "total_acc", "loan_to_income",
]  # fmt: skip
CATEGORICAL_FEATURES = ["grade", "home_ownership", "verification_status", "purpose"]

TARGET = "default_flag"
MONTH = "issue_month"
