"""Create the small committed fixture from the prepared full data. ``python -m ingest.sample``

Stratified: N loans per issue month (dev and monitor), so every monthly vintage exists in the sample.
"""

from __future__ import annotations

import pandas as pd

from common.paths import SAMPLE_LOANS, ensure_parent, prepared_loans

PER_MONTH = 70


def main() -> None:
    df = pd.read_parquet(prepared_loans())
    out = df.sample(frac=1, random_state=7).groupby("issue_month").head(PER_MONTH)
    out = out.sort_values("loan_id").reset_index(drop=True)
    out.to_parquet(ensure_parent(SAMPLE_LOANS), index=False, compression="zstd")
    print(f"sample rows: {len(out)} -> {SAMPLE_LOANS}")


if __name__ == "__main__":
    main()
