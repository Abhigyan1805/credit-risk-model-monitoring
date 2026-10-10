"""Offline demo: run the monitoring engine on synthetic scored vintages.

``python -m monitoring --months 6`` builds a synthetic loan book, runs the
contract results-table builder for each month and writes
``monitoring_results.csv`` plus prints the monthly headline RAG. Useful as a
walkthrough when the full LendingClub data is not available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.schemas import validate_results
from monitoring.config import load_thresholds
from monitoring.results import build_monitoring_results
from monitoring.synthetic import make_synthetic_scored

_HEADLINE = ("psi_score", "gini", "ks", "hl_pvalue")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline monitoring demo on synthetic data")
    parser.add_argument("--out-dir", default="out/monitoring")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    dev_df, months = make_synthetic_scored(n_months=args.months, seed=args.seed)
    thresholds = load_thresholds()

    frames = [
        build_monitoring_results(month, dev_df, month_df, thresholds) for month, month_df in months.items()
    ]
    results = pd.concat(frames, ignore_index=True)
    validate_results(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "monitoring_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(results)} rows)")

    overall = results[results["metric"] == "overall_rag"].set_index("month")["rag"]
    print("\nMonthly headline:")
    for month in months:
        values = results[(results["month"] == month) & (results["metric"].isin(_HEADLINE))]
        headline = ", ".join(
            f"{row.metric}={row.value:.4f}" for row in values.itertuples() if pd.notna(row.value)
        )
        print(f"  {month}  overall_rag={overall[month]:<5}  {headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
