"""``python -m monitoring.run --month YYYY-MM``

Compares the scored vintage ``month`` against the dev reference
(``scored_path("dev")``) and writes ``paths.monitoring_path(month)`` using the
``RESULT_COLUMNS`` schema, validated by ``common.schemas.validate_results``.
Idempotent: it overwrites only that month's file. Thresholds come from
``config/thresholds.yaml``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common import paths, schemas
from monitoring.config import load_thresholds
from monitoring.results import build_monitoring_results


def run(month: str, thresholds=None) -> pd.DataFrame:
    """Compute and persist the monitoring results for one vintage."""
    dev_path = paths.scored_path("dev")
    month_path = paths.scored_path(month)
    if not dev_path.exists():
        raise FileNotFoundError(f"dev reference not found: {dev_path} (run `make train` first)")
    if not month_path.exists():
        raise FileNotFoundError(f"scored vintage not found: {month_path} (run `make score` first)")

    dev_df = pd.read_parquet(dev_path)
    month_df = pd.read_parquet(month_path)
    thresholds = thresholds or load_thresholds()

    results = build_monitoring_results(month, dev_df, month_df, thresholds)
    schemas.validate_results(results)

    out_path = paths.ensure_parent(paths.monitoring_path(month))
    results.to_parquet(out_path, index=False)
    print(
        f"wrote {out_path} ({len(results)} rows); "
        f"overall_rag="
        f"{results.loc[results['metric'] == 'overall_rag', 'rag'].iloc[0]}"
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run monitoring for one vintage")
    parser.add_argument("--month", required=True, help="vintage YYYY-MM")
    args = parser.parse_args(argv)
    run(args.month)


if __name__ == "__main__":
    main()
