"""Builder #3 implements. ``python -m monitoring.run --month YYYY-MM``

Compares scored vintage ``month`` against the dev reference (``scored_path("dev")``) and writes
``paths.monitoring_path(month)`` (RESULT_COLUMNS, validate_results). Idempotent per month.
Thresholds live in ``config/thresholds.yaml``.
"""


def main() -> None:
    raise NotImplementedError("builder #3: monitoring/run.py")


if __name__ == "__main__":
    main()
