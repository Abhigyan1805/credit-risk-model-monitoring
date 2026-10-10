"""Builder #2 implements. ``python -m model.score --month YYYY-MM``

Scores the vintage (loans with issue_month == month) with the frozen model and writes
``paths.scored_path(month)`` (SCORED_COLUMNS). Idempotent: overwrites that file only.
default_flag is filled only when ``common.config.outcome_observable(month)``.
"""


def main() -> None:
    raise NotImplementedError("builder #2: model/score.py")


if __name__ == "__main__":
    main()
