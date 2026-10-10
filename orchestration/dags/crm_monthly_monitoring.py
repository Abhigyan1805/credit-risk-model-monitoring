"""Monthly model-monitoring DAG: ingest check -> train-if-needed -> score month -> monitor -> report.

* One run = one issue-month vintage (month = the run's logical date, ``YYYY-MM``).
* Idempotent: ``score``/``monitor``/``report`` overwrite only that month's files; ``ensure_*`` steps
  skip when their output already exists. Re-running a month never duplicates results.
* Backfill: ``catchup=True`` with a fixed start/end date replays 2014-01 .. 2016-12 sequentially
  (``max_active_runs=1``), exactly like a bank catching up on missed monitoring cycles.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
MONTH = "{{ logical_date.strftime('%Y-%m') }}"
ENV = {**os.environ, "PYTHONPATH": str(REPO)}


def step(task_id: str, name: str, with_month: bool = True) -> BashOperator:
    suffix = f" --month {MONTH}" if with_month else ""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {REPO} && {PY} -m orchestration.steps {name}{suffix}",
        env=ENV,
        append_env=True,
    )


with DAG(
    dag_id="crm_monthly_monitoring",
    description="Monthly PD-model monitoring: score vintage, PSI/CSI/Gini/KS/calibration, RAG, report",
    schedule="@monthly",
    start_date=datetime(2014, 1, 1),
    end_date=datetime(2016, 12, 31),
    catchup=True,
    max_active_runs=1,
    tags=["credit-risk", "model-monitoring"],
) as dag:
    ingest = step("ingest", "ensure_prepared", with_month=False)
    train = step("ensure_model", "ensure_model", with_month=False)
    score = step("score_month", "score")
    monitor = step("monitor_month", "monitor")
    report = step("report_month", "report")
    extracts = step("refresh_tableau_extracts", "extracts", with_month=False)

    ingest >> train >> score >> monitor >> report >> extracts
