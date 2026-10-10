#!/usr/bin/env bash
# Run the monitoring DAG natively (no scheduler/webserver, no Docker):
#   bash orchestration/run_dag_test.sh 2014-01 [2014-02 ...]   -> one `airflow dags test` per month
set -euo pipefail
cd "$(dirname "$0")/.."
export AIRFLOW_HOME="${AIRFLOW_HOME:-$PWD/airflow_home}"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/orchestration/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export PYTHONPATH="$PWD"
AF=.venv/bin/airflow
$AF db migrate >/dev/null 2>&1
for m in "$@"; do
  $AF dags test crm_monthly_monitoring "${m}-01"
done
