import ast
from pathlib import Path

DAG_FILE = Path(__file__).parent / "dags" / "crm_monthly_monitoring.py"


def test_dag_file_parses_and_declares_chain():
    src = DAG_FILE.read_text()
    ast.parse(src)
    for needle in ("catchup=True", "max_active_runs=1", "ingest >> train >> score >> monitor >> report"):
        assert needle in src
