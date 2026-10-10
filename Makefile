# Shared Makefile. All targets use .venv; CRM_DATA_DIR selects the data root (default ./data).
PY := .venv/bin/python
MONTHS = $(shell $(PY) -c "from common.config import monitor_months; print(' '.join(monitor_months()))")

.PHONY: setup setup-airflow lint test download ingest sample train score monitor report extracts \
        e2e demo airflow-test clean

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,ingest]"

setup-airflow:
	uv pip install --python $(PY) -e ".[airflow]"

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

test:
	$(PY) -m pytest -q

# --- data (full LendingClub run; needs ~/.kaggle/access_token) ---
download:
	$(PY) -m ingest.download

ingest:
	$(PY) -m ingest.prepare

sample:           ## regenerate the committed fixture from the prepared full data
	$(PY) -m ingest.sample

# --- pipeline stages (builder-owned modules) ---
train:
	$(PY) -m model.train

score:
	@for m in $(MONTHS); do $(PY) -m model.score --month $$m || exit 1; done

monitor:
	@for m in $(MONTHS); do $(PY) -m monitoring.run --month $$m || exit 1; done

report:
	@for m in $(MONTHS); do $(PY) -m reporting.report --month $$m || exit 1; done

extracts:
	$(PY) -m reporting.extracts

# Full real-data pipeline, all monthly vintages, one command.
e2e: download ingest train score monitor report extracts

# Small-fixture pipeline: no download, runs in a minute.
demo:
	CRM_DATA_DIR=data/demo $(PY) -m ingest.prepare --from-sample
	CRM_DATA_DIR=data/demo $(MAKE) train score monitor report extracts

airflow-test:
	bash orchestration/run_dag_test.sh 2014-01

clean:
	rm -rf data/demo data/prepared data/scored data/monitoring data/reports data/tableau data/model
