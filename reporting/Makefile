# Reporting component targets.
#
# The repository-level Makefile is owned by the monitoring skeleton; these
# targets are the reporting component's contract. If the skeleton includes
# component makefiles, add `include reporting/Makefile` there (or copy the
# `report` target) so `make report` works from the repo root.

PYTHON ?= python3
OUT_DIR ?= out/reporting
INPUT ?=

.PHONY: report report-fixture report-tests

# Generate every reporting artefact from a real monitoring-results file:
#   make report INPUT=results/monitoring_results.json
report:
	$(PYTHON) -m reporting.cli --input "$(INPUT)" --out-dir "$(OUT_DIR)"

# Generate every artefact from the committed synthetic fixture (no dataset).
report-fixture:
	$(PYTHON) -m reporting.cli --fixture --out-dir "$(OUT_DIR)"

# Reporting unit tests (synthetic data only; no LendingClub file required).
report-tests:
	$(PYTHON) -m pytest reporting/tests -q
