"""Reporting component (builder #4). See ``docs/CONTRACTS.md`` sections 6 and 11.

Two entry points:

* ``python -m reporting.report --month YYYY-MM`` — self-contained monthly HTML report
  (``data/reports/monitoring_report_<month>.html``).
* ``python -m reporting.extracts`` — Tableau-ready CSVs (``data/tableau/``).

The Tableau build guide is ``docs/DASHBOARD_SPEC.md``; design and interview notes are
``docs/reporting/REPORTING.md``.
"""
