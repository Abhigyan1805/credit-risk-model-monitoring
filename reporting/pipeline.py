"""Orchestrate a full reporting run: extracts + report + dashboard.

One call produces every reporting artefact from a monitoring-results document::

    from reporting import generate_outputs
    paths = generate_outputs("monitoring_results.json", "out/reporting")

The individual steps are kept separate (and independently testable) so a
failure in, say, the dashboard does not prevent the CSV extracts from being
written — those extracts feed the Tableau dashboard, which is the critical
deliverable.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from reporting.dashboard import write_dashboard
from reporting.extracts import write_extracts
from reporting.report import write_report
from reporting.schema import MonitoringResults, load_results, normalize_results

DEFAULT_FORMATS: tuple[str, ...] = ("extracts", "report", "dashboard")


def generate_outputs(
    source: str | dict | MonitoringResults,
    out_dir: str,
    formats: Iterable[str] = DEFAULT_FORMATS,
    inline_js: bool = False,
) -> dict[str, Any]:
    """Generate all requested reporting artefacts into ``out_dir``.

    Parameters
    ----------
    source:
        Path to a monitoring-results JSON file, a raw payload dict, or an
        already-normalized :class:`MonitoringResults`.
    out_dir:
        Output directory. Layout: ``extracts/``, ``report/``, ``dashboard/``.
    formats:
        Any subset of ``{"extracts", "report", "dashboard"}``.
    inline_js:
        Embed plotly.js inside the HTML (fully offline) instead of a CDN link.

    Returns a mapping of artefact name to path(s).
    """
    results = _coerce(source)
    formats = tuple(formats)
    out: dict[str, Any] = {"out_dir": out_dir, "formats": list(formats), "files": {}}

    if "extracts" in formats:
        out["files"].update(write_extracts(results, os.path.join(out_dir, "extracts")))

    if "report" in formats:
        out["files"]["report"] = write_report(
            results, os.path.join(out_dir, "report", "monitoring_report.html"), inline_js=inline_js
        )

    if "dashboard" in formats:
        out["files"]["dashboard"] = write_dashboard(
            results, os.path.join(out_dir, "dashboard", "monitoring_dashboard.html"), inline_js=inline_js
        )

    return out


def _coerce(source: str | dict | MonitoringResults) -> MonitoringResults:
    if isinstance(source, MonitoringResults):
        return source
    if isinstance(source, dict):
        return normalize_results(source)
    return load_results(source)
