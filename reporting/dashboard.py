"""Fallback interactive dashboard (single self-contained HTML file).

If Tableau is not available (or for quick review inside the repo / CI), this
renders the same four views as the Tableau specification using Plotly: hover to
read exact values, drag to zoom, click a legend entry to isolate a series. It
is not a replacement for the Tableau dashboard the captain builds, but it
guarantees the monitoring story is always viewable.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from reporting import charts
from reporting.rag import status_for_color
from reporting.schema import MonitoringResults
from reporting.tables import build_fact_table

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def build_dashboard_context(results: MonitoringResults, inline_js: bool = True) -> dict[str, Any]:
    _, fact_rows = build_fact_table(results)
    latest = fact_rows[-1] if fact_rows else {}
    context: dict[str, Any] = {
        "model": results.model,
        "generated_at": results.generated_at,
        "latest": latest,
        "latest_color": status_for_color(latest.get("rag_overall")),
        "fact_rows": fact_rows,
        "n_periods": len(fact_rows),
        "breaches": results.breaches,
    }
    if charts.has_plotly():
        figures = charts.build_plotly_figures(results)
        fragments, plotlyjs = charts.figures_to_html(figures, inline_js=inline_js)
        context["charts"] = fragments
        context["plotlyjs"] = plotlyjs
        context["is_interactive"] = True
    else:
        context["charts"] = charts.svg_fallback_charts(results)
        context["plotlyjs"] = ""
        context["is_interactive"] = False
    return context


def render_dashboard(results: MonitoringResults, inline_js: bool = True) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("dashboard.html.j2").render(
        **build_dashboard_context(results, inline_js=inline_js)
    )


def write_dashboard(results: MonitoringResults, path: str, inline_js: bool = True) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_dashboard(results, inline_js=inline_js))
    return path
