"""Auto-generated per-run monitoring report (single HTML file).

The report answers the four questions a model-monitoring reviewer asks:

1. Is the model healthy *right now*? — executive summary + overall RAG status.
2. Has the input population moved? — PSI / CSI trend.
3. Is the model still discriminating? — Gini / KS trend.
4. Is it still calibrated, and what breached? — calibration-by-band table and
   the breach list.

Charts use Plotly when available and fall back to inline SVG otherwise, so a
report can always be produced on a machine without the optional dependency.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from reporting.rag import METRIC_LABELS, MONITORED_METRICS, status_for_color
from reporting.schema import MonitoringResults
from reporting.tables import build_calibration_table, build_fact_table

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _kpis(latest_row: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = [
        ("psi", "rag_psi"),
        ("csi", "rag_csi"),
        ("auc", "rag_auc"),
        ("gini", "rag_gini"),
        ("ks", "rag_ks"),
        ("calibration_gap", "rag_calibration"),
    ]
    kpis = []
    for metric, rag_col in mapping:
        kpis.append(
            {
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "value": _fmt(latest_row.get(metric)),
                "status": latest_row.get(rag_col),
                "color": status_for_color(latest_row.get(rag_col)),
            }
        )
    return kpis


def build_report_context(results: MonitoringResults, inline_js: bool = False) -> dict[str, Any]:
    """Assemble the template context, choosing Plotly or the SVG fallback."""
    from reporting import charts

    _, fact_rows = build_fact_table(results)
    _, calib_rows = build_calibration_table(results)
    latest = results.latest_period or {}
    latest_period = latest.get("period")
    latest_fact = next((r for r in fact_rows if r["period"] == latest_period), {})
    latest_calib = [r for r in calib_rows if r["period"] == latest_period]

    overall_status = latest.get("rag")
    context: dict[str, Any] = {
        "model": results.model,
        "generated_at": results.generated_at,
        "overall_status": overall_status,
        "overall_color": status_for_color(overall_status),
        "latest_period": latest_period,
        "kpis": _kpis(latest_fact),
        "calibration_rows": latest_calib,
        "latest_fact": latest_fact,
        "breaches": results.breaches,
        "fact_rows": fact_rows,
        "n_periods": len(fact_rows),
        "is_interactive": charts.has_plotly(),
        "metric_labels": METRIC_LABELS,
        "monitored_metrics": MONITORED_METRICS,
    }

    if charts.has_plotly():
        figures = charts.build_plotly_figures(results)
        fragments, plotlyjs = charts.figures_to_html(figures, inline_js=inline_js)
        context["charts"] = fragments
        context["plotlyjs"] = plotlyjs
    else:
        context["charts"] = charts.svg_fallback_charts(results)
        context["plotlyjs"] = ""
    return context


def render_report(results: MonitoringResults, inline_js: bool = False) -> str:
    """Render the full monitoring report as an HTML string."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")
    return template.render(**build_report_context(results, inline_js=inline_js))


def write_report(results: MonitoringResults, path: str, inline_js: bool = False) -> str:
    """Render and write the report to ``path``; returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_report(results, inline_js=inline_js))
    return path
