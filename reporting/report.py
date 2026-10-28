"""Monthly monitoring report. ``python -m reporting.report --month YYYY-MM``

Writes ``common.paths.reports_dir()/monitoring_report_<month>.html`` — a single
self-contained HTML file (charts embedded as base64 PNGs) covering every month
up to ``month``. It answers the questions a model-monitoring reviewer asks:
overall health, population stability, discrimination, calibration, breaches.

See ``docs/reporting/REPORTING.md`` for the design and interview notes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from commentary.run import load_commentary
from common import paths
from reporting import charts
from reporting.data import available_months, load_results
from reporting.rag import rag_color
from reporting.tables import (
    build_breaches,
    build_calibration_by_band,
    build_csi_by_feature,
    build_monthly_summary,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

#: KPI tiles on the executive summary: (summary column, RAG column, human label, decimals).
_KPIS: tuple[tuple[str, str, str, int], ...] = (
    ("psi_score", "rag_psi_score", "PSI (score)", 4),
    ("csi_max", "rag_csi_max", "CSI (max feature)", 4),
    ("gini", "rag_gini", "Gini", 4),
    ("ks", "rag_ks", "KS", 4),
    ("calibration_ratio", "rag_calibration_ratio", "Calibration ratio", 3),
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _thresholds(results: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    rows = results[(results["metric"] == metric) & (results["segment"] == "ALL")]
    if rows.empty:
        return None, None
    row = rows.iloc[-1]
    amber = row["amber_threshold"]
    red = row["red_threshold"]
    return (None if pd.isna(amber) else float(amber), None if pd.isna(red) else float(red))


def build_report_context(results: pd.DataFrame, generated_at: str | None = None) -> dict[str, Any]:
    """Assemble everything the template needs from the long results table."""
    summary = build_monthly_summary(results)
    csi_by_feature = build_csi_by_feature(results)
    calibration = build_calibration_by_band(results)
    breaches = build_breaches(results)
    latest = summary.iloc[-1]
    latest_month = str(latest["month"])
    calibration_latest = calibration[calibration["month"] == latest_month]

    psi_amber, psi_red = _thresholds(results, "psi_score")
    calib_amber, calib_red = _thresholds(results, "calibration_ratio")
    gini_amber, gini_red = _thresholds(results, "gini")
    ks_amber, ks_red = _thresholds(results, "ks")

    charts_map = {
        "rag_strip": charts.rag_strip(summary),
        "psi_trend": charts.psi_trend(summary, psi_amber or 0.10, psi_red or 0.25),
        "discrimination_trend": charts.discrimination_trend(summary, gini_amber, gini_red, ks_amber, ks_red),
        "calibration": charts.calibration_by_band(calibration_latest, calib_amber or 0.15, calib_red or 0.30),
        "csi_heatmap": charts.csi_heatmap(csi_by_feature),
    }

    return {
        "model_version": str(latest.get("model_version", results["model_version"].iloc[-1])),
        "generated_at": generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "latest_month": latest_month,
        "overall_rag": str(latest.get("overall_rag", "NA")),
        "overall_color": rag_color(latest.get("overall_rag")),
        "kpis": [
            {
                "label": label,
                "value": _fmt(latest.get(col), digits),
                "rag": str(latest.get(rag_col, "NA")),
                "color": rag_color(latest.get(rag_col)),
            }
            for col, rag_col, label, digits in _KPIS
        ],
        "charts": charts_map,
        "summary_rows": summary.to_dict("records"),
        "calibration_rows": calibration_latest.to_dict("records"),
        "breaches": breaches.to_dict("records"),
        "month_min": str(summary["month"].iloc[0]),
        "month_max": latest_month,
        "n_months": int(len(summary)),
        "default_rate": _fmt(latest.get("default_rate")),
        "commentary": _commentary_context(latest_month),
    }


#: How each publication path is described next to the commentary.
_COMMENTARY_PATHS = {
    "llm_first_pass": "AI-drafted, verified: the first LLM draft passed every check.",
    "llm_after_retry": "AI-drafted, verified: an earlier LLM draft failed the checker; this revision passed.",
    "template_fallback": (
        "Template fallback: no LLM draft passed the checker, so a deterministic summary is shown."
    ),
    "template_only": "Deterministic template summary (no LLM was run).",
}


def _commentary_context(month: str) -> dict[str, Any] | None:
    """The published commentary for ``month`` (``commentary.run`` output), or None if none was made."""
    c = load_commentary(month)
    if c is None:
        return None
    rejected = [a for a in c["attempts"] if not a["passed"]]
    return {
        "paragraphs": [p.strip() for p in c["text"].split("\n") if p.strip()],
        "path": c["path"],
        "is_llm": c["path"].startswith("llm"),
        "path_text": _COMMENTARY_PATHS.get(c["path"], c["path"]),
        "model": c["model"].get("model_id", c["model"].get("name", "")),
        "quantization": c["model"].get("quantization"),
        "n_claims": c["n_claims"],
        "n_attempts": len(c["attempts"]),
        "rejected_issues": [i for a in rejected for i in a["issues"]],
        "generated_at": c["generated_at"],
    }


def render_report(results: pd.DataFrame, generated_at: str | None = None) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=select_autoescape(["html", "xml"])
    )
    return env.get_template("report.html.j2").render(**build_report_context(results, generated_at))


def write_report(results: pd.DataFrame, month: str, out_path: Path | None = None) -> Path:
    out_path = out_path or (paths.reports_dir() / f"monitoring_report_{month}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(results), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reporting.report", description="Build the monthly monitoring report."
    )
    parser.add_argument(
        "--month", required=True, help="Report month, YYYY-MM (all months up to it are shown)."
    )
    parser.add_argument(
        "--out", help=f"Output HTML path (default {paths.reports_dir()}/monitoring_report_<month>.html)."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    months = available_months(up_to=args.month)
    results = load_results(months)
    out_path = Path(args.out) if args.out else None
    written = write_report(results, args.month, out_path)
    print(written)


if __name__ == "__main__":
    main()
