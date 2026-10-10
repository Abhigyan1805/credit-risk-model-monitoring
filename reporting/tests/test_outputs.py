"""End-to-end tests for report/dashboard generation and the pipeline."""

import json
import os

import pytest

from reporting import dashboard, report
from reporting import charts as charts_module
from reporting.pipeline import generate_outputs
from reporting.schema import normalize_results

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_monitoring_results.json"
)


@pytest.fixture()
def results():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return normalize_results(json.load(fh))


def test_report_html_contains_key_sections(results):
    html = report.render_report(results)
    assert "Overall status" in html
    assert "Executive summary" in html
    assert "Calibration by score band" in html
    assert "Feature stability heat-map" in html
    assert "Breaches" in html
    assert results.periods_sorted[-1]["period"] in html


def test_dashboard_html_contains_key_views(results):
    html = dashboard.render_dashboard(results)
    assert "monitoring dashboard" in html.lower()
    assert "Plotly" in html or "plotly" in html


def test_report_uses_svg_fallback_without_plotly(results, monkeypatch):
    monkeypatch.setattr(charts_module, "has_plotly", lambda: False)
    html = report.render_report(results)
    assert "<svg" in html
    assert "Overall status" in html


def test_pipeline_writes_all_artifacts(results, tmp_path):
    out = generate_outputs(results, str(tmp_path), inline_js=True)
    files = out["files"]
    assert os.path.exists(files["fact"])
    assert os.path.exists(files["report"])
    assert os.path.exists(files["dashboard"])
    # Inline JS makes the report fully offline (no external plotly script tag).
    with open(files["report"], "r", encoding="utf-8") as fh:
        report_html = fh.read()
    assert 'src="https://cdn.plot.ly' not in report_html
    assert len(report_html) > 1_000_000  # inline bundle is ~3-4 MB


def test_pipeline_formats_subset(results, tmp_path):
    out = generate_outputs(results, str(tmp_path), formats=["extracts"])
    assert "report" not in out["files"]
    assert "dashboard" not in out["files"]
    assert os.path.exists(out["files"]["fact"])


def test_pipeline_accepts_raw_dict_and_path(results, tmp_path):
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    from_dict = generate_outputs(raw, str(tmp_path / "from_dict"), formats=["extracts"])
    from_path = generate_outputs(FIXTURE, str(tmp_path / "from_path"), formats=["extracts"])
    assert os.path.exists(from_dict["files"]["fact"])
    assert os.path.exists(from_path["files"]["fact"])
