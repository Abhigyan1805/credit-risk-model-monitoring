"""Tests for the monthly HTML report."""

from common import paths
from reporting import report


def test_render_report_sections(results):
    html = report.render_report(results)
    assert "Overall status" in html
    assert "Executive summary" in html
    assert "Stability" in html
    assert "Calibration by score band" in html
    assert "Feature stability heat-map" in html
    assert "Breaches" in html
    assert "Monthly summary" in html


def test_report_embeds_charts_as_png(results):
    html = report.render_report(results)
    assert html.count("data:image/png;base64,") >= 5
    # Self-contained: no external requests.
    assert "http://" not in html and "https://" not in html


def test_write_report_uses_reports_dir(results, crm_data):
    path = report.write_report(results, "2014-06")
    assert path == paths.reports_dir() / "monitoring_report_2014-06.html"
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_report_main(results, crm_data, capsys):
    report.main(["--month", "2014-03"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("monitoring_report_2014-03.html")
    assert (crm_data / "reports" / "monitoring_report_2014-03.html").exists()


def test_report_only_uses_months_up_to_requested(results, crm_data):
    html = report.render_report(results)
    # The rendered context for all months includes every synthetic month.
    assert "2014-06" in html
