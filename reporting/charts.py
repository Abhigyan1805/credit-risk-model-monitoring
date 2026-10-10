"""Chart builders for the report and dashboard.

Two rendering paths are provided:

* **Plotly** (preferred) — interactive charts used by both the per-run report
  and the fallback dashboard. Requires the optional ``plotly`` dependency.
* **Inline SVG** (fallback) — small, dependency-free renderers used when
  Plotly is not installed, so a report can always be produced.

The Plotly figures are returned as HTML ``<div>`` fragments (no full HTML
wrapper), ready to drop into a Jinja2 template. ``include_plotlyjs`` is left
to the caller so plotly.js is loaded exactly once per page.
"""

from __future__ import annotations

from typing import Optional

from reporting.rag import METRIC_LABELS, status_for_color
from reporting.schema import MonitoringResults
from reporting.tables import build_calibration_table, build_fact_table, build_feature_csi_table

GREEN = "#1b8a4b"
AMBER = "#d99100"
RED = "#c0392b"
GREY = "#6b7280"


def has_plotly() -> bool:
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Plotly figures
# --------------------------------------------------------------------------- #
def psi_trend_figure(results: MonitoringResults):
    """PSI and CSI over time with green/amber/red reference bands."""
    import plotly.graph_objects as go

    columns, rows = build_fact_table(results)
    periods = [r["period"] for r in rows]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.10, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.10, y1=0.25, fillcolor=AMBER, opacity=0.10, line_width=0)
    fig.add_hrect(y0=0.25, y1=max(0.35, _max_metric(rows, ["psi", "csi"])), fillcolor=RED, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=periods, y=[r["psi"] for r in rows], name="PSI", mode="lines+markers", line=dict(color="#1f4e79")))
    fig.add_trace(go.Scatter(x=periods, y=[r["csi"] for r in rows], name="CSI", mode="lines+markers", line=dict(color="#8e44ad")))
    fig.update_layout(
        title="Population / Characteristic Stability (PSI, CSI)",
        yaxis_title="Index value",
        xaxis_title="Monitoring period",
        margin=dict(l=50, r=20, t=50, b=50),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def discrimination_trend_figure(results: MonitoringResults):
    """Gini and KS over time with the RAG-coloured final markers."""
    import plotly.graph_objects as go

    reference = results.reference_metrics
    _, rows = build_fact_table(results)
    periods = [r["period"] for r in rows]
    colors = [status_for_color(r["rag_gini"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=[r["gini"] for r in rows], name="Gini", mode="lines+markers", line=dict(color="#1f4e79"), marker=dict(color=colors, size=9)))
    fig.add_trace(go.Scatter(x=periods, y=[r["ks"] for r in rows], name="KS", mode="lines+markers", line=dict(color="#16a085")))
    if reference.get("gini") is not None:
        fig.add_hline(y=reference["gini"], line_dash="dot", line_color=GREY, annotation_text="Gini reference", annotation_position="top left")
    if reference.get("ks") is not None:
        fig.add_hline(y=reference["ks"], line_dash="dot", line_color=GREY, annotation_text="KS reference", annotation_position="bottom left")
    fig.update_layout(
        title="Discrimination over time (Gini, KS)",
        yaxis_title="Statistic value",
        xaxis_title="Monitoring period",
        margin=dict(l=50, r=20, t=50, b=50),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def calibration_figure(results: MonitoringResults, period: Optional[str] = None):
    """Predicted vs observed default rate by score band (bar + line)."""
    import plotly.graph_objects as go

    _, rows = build_calibration_table(results)
    if period is None:
        latest = results.latest_period
        period = latest["period"] if latest else None
    rows = [r for r in rows if r["period"] == period]
    bands = [r["band"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bands, y=[r["observed_pd"] for r in rows], name="Observed PD", marker_color="#1f4e79"))
    fig.add_trace(go.Scatter(x=bands, y=[r["predicted_pd"] for r in rows], name="Predicted PD", mode="lines+markers", line=dict(color=AMBER, width=3)))
    fig.update_layout(
        title=f"Calibration by score band — {period}",
        yaxis_title="Default rate",
        xaxis_title="Score band (safest → riskiest)",
        margin=dict(l=50, r=20, t=50, b=50),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig


def csi_heatmap_figure(results: MonitoringResults):
    """Feature CSI heat-map: features (rows) x periods (columns)."""
    import plotly.graph_objects as go

    _, rows = build_feature_csi_table(results)
    features = sorted({r["feature"] for r in rows})
    periods = sorted({r["period"] for r in rows})
    index = {(r["feature"], r["period"]): r["csi"] for r in rows}
    z = [[index.get((f, p)) for p in periods] for f in features]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=periods,
            y=features,
            colorscale=[[0.0, "#e8f5ec"], [0.4, GREEN], [0.7, AMBER], [1.0, RED]],
            zmin=0,
            zmax=0.25,
            colorbar=dict(title="CSI"),
        )
    )
    fig.update_layout(
        title="Feature Characteristic Stability Index (CSI)",
        xaxis_title="Monitoring period",
        margin=dict(l=120, r=20, t=50, b=50),
        height=360,
        template="plotly_white",
    )
    return fig


def _max_metric(rows, metrics) -> float:
    values = [r[m] for r in rows for m in metrics if isinstance(r.get(m), (int, float))]
    return max(values) if values else 0.35


def figures_to_html(figures: dict, inline_js: bool = False) -> tuple[dict[str, str], str]:
    """Render a mapping of name -> Plotly figure to HTML fragments.

    Returns ``(fragments, plotlyjs)`` where ``fragments`` maps the same names to
    ``<div>`` HTML and ``plotlyjs`` is either a ``<script>`` tag with the CDN
    URL or the inline plotly.js bundle (when ``inline_js=True``).
    """
    import plotly.io as pio

    fragments = {
        name: pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
        for name, fig in figures.items()
    }
    plotlyjs = pio.to_html(
        _blank_figure(), include_plotlyjs="inline" if inline_js else "cdn", full_html=False
    )
    return fragments, plotlyjs


def _blank_figure():
    import plotly.graph_objects as go

    return go.Figure()


# --------------------------------------------------------------------------- #
# Dependency-free SVG fallback
# --------------------------------------------------------------------------- #
def line_chart_svg(
    x_labels: list[str],
    series: dict[str, list[Optional[float]]],
    width: int = 720,
    height: int = 260,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> str:
    """Render one or more line series as a standalone inline SVG string."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 40
    values = [v for vals in series.values() for v in vals if isinstance(v, (int, float))]
    lo = y_min if y_min is not None else (min(values) if values else 0.0)
    hi = y_max if y_max is not None else (max(values) if values else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = max(1, len(x_labels) - 1)

    def sx(i: int) -> float:
        return pad_l + plot_w * (i / n)

    def sy(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="cf-chart" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="11">',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#ddd"/>',
    ]
    for tick in range(5):
        v = lo + (hi - lo) * tick / 4
        y = sy(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{v:.2f}</text>')
    step = max(1, len(x_labels) // 6)
    for i, label in enumerate(x_labels):
        if i % step == 0:
            parts.append(f'<text x="{sx(i):.1f}" y="{height - 12}" text-anchor="middle" fill="#666">{label}</text>')
    palette = ["#1f4e79", "#8e44ad", "#16a085", "#c0392b", "#d99100"]
    for s, (name, vals) in enumerate(series.items()):
        color = palette[s % len(palette)]
        points = " ".join(
            f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals) if isinstance(v, (int, float))
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{pad_l + 8 + s * 90}" y="{pad_t + 14}" fill="{color}">■ {name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[Optional[float]]],
    width: int = 720,
    height: int = 300,
    v_max: float = 0.25,
) -> str:
    """Render a matrix as an inline SVG heat-map."""
    pad_l, pad_t = 120, 30
    cell_w = max(6.0, (width - pad_l - 10) / max(1, len(col_labels)))
    cell_h = max(6.0, (height - pad_t - 10) / max(1, len(row_labels)))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="cf-chart" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="10">'
    ]
    for r, label in enumerate(row_labels):
        parts.append(f'<text x="{pad_l - 6}" y="{pad_t + r * cell_h + cell_h * 0.7:.1f}" text-anchor="end" fill="#555">{label}</text>')
    step = max(1, len(col_labels) // 8)
    for c, label in enumerate(col_labels):
        if c % step == 0:
            parts.append(f'<text x="{pad_l + c * cell_w + cell_w / 2:.1f}" y="{pad_t - 8}" text-anchor="middle" fill="#555">{label}</text>')
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            fill = _heat_color(value, v_max)
            x = pad_l + c * cell_w
            y = pad_t + r * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#fff" stroke-width="0.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def _heat_color(value: Optional[float], v_max: float) -> str:
    if not isinstance(value, (int, float)):
        return "#f0f0f0"
    t = max(0.0, min(1.0, value / v_max))
    if t < 0.4:
        return "#7fbf8f"
    if t < 0.7:
        return "#e0b84c"
    return "#c0392b"


def svg_fallback_charts(results: MonitoringResults) -> dict[str, str]:
    """Build all four views as inline SVG strings (no Plotly required)."""
    _, fact = build_fact_table(results)
    periods = [r["period"] for r in fact]
    csi_svg = line_chart_svg(
        periods,
        {"Gini": [r["gini"] for r in fact], "KS": [r["ks"] for r in fact]},
    )
    psi_svg = line_chart_svg(
        periods,
        {"PSI": [r["psi"] for r in fact], "CSI": [r["csi"] for r in fact]},
        y_min=0,
    )
    _, calib = build_calibration_table(results)
    latest = periods[-1] if periods else None
    calib_rows = [r for r in calib if r["period"] == latest]
    matrix = [[r["observed_pd"] for r in calib_rows]]
    calib_svg = heatmap_svg(
        ["Observed PD"], [r["band"] for r in calib_rows], matrix, v_max=0.5
    )
    _, feat = build_feature_csi_table(results)
    features = sorted({r["feature"] for r in feat})
    csi_index = {(r["feature"], r["period"]): r["csi"] for r in feat}
    csi_matrix = [[csi_index.get((f, p)) for p in periods] for f in features]
    csi_heat = heatmap_svg(features, periods, csi_matrix)
    return {
        "psi_trend": psi_svg,
        "discrimination_trend": csi_svg,
        "calibration": calib_svg,
        "csi_heatmap": csi_heat,
    }


def build_plotly_figures(results: MonitoringResults) -> dict:
    """Build the standard set of Plotly figures for a results object."""
    return {
        "psi_trend": psi_trend_figure(results),
        "discrimination_trend": discrimination_trend_figure(results),
        "calibration": calibration_figure(results),
        "csi_heatmap": csi_heatmap_figure(results),
    }


__all__ = [
    "has_plotly",
    "build_plotly_figures",
    "figures_to_html",
    "svg_fallback_charts",
    "psi_trend_figure",
    "discrimination_trend_figure",
    "calibration_figure",
    "csi_heatmap_figure",
    "line_chart_svg",
    "heatmap_svg",
]
