"""Matplotlib charts for the monthly report, rendered as embedded PNGs.

The report must be a single self-contained HTML file, so each chart is drawn on
an in-memory figure and returned as a base64 ``data:`` URI. Matplotlib is a
pinned dependency (see ``pyproject.toml``); the ``Agg`` backend is used so no
display is required (CI-safe).
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from reporting.rag import RAG_COLORS, rag_color  # noqa: E402

_CSI_CMAP = LinearSegmentedColormap.from_list("csi", ["#e8f5ec", "#7fbf8f", "#e0b84c", "#c0392b"])


def _to_data_uri(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def psi_trend(summary: pd.DataFrame, amber: float = 0.10, red: float = 0.25) -> str:
    """PSI (score) and the largest feature CSI over time, with RAG reference bands."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(summary))
    top = float(max(summary["psi_score"].max(), summary["csi_max"].max(), red)) * 1.15
    ax.axhspan(0, amber, color=RAG_COLORS["GREEN"], alpha=0.08)
    ax.axhspan(amber, red, color=RAG_COLORS["AMBER"], alpha=0.10)
    ax.axhspan(red, top, color=RAG_COLORS["RED"], alpha=0.08)
    ax.plot(x, summary["psi_score"], marker="o", color="#1f4e79", label="PSI (score)")
    ax.plot(x, summary["csi_max"], marker="s", color="#8e44ad", label="CSI (max feature)")
    ax.axhline(amber, color=RAG_COLORS["AMBER"], ls="--", lw=1)
    ax.axhline(red, color=RAG_COLORS["RED"], ls="--", lw=1)
    ax.set_ylim(0, top)
    _month_axis(ax, summary)
    ax.set_title("Population / characteristic stability (PSI, CSI)")
    ax.set_ylabel("Index value")
    ax.legend(loc="upper left", fontsize=8)
    return _to_data_uri(fig)


def discrimination_trend(
    summary: pd.DataFrame,
    gini_amber: float | None = None,
    gini_red: float | None = None,
    ks_amber: float | None = None,
    ks_red: float | None = None,
) -> str:
    """Gini and KS over time with their lower-worse reference lines."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = range(len(summary))
    for col, color, label in (("gini", "#1f4e79", "Gini"), ("ks", "#16a085", "KS")):
        if col not in summary:
            continue
        colors = [rag_color(r) for r in summary.get(f"rag_{col}", pd.Series(["NA"] * len(summary)))]
        ax.plot(x, summary[col], color=color, marker="o", label=label)
        ax.scatter(x, summary[col], c=colors, s=42, zorder=3, edgecolors="white", linewidths=0.6)
    if gini_amber is not None:
        ax.axhline(gini_amber, color="#1f4e79", ls=":", lw=1, label="Gini amber")
    if gini_red is not None:
        ax.axhline(gini_red, color="#1f4e79", ls="--", lw=1, label="Gini red")
    if ks_amber is not None:
        ax.axhline(ks_amber, color="#16a085", ls=":", lw=1, label="KS amber")
    if ks_red is not None:
        ax.axhline(ks_red, color="#16a085", ls="--", lw=1, label="KS red")
    _month_axis(ax, summary)
    ax.set_title("Discrimination over time (Gini, KS)")
    ax.set_ylabel("Statistic")
    ax.legend(loc="best", fontsize=8, ncol=2)
    return _to_data_uri(fig)


def calibration_by_band(calib: pd.DataFrame, amber: float = 0.15, red: float = 0.30) -> str:
    """Observed/predicted calibration ratio by band, with a target line at 1.0."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    if calib.empty:
        ax.text(0.5, 0.5, "No calibration data for this month", ha="center", va="center")
        ax.axis("off")
        return _to_data_uri(fig)
    bands = calib["band"].astype(str)
    colors = [rag_color(r) for r in calib["rag_calibration_ratio"].fillna("NA")]
    ax.bar(bands, calib["calibration_ratio"], color=colors)
    ax.axhline(1.0, color="#333", lw=1.2, ls="-")
    ax.axhline(1.0 + amber, color=RAG_COLORS["AMBER"], ls="--", lw=1)
    ax.axhline(1.0 - amber, color=RAG_COLORS["AMBER"], ls="--", lw=1)
    ax.axhline(1.0 + red, color=RAG_COLORS["RED"], ls=":", lw=1)
    ax.axhline(1.0 - red, color=RAG_COLORS["RED"], ls=":", lw=1)
    ax.set_title("Calibration by score band (observed / predicted)")
    ax.set_xlabel("Score band (1 = safest, 10 = riskiest)")
    ax.set_ylabel("Calibration ratio")
    return _to_data_uri(fig)


def csi_heatmap(csi_by_feature: pd.DataFrame, v_max: float = 0.25) -> str:
    """Feature (rows) x month (columns) heat-map of CSI."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    if csi_by_feature.empty:
        ax.text(0.5, 0.5, "No CSI data", ha="center", va="center")
        ax.axis("off")
        return _to_data_uri(fig)
    matrix = csi_by_feature.pivot_table(index="feature", columns="month", values="csi", aggfunc="first")
    image = ax.imshow(matrix.values, aspect="auto", cmap=_CSI_CMAP, vmin=0, vmax=v_max)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    fig.colorbar(image, ax=ax, label="CSI", fraction=0.046, pad=0.02)
    ax.set_title("Feature Characteristic Stability Index (CSI)")
    return _to_data_uri(fig)


def rag_strip(summary: pd.DataFrame) -> str:
    """A one-row-per-month strip coloured by the headline RAG status."""
    fig, ax = plt.subplots(figsize=(7.2, 1.0))
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.barh(0, 1, left=i, color=rag_color(row.get("overall_rag")), edgecolor="white")
    ax.set_xlim(0, len(summary))
    ax.set_yticks([])
    ax.set_xticks([i + 0.5 for i in range(len(summary))])
    ax.set_xticklabels(list(summary["month"]), rotation=90, fontsize=7)
    ax.set_title("Headline status by month", fontsize=10)
    return _to_data_uri(fig)


def _month_axis(ax, summary: pd.DataFrame) -> None:
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(list(summary["month"]), rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
