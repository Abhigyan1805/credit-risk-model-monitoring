"""RAG (Red/Amber/Green) helpers shared by the report and the extracts.

The monitoring engine (builder #3) already computes a ``rag`` value for every
metric row in ``data/monitoring/results/month=<YYYY-MM>.parquet`` using the
thresholds in ``config/thresholds.yaml``. Reporting does **not** re-derive those
statuses; it renders them so the rule stays defined in exactly one place.

These helpers cover the two things reporting still needs:

* turning a metric's value + thresholds + direction into a status, for the few
  derived tables that have no producer row (e.g. a share change); and
* combining several statuses into one headline status.

Status strings are the contract's uppercase values: ``GREEN``, ``AMBER``,
``RED`` and ``NA`` (metric not computable).
"""

from __future__ import annotations

from collections.abc import Iterable

RAG_VALUES: tuple[str, ...] = ("GREEN", "AMBER", "RED", "NA")

#: Ordering used to find the worst status; ``NA`` never outranks a real status.
RAG_ORDER: dict[str, int] = {"NA": 0, "GREEN": 1, "AMBER": 2, "RED": 3}

#: Colours shared by matplotlib charts and the HTML report (match the spec).
RAG_COLORS: dict[str, str] = {
    "GREEN": "#1b8a4b",
    "AMBER": "#d99100",
    "RED": "#c0392b",
    "NA": "#6b7280",
}


def rag_from_value(
    value: float | None,
    amber: float | None,
    red: float | None,
    direction: str,
) -> str:
    """Classify ``value`` against thresholds for a ``higher_worse``/``lower_worse`` metric.

    ``higher_worse`` (PSI, CSI, calibration gap): large is bad, so ``red`` is the
    higher threshold. ``lower_worse`` (Gini, KS): small is bad. Returns ``"NA"``
    when the value or an applicable threshold is missing. Out-of-range directions
    raise ``ValueError``.
    """
    if value is None:
        return "NA"
    if direction == "higher_worse":
        if red is not None and value >= red:
            return "RED"
        if amber is not None and value >= amber:
            return "AMBER"
        return "GREEN"
    if direction == "lower_worse":
        if red is not None and value <= red:
            return "RED"
        if amber is not None and value <= amber:
            return "AMBER"
        return "GREEN"
    raise ValueError(f"Unknown direction: {direction!r}")


def worst_rag(values: Iterable[str]) -> str:
    """Return the worst of several statuses, treating ``NA`` as the lowest."""
    cleaned = [v if v in RAG_ORDER else "NA" for v in values]
    if not cleaned:
        return "NA"
    return max(cleaned, key=lambda v: RAG_ORDER[v])


def is_breach(rag: str | None) -> bool:
    """True for the amber/red statuses that belong in a breach list."""
    return rag in ("AMBER", "RED")


def rag_color(rag: str | None) -> str:
    """Hex colour for a status, defaulting to grey for unknown/NA."""
    return RAG_COLORS.get((rag or "NA").upper(), RAG_COLORS["NA"])
