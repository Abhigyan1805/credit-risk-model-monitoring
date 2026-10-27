"""Render the fact sheet the LLM drafts from, and the retry message carrying the checker's feedback."""

from __future__ import annotations

from commentary.facts import SCALAR_METRICS, MonthFacts, feature_label, fmt

SYSTEM = (
    "You are a model-risk analyst at a bank writing the monthly monitoring commentary for a "
    "probability-of-default (PD) scorecard. You write short, factual, plain-English commentary "
    "and you only state facts that appear in the fact sheet you are given."
)

RULES = """Write the commentary as exactly four short paragraphs, each starting with its label:
What changed: / Alerts: / Likely drivers: / Recommended action:

Rules (a checker verifies every claim and rejects the draft if any rule is broken):
1. Use only numbers that appear in the fact sheet, copied at the precision shown. Do not compute new numbers, percentages or differences.
2. Refer to the month as {month}; the only other month you may name is the previous month {prev}.
3. Use "rose"/"fell" (or "improved"/"deteriorated") only as given in the "change vs previous month" column.
4. State RAG statuses in capitals (GREEN, AMBER, RED) exactly as given. Do not give a status to a metric that has none.
5. Mention only the metrics and features listed in the fact sheet. Metrics shown as n/a are not available this month: do not give them a value.
6. In "Likely drivers", name only features whose CSI is AMBER or RED or that are listed as the top drifting features, and say "likely" or "may" (drift is evidence, not proof).
7. Keep it under 170 words. No headings, no bullet lists, no markdown."""


def _direction_word(f) -> str:
    if f.delta is None:
        return "no previous month" if f.prev_rag is None else "n/a"
    if f.delta == 0:
        return "unchanged"
    moved = "rose" if f.delta > 0 else "fell"
    bad_now, bad_before = f.badness(f.value), f.badness(f.prev)
    if bad_now is None or bad_before is None or bad_now == bad_before:
        return moved
    return f"{moved} ({'deteriorated' if bad_now > bad_before else 'improved'})"


def _thresholds(f) -> str:
    if f.metric == "calibration_ratio":
        return "AMBER if more than 0.15 away from 1.0, RED if more than 0.30 away"
    if f.amber is None or f.red is None:
        return "no threshold (context only)"
    return f"AMBER at {fmt(f.amber, f.metric)}, RED at {fmt(f.red, f.metric)}"


def fact_sheet(facts: MonthFacts) -> str:
    prev = facts.prev_month or "none (first monitored month)"
    lines = [
        f"Month: {facts.month}. Previous month: {prev}. Model: {facts.model_version}.",
        f"Overall status: {facts.overall_rag}"
        + (f" (previous month: {facts.prev_overall_rag})." if facts.prev_overall_rag else "."),
        "Outcomes observable: "
        + (
            "yes."
            if facts.outcomes_observable
            else "no - this vintage has not matured, so Gini, KS, AUC and calibration are n/a; "
            "only population stability can be assessed."
        ),
        "",
        "Model-level metrics (value | previous month | change vs previous month | status | thresholds):",
    ]
    for metric, label in SCALAR_METRICS:
        f = facts.get(metric)
        if f is None:
            continue
        status = f.rag if f.rag != "NA" else "no status"
        lines.append(
            f"- {label}: {fmt(f.value, metric)} | {fmt(f.prev, metric)} | {_direction_word(f)} | "
            f"{status} | {_thresholds(f)}"
        )
        if f.baseline is not None:
            lines.append(f"  ({label} development baseline: {fmt(f.baseline, metric)})")
    lines += ["", "Feature drift, CSI, most drifted first (value | previous month | change | status):"]
    lines.append("  CSI thresholds: AMBER at 0.100, RED at 0.250.")
    for f in facts.csi():
        lines.append(
            f"- {feature_label(f.segment)}: {fmt(f.value, 'csi')} | {fmt(f.prev, 'csi')} | "
            f"{_direction_word(f)} | {f.rag}"
        )
    top = ", ".join(feature_label(f.segment) for f in facts.csi()[:3])
    lines.append(f"Top drifting features: {top}.")
    alerts = facts.alerts()
    lines += ["", f"Headline alerts this month ({len(alerts)}):"]
    for f in alerts:
        name = f"CSI {feature_label(f.segment)}" if f.metric == "csi" else dict(SCALAR_METRICS)[f.metric]
        lines.append(f"- {name}: {f.rag}")
    if not alerts:
        lines.append("- none: every headline metric is GREEN.")
    bands = facts.band_breaches()
    if bands:
        listed = ", ".join(f"band {b.segment.removeprefix('band=')} ({b.rag})" for b in bands)
        lines.append(f"Score bands failing the binomial calibration test: {listed}.")
    return "\n".join(lines)


def user_message(facts: MonthFacts) -> str:
    prev = facts.prev_month or "(none)"
    return f"FACT SHEET\n{fact_sheet(facts)}\n\n{RULES.format(month=facts.month, prev=prev)}"


def retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "The checker rejected your draft. Problems found:\n"
        f"{listed}\n\n"
        "Rewrite the whole commentary, fixing every problem and following all rules. "
        "Only state facts from the fact sheet."
    )


def build_messages(facts: MonthFacts, history: list[tuple[str, list[str]]] | None = None) -> list[dict]:
    """Chat messages: the fact sheet, then (draft, problems) pairs from earlier rejected attempts."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(facts)}]
    for draft, problems in history or []:
        messages.append({"role": "assistant", "content": draft})
        messages.append({"role": "user", "content": retry_message(problems)})
    return messages
