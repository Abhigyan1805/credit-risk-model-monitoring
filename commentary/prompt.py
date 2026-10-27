"""Render the fact sheet the LLM drafts from, and the retry message carrying the checker's feedback."""

from __future__ import annotations

from commentary.facts import SCALAR_METRICS, MonthFacts, feature_label, fmt

SYSTEM = (
    "You are a model-risk analyst at a bank writing the monthly monitoring commentary for a "
    "probability-of-default (PD) scorecard. You write short, factual, plain-English commentary "
    "and you only state facts that appear in the fact sheet you are given."
)

RULES = """Write the commentary as exactly four short paragraphs, each starting with its label, in this order:
What changed: ...
Alerts: ...
Likely drivers: ...
Recommended action: ...
All four paragraphs are required; the commentary is rejected if "Recommended action:" is missing.

Rules (a checker verifies every claim and rejects the draft if any rule is broken):
1. Use only numbers that appear in the fact sheet, copied exactly as shown. Do not compute new numbers, percentages or differences, and never add a % sign (a default rate of 0.129 is written 0.129).
2. Refer to the month as {month}; the only other month you may name is the previous month {prev}.
3. Use "rose"/"fell" (or "improved"/"deteriorated") only exactly as stated in the fact sheet ("so it rose ..."). If there is no previous month, do not describe any change.
4. State RAG statuses in capitals (GREEN, AMBER, RED) exactly as given. Metrics marked "no status" (AUC, mean PD, default rate, volume) must not be called GREEN, AMBER or RED. Only say a status changed if the fact sheet gives last month's status.
5. Mention only the metrics and features listed in the fact sheet. Metrics that are not available this month must not be given a value or a status.
6. In "Likely drivers", name only features whose CSI is AMBER or RED or that are listed as the top drifting features, and say "likely" or "may" (drift is evidence, not proof).
7. Keep it under 150 words; focus on the alerts rather than listing every metric. No headings, no bullet lists, no markdown."""


def _change(f) -> str:
    """'Last month 0.151, so it rose (deteriorated).' with the direction pre-computed for the model."""
    if f.prev_rag is None:
        return ""
    if f.prev is None or f.value is None:
        return ""
    if fmt(f.value, f.metric) == fmt(f.prev, f.metric):
        return f" Last month {fmt(f.prev, f.metric)}, so it is unchanged."
    moved = "rose" if f.value > f.prev else "fell"
    bad_now, bad_before = f.badness(f.value), f.badness(f.prev)
    if bad_now is not None and bad_before is not None and bad_now != bad_before:
        moved += f" ({'deteriorated' if bad_now > bad_before else 'improved'})"
    return f" Last month {fmt(f.prev, f.metric)}, so it {moved}."


def _status(f) -> str:
    return f"{f.rag}" if f.rag in ("GREEN", "AMBER", "RED") else "no status"


def fact_sheet(facts: MonthFacts) -> str:
    prev = facts.prev_month
    lines = [
        f"Month: {facts.month}. Previous month: "
        + (
            prev
            if prev
            else "none (first monitored month): do not describe any change or compare with last month"
        )
        + ".",
        f"Overall status: {facts.overall_rag}"
        + (f" (last month: {facts.prev_overall_rag})." if facts.prev_overall_rag else "."),
    ]
    if not facts.outcomes_observable:
        lines.append(
            "Outcomes are not observable yet (immature vintage): Gini, KS, AUC, calibration ratio and "
            "Hosmer-Lemeshow are not available. Only population stability can be assessed."
        )
    alerts = facts.alerts()
    names = [
        f"{'CSI ' + feature_label(f.segment) if f.metric == 'csi' else dict(SCALAR_METRICS)[f.metric]} {f.rag}"
        for f in alerts
    ]
    lines += [
        "",
        f"HEADLINE ALERTS ({len(alerts)}): "
        + ("; ".join(names) if names else "none, every headline metric is GREEN")
        + ".",
    ]
    bands = facts.band_breaches()
    if bands:
        listed = ", ".join(f"band {b.segment.removeprefix('band=')} ({b.rag})" for b in bands)
        lines.append(f"Score bands failing the binomial calibration test (detail, not headline): {listed}.")
    lines += ["", "MODEL-LEVEL METRICS"]
    for metric, label in SCALAR_METRICS:
        f = facts.get(metric)
        if f is None or not f.available:
            continue
        lines.append(f"- {label} is {fmt(f.value, metric)}, {_status(f)}.{_change(f)}")
    lines += ["", "FEATURE DRIFT (CSI, most drifted first)"]
    csi = facts.csi()
    shown = [f for f in csi if f.rag in ("AMBER", "RED")] or csi[:1]
    shown += [f for f in csi[:3] if f not in shown]
    for f in shown:
        lines.append(f"- {feature_label(f.segment)}: CSI {fmt(f.value, 'csi')}, {f.rag}.{_change(f)}")
    rest = [feature_label(f.segment) for f in csi if f not in shown]
    if rest:
        lines.append(f"- All other features ({', '.join(rest)}): CSI GREEN.")
    lines.append("Top drifting features: " + ", ".join(feature_label(f.segment) for f in csi[:3]) + ".")
    return "\n".join(lines)


def user_message(facts: MonthFacts) -> str:
    prev = facts.prev_month or "(none)"
    return f"FACT SHEET\n{fact_sheet(facts)}\n\n{RULES.format(month=facts.month, prev=prev)}"


def retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "The checker rejected your draft. Problems found:\n"
        f"{listed}\n\n"
        "Write a new commentary from the fact sheet, fixing every problem and following all rules. "
        "Do not reuse the rejected sentences. Only state facts from the fact sheet."
    )


def build_messages(facts: MonthFacts, history: list[tuple[str, list[str]]] | None = None) -> list[dict]:
    """Chat messages: the fact sheet plus, on a retry, the checker's findings on the last rejected draft.

    The findings quote the offending sentences; earlier drafts themselves are not sent, so the context
    stays short and the model writes afresh instead of copying.
    """
    content = user_message(facts)
    for _draft, problems in (history or [])[-1:]:
        # The rejected draft is not replayed as an assistant turn: the model then tends to copy it.
        content += "\n\n" + retry_message(problems)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
