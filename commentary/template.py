"""Deterministic template commentary: the fallback when no LLM draft passes the checker.

Built only from the fact sheet with fixed wording, so it is always verifiable (the checker is run on
it too) but reads mechanically and offers no judgement beyond simple rules.
"""

from __future__ import annotations

from commentary.facts import MetricFact, MonthFacts, feature_label, fmt

_LABELS = {
    "psi_score": "score PSI",
    "gini": "Gini",
    "ks": "KS",
    "calibration_ratio": "calibration ratio",
    "hl_pvalue": "Hosmer-Lemeshow p-value",
}


def _move(f: MetricFact, prev_month: str | None) -> str:
    if f.prev is None or prev_month is None:
        return ""
    if fmt(f.value, f.metric) == fmt(f.prev, f.metric):
        return f", unchanged from {prev_month}"
    word = "rose" if (f.value or 0) > f.prev else "fell"
    return f" ({word} from {fmt(f.prev, f.metric)} in {prev_month})"


def _metric_sentence(f: MetricFact, prev_month: str | None) -> str:
    label = _LABELS[f.metric]
    return f"The {label} is {fmt(f.value, f.metric)}, {f.rag}{_move(f, prev_month)}."


def _actions(facts: MonthFacts) -> list[str]:
    alerts = {(a.metric, a.rag) for a in facts.alerts()}
    out = []
    if ("calibration_ratio", "RED") in alerts or ("hl_pvalue", "RED") in alerts:
        out.append("Review PD calibration with the model owner and assess whether a recalibration is needed.")
    if ("psi_score", "RED") in alerts or any(m == "csi" and r == "RED" for m, r in alerts):
        out.append("Investigate the population shift with the business before relying on the scores.")
    if any(m in ("gini", "ks") and r == "RED" for m, r in alerts):
        out.append("Escalate the loss of discrimination to model risk and consider redevelopment.")
    if not out and alerts:
        out.append("Keep the AMBER metrics under review next month.")
    if not alerts:
        out.append("No action beyond routine monitoring.")
    if not facts.outcomes_observable:
        out.append("Assess discrimination and calibration once this vintage's outcomes mature.")
    return out


def template_commentary(facts: MonthFacts) -> str:
    m, prev = facts.month, facts.prev_month
    changed = [f"In {m} the overall status is {facts.overall_rag}."]
    for metric in ("psi_score", "gini", "ks", "calibration_ratio"):
        f = facts.get(metric)
        if f is not None and f.available:
            changed.append(_metric_sentence(f, prev))
    if not facts.outcomes_observable:
        changed.append("Outcomes are not yet observable, so Gini, KS and calibration are not available.")

    alerts = facts.alerts()
    if alerts:
        names = []
        for a in alerts:
            name = f"CSI {feature_label(a.segment)}" if a.metric == "csi" else _LABELS[a.metric]
            names.append(f"{name} {a.rag}")
        alert_text = f"{len(alerts)} headline alerts: " + "; ".join(names) + "."
    else:
        alert_text = "Every headline metric is GREEN."
    bands = facts.band_breaches()
    if bands:
        alert_text += (
            " The binomial test flags score "
            + ", ".join(f"band {b.segment.removeprefix('band=')} {b.rag}" for b in bands)
            + "."
        )

    drifted = [c for c in facts.csi() if c.rag in ("AMBER", "RED")][:3] or facts.csi()[:1]
    drivers = "; ".join(f"{feature_label(c.segment)} (CSI {fmt(c.value, 'csi')}, {c.rag})" for c in drifted)
    driver_text = f"The most drifted features are {drivers}, which may explain the score shift."

    return "\n\n".join(
        [
            "What changed: " + " ".join(changed),
            "Alerts: " + alert_text,
            "Likely drivers: " + driver_text,
            "Recommended action: " + " ".join(_actions(facts)),
        ]
    )
