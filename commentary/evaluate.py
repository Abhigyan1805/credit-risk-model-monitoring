"""Evaluate the AI commentary over every monitored month. ``python -m commentary.evaluate``

Reads the commentary JSONs written by ``commentary.run --all`` and measures:

1. the drafting loop: first-draft pass rate, published path (first pass / after retry / fallback),
   the error types the checker caught, attempts and generation time per month, words published;
2. the checker itself, on seeded *corrupted* drafts: each clean draft (published LLM text and the
   template) gets one deliberate error of a known type; recall = share flagged. False alarms are
   measured on clean drafts written in a second, independent phrasing;
3. residual errors in the published text, by an *independent* audit that re-reads the raw results
   Parquet and shares no code with the checker (every number must exist in the month's results
   table, every month named must be this month or the previous one).

Writes ``data/commentary/eval_summary.json`` and the generated block of ``docs/AI_COMMENTARY.md``.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from commentary.checker import check
from commentary.facts import FEATURE_LABELS, MonthFacts, feature_label, fmt, load_facts
from commentary.run import load_commentary
from commentary.template import template_commentary
from common import paths
from common.config import monitor_months
from common.paths import REPO_ROOT

DOC = REPO_ROOT / "docs" / "AI_COMMENTARY.md"
START, END = "<!-- eval:start -->", "<!-- eval:end -->"
SEED = 20261027

# --------------------------------------------------------------------------- corruptions

_NUM = re.compile(r"(?<![\w.-])(\d+\.\d+)(?![\d-])")
_RAGS = ("GREEN", "AMBER", "RED")
_DIRS = {
    "rose": "fell",
    "fell": "rose",
    "increased": "decreased",
    "decreased": "increased",
    "improved": "deteriorated",
    "deteriorated": "improved",
    "rising": "falling",
    "falling": "rising",
    "declined": "rose",
    "dropped": "rose",
}
_INVENTED = [
    "The Brier score was 0.118.",
    "The approval rate fell to 61% this month.",
    "Expected loss rose to 2.4% of exposure.",
    "Information value for the scorecard is 0.31.",
    "The delinquency rate increased to 3.1%.",
]


def _perturb_number(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    cands = [m for m in _NUM.finditer(text) if not re.match(r"20\d\d", m.group(1))]
    if not cands:
        return None
    m = rng.choice(cands)
    d = len(m.group(1).split(".")[1])
    value = float(m.group(1))
    factor = 1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.5)
    new = f"{value * factor:.{d}f}"
    if new == m.group(1) or value == 0:
        new = f"{value + 0.1:.{d}f}"
    return text[: m.start()] + new + text[m.end() :]


def _swap_rag(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    cands = [m for m in re.finditer(r"\b(GREEN|AMBER|RED)\b", text)]
    # Skip "AMBER or RED" style sets: swapping one member does not make the claim false.
    cands = [m for m in cands if not re.search(r"(?:or|/)\s*$", text[: m.start()][-5:])]
    if not cands:
        return None
    m = rng.choice(cands)
    new = rng.choice([r for r in _RAGS if r != m.group(1)])
    return text[: m.start()] + new + text[m.end() :]


def _flip_direction(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    cands = [m for m in re.finditer(r"\b(" + "|".join(_DIRS) + r")\b", text, re.IGNORECASE)]
    if not cands:
        return None
    m = rng.choice(cands)
    word = m.group(1)
    new = _DIRS[word.lower()]
    new = new.capitalize() if word[0].isupper() else new
    return text[: m.start()] + new + text[m.end() :]


def _shift_month(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    cands = list(re.finditer(r"\b20\d\d-\d\d\b", text))
    if not cands:
        return None
    m = rng.choice(cands)
    shifted = (pd.Period(m.group(0), freq="M") + rng.choice([-12, -6, -3, 2, 4, 9])).strftime("%Y-%m")
    return text[: m.start()] + shifted + text[m.end() :]


def _insert_invented(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    paras = text.split("\n\n")
    i = rng.randrange(len(paras))
    paras[i] = paras[i].rstrip() + " " + rng.choice(_INVENTED)
    return "\n\n".join(paras)


def _false_all_clear(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    if not facts.alerts():
        return None
    claim = rng.choice(["No headline alerts fired this month.", "All headline metrics are within tolerance."])
    return text.replace("Alerts:", "Alerts: " + claim, 1) if "Alerts:" in text else claim + " " + text


def _swap_driver(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    allowed = facts.driver_features()
    bad = [f.segment for f in facts.csi() if f.segment not in allowed and f.rag == "GREEN"]
    m = re.search(r"likely drivers?:", text, re.IGNORECASE)
    if not bad or not m:
        return None
    para_end = text.find("\n\n", m.end())
    para_end = len(text) if para_end == -1 else para_end
    para = text[m.end() : para_end]
    for feat in sorted(allowed, key=lambda f: -len(FEATURE_LABELS[f])):
        label = FEATURE_LABELS[feat]
        found = re.search(re.escape(label), para, re.IGNORECASE) or re.search(
            re.escape(label.split(" (")[0]), para, re.IGNORECASE
        )
        if found:
            new = para[: found.start()] + feature_label(rng.choice(bad)) + para[found.end() :]
            return text[: m.end()] + new + text[para_end:]
    return None


def _value_for_unavailable(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    if facts.outcomes_observable:
        return None
    return text.rstrip() + " " + rng.choice(["Gini was 0.334 this month.", "KS fell to 0.241."])


def _drop_red_alert(text: str, rng: random.Random, facts: MonthFacts) -> str | None:
    """Delete every sentence that mentions a RED model-level metric (an omission)."""
    red = [a for a in facts.alerts() if a.rag == "RED" and a.metric != "csi"]
    if not red:
        return None
    pats = {
        "psi_score": r"\bpsi\b|population stability",
        "gini": r"\bgini\b",
        "ks": r"\bks\b",
        "calibration_ratio": r"calibration|observed[- /]to[- /]predicted",
        "hl_pvalue": r"hosmer|\bhl\b",
    }
    target = rng.choice(red)
    pat = re.compile(pats[target.metric], re.IGNORECASE)
    out = []
    for para in text.split("\n\n"):
        sentences = re.split(r"(?<=[.!?])\s+", para)
        kept = [s for s in sentences if not pat.search(s)]
        out.append(" ".join(kept) if kept else para.split(":")[0] + ":")
        if not kept:
            out[-1] = out[-1]
    new = "\n\n".join(out)
    if pat.search(new) or new == text:
        return None
    return new


CORRUPTIONS = {
    "wrong_number": _perturb_number,
    "wrong_status": _swap_rag,
    "wrong_direction": _flip_direction,
    "wrong_month": _shift_month,
    "invented_metric": _insert_invented,
    "false_all_clear": _false_all_clear,
    "unsupported_driver": _swap_driver,
    "value_for_unavailable_metric": _value_for_unavailable,
    "omitted_red_alert": _drop_red_alert,
}


# --------------------------------------------------------------------------- clean paraphrases


def alt_phrasing(facts: MonthFacts) -> str:
    """A second, independently worded correct commentary (for the checker's false-alarm rate)."""
    m = facts.month
    parts = [f"What changed: The model's overall status for {m} is {facts.overall_rag}."]
    psi = facts.get("psi_score")
    if psi and psi.available:
        if psi.prev is not None and round(psi.value, 2) != round(psi.prev, 2):
            verb = "increased" if psi.value > psi.prev else "decreased"
            parts.append(
                f"Population stability (PSI) {verb} from {psi.prev:.2f} to {psi.value:.2f} and is {psi.rag}."
            )
        else:
            parts.append(f"Population stability (PSI) stands at {psi.value:.2f} ({psi.rag}).")
    for metric, label in (("gini", "Gini"), ("ks", "KS")):
        f = facts.get(metric)
        if f and f.available:
            parts.append(f"{label} is {f.value:.2f}, {f.rag}.")
    cal = facts.get("calibration_ratio")
    if cal and cal.available:
        parts.append(
            f"Observed defaults run at {cal.value:.2f} times the predicted level, so calibration is {cal.rag}."
        )
    dr = facts.get("default_rate")
    if dr and dr.available:
        parts.append(f"The observed default rate was {dr.value * 100:.1f}%.")
    if not facts.outcomes_observable:
        parts.append("Gini and KS are not yet available because outcomes are immature.")
    hl = facts.get("hl_pvalue")
    alerts = facts.alerts()
    a = [f"Alerts: {len(alerts)} headline alerts fired."] if alerts else ["Alerts: No headline alerts fired."]
    if hl and hl.rag in ("AMBER", "RED"):
        a.append(f"The Hosmer-Lemeshow test is {hl.rag} (p-value {fmt(hl.value, 'hl_pvalue')}).")
    reds = [f for f in facts.csi() if f.rag == "RED"]
    if reds:
        a.append("CSI is RED for " + " and ".join(feature_label(f.segment) for f in reds[:2]) + ".")
    top = facts.csi()[0]
    d = f"Likely drivers: Drift in {feature_label(top.segment)} (CSI {top.value:.2f}) may be behind the shift."
    act = "Recommended action: Discuss the findings with the model owner"
    act += " and consider recalibration." if cal and cal.rag == "RED" else "."
    return "\n\n".join([" ".join(parts), " ".join(a), d, act])


# --------------------------------------------------------------------------- independent audit

_AUDIT_NUM = re.compile(r"(?<![\w.-])(\d[\d,]*(?:\.\d+)?)(%?)")


def independent_audit(text: str, month: str) -> list[str]:
    """Second-line check that shares no code with the checker: raw Parquet values only.

    Flags (a) numbers that equal no value, threshold or simple transform (x100, x-1) of any row in
    this or the previous month's results table, at the precision written; (b) months other than
    this one or the previous one; (c) a "no alerts" claim in a month with AMBER/RED rows.
    """
    cur = pd.read_parquet(paths.monitoring_path(month))
    prev_m = (pd.Period(month, freq="M") - 1).strftime("%Y-%m")
    frames = [cur]
    if paths.monitoring_path(prev_m).exists():
        frames.append(pd.read_parquet(paths.monitoring_path(prev_m)))
    allowed = pd.concat(frames)[["value", "amber_threshold", "red_threshold"]].to_numpy().ravel()
    pool = [float(x) for x in allowed if pd.notna(x)]
    pool += [x * 100 for x in pool] + [abs(x - 1) for x in pool] + [abs(x - 1) * 100 for x in pool]
    pool += [0.15, 0.30, 0.85, 1.15, 0.70, 1.30, 1.0, 0.001, 10, 12, 36, 100]
    findings = []
    stripped = re.sub(r"\b20\d\d-\d\d\b|\bbands?\s+[\d,\sand-]+|\b20\d\d\b", " ", text)
    for m in _AUDIT_NUM.finditer(stripped):
        raw = m.group(1).replace(",", "").rstrip(".")
        if not raw:
            continue
        val = float(raw)
        d = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10**-d + 1e-12
        if not any(abs(val - x) <= tol for x in pool) and not (d == 0 and val <= 36):
            findings.append(f"number {m.group(0)} not in results")
    for mm in re.findall(r"\b20\d\d-\d\d\b", text):
        if mm not in (month, prev_m):
            findings.append(f"month {mm}")
    has_alerts = cur["rag"].isin(["AMBER", "RED"]).any()
    if has_alerts and re.search(
        r"\bno (?:headline )?alerts\b|\ball (?:headline )?metrics are green", text, re.I
    ):
        findings.append("all-clear claim with alerts present")
    return findings


# --------------------------------------------------------------------------- report


def _words(text: str) -> int:
    return len(re.findall(r"\b\w[\w.%-]*\b", text))


def evaluate(months: list[str] | None = None, seed: int = SEED) -> dict:
    months = months or [m for m in monitor_months() if paths.commentary_path(m).exists()]
    rng = random.Random(seed)
    records = [load_commentary(m) for m in months]
    facts = {m: load_facts(m) for m in months}

    llm = [r for r in records if r["attempts"]]
    first_pass = [r for r in llm if r["attempts"][0]["passed"]]
    path_counts = Counter(r["path"] for r in records)
    first_kinds = Counter(i["kind"] for r in llm for i in r["attempts"][0]["issues"])
    all_kinds = Counter(i["kind"] for r in llm for a in r["attempts"] for i in a["issues"])
    first_drafts_with = Counter(k for r in llm for k in {i["kind"] for i in r["attempts"][0]["issues"]})
    gen = [a["seconds"] for r in llm for a in r["attempts"]]
    per_month = [r["generation_seconds"] for r in llm]
    published_llm = [r for r in records if r["path"].startswith("llm")]

    # checker on corrupted drafts
    clean_sources = [("template", m, template_commentary(facts[m])) for m in months]
    clean_sources += [("llm", r["month"], r["text"]) for r in published_llm]
    corrupt_rows = []
    for source, m, text in clean_sources:
        if not check(text, facts[m]).passed:
            continue  # only corrupt texts the checker accepts, so the inserted error is the only one
        for name, fn in CORRUPTIONS.items():
            bad = fn(text, rng, facts[m])
            if bad is None or bad == text:
                continue
            result = check(bad, facts[m])
            corrupt_rows.append(
                {"source": source, "month": m, "corruption": name, "caught": not result.passed}
            )
    cdf = pd.DataFrame(corrupt_rows)
    recall = cdf.groupby("corruption")["caught"].agg(["mean", "size"]) if len(cdf) else pd.DataFrame()

    # false alarms on clean, independently phrased drafts
    alt = [(m, alt_phrasing(facts[m])) for m in months]
    alt_flagged = {m: check(t, facts[m]).feedback() for m, t in alt}
    false_alarms = {m: f for m, f in alt_flagged.items() if f}

    # independent audit of what was published
    audit = {r["month"]: independent_audit(r["text"], r["month"]) for r in records}
    audit_llm = {m: f for m, f in audit.items() if f and load_commentary(m)["path"].startswith("llm")}

    summary = {
        "months": len(records),
        "model": llm[0]["model"] if llm else None,
        "llm_months": len(llm),
        "first_draft_pass": len(first_pass),
        "path_counts": dict(path_counts),
        "first_draft_issue_counts": dict(first_kinds.most_common()),
        "first_drafts_with_issue_kind": dict(first_drafts_with.most_common()),
        "all_attempt_issue_counts": dict(all_kinds.most_common()),
        "attempts_total": sum(len(r["attempts"]) for r in llm),
        "seconds_per_attempt_median": round(statistics.median(gen), 1) if gen else None,
        "seconds_per_attempt_max": round(max(gen), 1) if gen else None,
        "seconds_per_month_median": round(statistics.median(per_month), 1) if per_month else None,
        "seconds_per_month_mean": round(statistics.mean(per_month), 1) if per_month else None,
        "seconds_total": round(sum(per_month), 1),
        "words_published_llm_median": statistics.median([_words(r["text"]) for r in published_llm])
        if published_llm
        else None,
        "words_published_template_median": statistics.median(
            [_words(template_commentary(facts[m])) for m in months]
        ),
        "claims_checked_per_published_median": statistics.median([r["n_claims"] for r in records]),
        "corrupted_drafts": len(cdf),
        "corrupted_caught": int(cdf["caught"].sum()) if len(cdf) else 0,
        "recall_by_corruption": {k: [round(v["mean"], 3), int(v["size"])] for k, v in recall.iterrows()},
        "recall_by_source": cdf.groupby("source")["caught"].mean().round(3).to_dict() if len(cdf) else {},
        "clean_alt_drafts": len(alt),
        "clean_alt_false_alarms": len(false_alarms),
        "false_alarm_examples": {m: f[:3] for m, f in list(false_alarms.items())[:5]},
        "independent_audit_findings": {m: f for m, f in audit.items() if f},
        "independent_audit_llm_months_flagged": len(audit_llm),
    }
    return summary


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def render(s: dict) -> str:
    n, nl = s["months"], s["llm_months"]
    pc = s["path_counts"]
    lines = [
        START,
        "<!-- generated by `python -m commentary.evaluate`; do not edit by hand -->",
        f"Months evaluated: **{n}**. Drafting model: `{(s['model'] or {}).get('model_id', 'n/a')}` "
        f"({(s['model'] or {}).get('quantization', '')}, greedy decoding).",
        "",
        "| drafting loop | value |",
        "|---|---|",
        f"| first drafts that passed the checker | {_pct(s['first_draft_pass'], nl)} |",
        f"| published: LLM, first draft | {_pct(pc.get('llm_first_pass', 0), n)} |",
        f"| published: LLM, after checker feedback (max 2 retries) | {_pct(pc.get('llm_after_retry', 0), n)} |",
        f"| published: deterministic template fallback | {_pct(pc.get('template_fallback', 0), n)} |",
        f"| LLM drafts generated in total | {s['attempts_total']} |",
        f"| generation time per draft, median / max | {s['seconds_per_attempt_median']} s / "
        f"{s['seconds_per_attempt_max']} s |",
        f"| time per month incl. retries, median / mean | {s['seconds_per_month_median']} s / "
        f"{s['seconds_per_month_mean']} s |",
        f"| total generation time, all months | {s['seconds_total'] / 60:.1f} min |",
        f"| words per published LLM commentary, median | {s['words_published_llm_median']} |",
        f"| words per template commentary, median | {s['words_published_template_median']} |",
        f"| claims checked per published commentary, median | {s['claims_checked_per_published_median']} |",
        "",
        "Error types the checker found in the **first drafts** (months with at least one / total issues):",
        "",
        "| issue kind | first drafts with it | issues in first drafts | issues in all drafts |",
        "|---|---|---|---|",
    ]
    for kind, cnt in s["all_attempt_issue_counts"].items():
        lines.append(
            f"| {kind} | {s['first_drafts_with_issue_kind'].get(kind, 0)} | "
            f"{s['first_draft_issue_counts'].get(kind, 0)} | {cnt} |"
        )
    lines += [
        "",
        f"Checker recall on **{s['corrupted_drafts']} seeded corrupted drafts** (one deliberate error each, "
        f"inserted into checker-accepted texts; caught {_pct(s['corrupted_caught'], s['corrupted_drafts'])}):",
        "",
        "| corruption | recall | drafts |",
        "|---|---|---|",
    ]
    for kind, (rec, size) in s["recall_by_corruption"].items():
        lines.append(f"| {kind} | {rec:.0%} | {size} |")
    by_src = ", ".join(f"{k} {v:.0%}" for k, v in s["recall_by_source"].items())
    lines += [
        "",
        f"Recall by source text: {by_src}.",
        "",
        f"False alarms on clean, independently phrased drafts: "
        f"**{_pct(s['clean_alt_false_alarms'], s['clean_alt_drafts'])}** flagged.",
    ]
    for m, f in s["false_alarm_examples"].items():
        lines.append(f"* {m}: " + "; ".join(f"`{x}`" for x in f))
    audit = s["independent_audit_findings"]
    lines += [
        "",
        f"Independent audit of the **published** text (raw-Parquet re-check, no shared code): "
        f"{len(audit)} of {n} months flagged ({s['independent_audit_llm_months_flagged']} of them LLM-published).",
    ]
    for m, f in audit.items():
        lines.append(f"* {m}: " + "; ".join(f))
    lines.append(END)
    return "\n".join(lines)


def write_doc(block: str, doc: Path = DOC) -> None:
    text = doc.read_text(encoding="utf-8")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    doc.write_text(pre + block + post, encoding="utf-8")


def main() -> None:
    summary = evaluate()
    out = paths.ensure_parent(paths.data_dir() / "commentary" / "eval_summary.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    block = render(summary)
    write_doc(block)
    print(block)


if __name__ == "__main__":
    main()
