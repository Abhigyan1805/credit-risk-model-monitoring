"""Verify a commentary draft against the month's computed monitoring results.

The checker is deterministic and rule-based on purpose: it is the control that decides whether text
reaches the report, so it must be auditable and must not share the drafting model's failure modes.

It splits the draft into sentences and clauses, finds every *mention* (a metric name, a CSI feature,
a score band) and every *claim item* (a number, a RAG status, a direction word, a threshold
comparison, a month, an "is not available" phrase), attaches each item to the mention it is about,
and verifies it against :class:`commentary.facts.MonthFacts`.

Tolerances (explicit, see ``docs/AI_COMMENTARY.md``):

* a number must equal a true value correctly rounded to the precision written: ``|claimed - true|
  <= 0.5 * 10**-decimals`` (scientific notation: half a unit of the last mantissa digit) (percent claims are compared on the percent scale); "about"/"around"
  widens this to 2% of the true value;
* "stable"/"unchanged" needs ``|change| <= max(0.005, 2% of last month)``; "rose"/"fell" needs the sign
  of the change to match; "improved"/"deteriorated" uses the metric's risk direction (for the
  calibration ratio, distance from 1.0);
* a RAG word must equal the stored RAG; "above/below the red/amber threshold" must be consistent with it.

Anything that cannot be attached to a fact, or is attached to a fact that is not available this
month, is flagged rather than waved through.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from commentary.facts import FEATURE_LABELS, MetricFact, MonthFacts

ISSUE_KINDS = (
    "wrong_number",
    "unsupported_number",
    "wrong_direction",
    "wrong_status",
    "wrong_month",
    "invented_metric",
    "unavailable_metric",
    "unsupported_driver",
    "contradiction",
    "omission",
)


@dataclass(frozen=True)
class Issue:
    kind: str
    claim: str
    detail: str

    def feedback(self) -> str:
        return f'{self.kind.replace("_", " ")}: "{self.claim}" - {self.detail}'


@dataclass
class CheckResult:
    issues: list[Issue] = field(default_factory=list)
    n_claims: int = 0  # claim items that were verified

    @property
    def passed(self) -> bool:
        return not self.issues

    def kinds(self) -> list[str]:
        return [i.kind for i in self.issues]

    def feedback(self) -> list[str]:
        return [i.feedback() for i in self.issues]


# --------------------------------------------------------------------------- lexicon

_FLAGS = re.IGNORECASE
# (metric, segment, pattern). Order matters: earlier patterns claim their span first.
_METRIC_PATTERNS: list[tuple[str, str | None, str]] = [
    (
        "hl_pvalue",
        "ALL",
        r"hosmer[- ]lemeshow(?: calibration)?(?: test)?(?: p-?value)?|\bh-?l (?:test|p-?value)\b",
    ),
    ("binomial_pvalue", None, r"\bbinomial(?: calibration)?(?: tests?)?(?: p-?values?)?"),
    (
        "mean_pd",
        "ALL",
        r"\b(?:mean|average) (?:predicted )?(?:pd|probability of default)\b|\bmean pd\b"
        r"|\bpredicted (?:default rate|pd)\b|\b(?:mean|average) predicted default rate\b",
    ),
    ("default_rate", "ALL", r"\b(?:observed |actual |realised |realized )?(?:default|bad) rate\b"),
    (
        "calibration_ratio",
        "ALL",
        r"\bcalibration ratio\b|\bobserved[- ]to[- ]predicted(?: ratio)?|observed\s*/\s*predicted(?: ratio)?"
        r"|\bo/e(?: ratio)?\b|\bcalibration\b",
    ),
    ("psi_score", "ALL", r"\b(?:score )?psi\b|\bpopulation stability(?: index)?\b"),
    ("csi", None, r"\bcsi\b|\bcharacteristic stability(?: index)?\b"),
    ("gini", "ALL", r"\bgini(?: coefficient)?\b"),
    ("ks", "ALL", r"\bks\b(?: statistic)?|\bkolmogorov[- ]smirnov\b"),
    ("auc", "ALL", r"\bauc\b|\barea under the (?:roc )?curve\b"),
    ("volume", "ALL", r"\bvolume\b|\bnumber of loans\b"),
    (
        "overall_rag",
        "ALL",
        r"\boverall (?:status|rag|rating|health|model status|model health)\b|\bheadline status\b"
        r"|\bmodel status\b|\boverall\b(?= (?:is|was|remains|remained|stays|stayed)\b)",
    ),
]
_FEATURE_PATTERNS: dict[str, str] = {
    "loan_amnt": r"\bloan amounts?\b|\bloan sizes?\b|\bloan_amnt\b",
    "int_rate": r"\binterest rates?\b|\bint_rate\b",
    "dti": r"\bdebt[- ]to[- ]income(?: ratio)?(?: \(dti\))?|\bdti\b",
    "annual_inc": r"(?<!-)\b(?:annual )?incomes?\b(?! verification)|\bannual_inc\b",
    "fico": r"\bfico(?: scores?)?\b",
    "revol_util": r"\brevolving (?:credit )?utili[sz]ation\b|\brevol_util\b",
    "credit_history_years": r"\bcredit history(?: length)?\b|\blength of credit history\b|\bcredit_history_years\b",
    "emp_length_years": r"\bemployment length\b|\bemp_length(?:_years)?\b|\blength of employment\b",
    "grade": r"\b(?:loan )?grades?\b|\bgrade mix\b",
    "home_ownership": r"\bhome[- _]ownership\b",
    "verification_status": r"\b(?:income )?verification(?: status)?\b|\bverification_status\b",
    "purpose": r"\b(?:loan )?purposes?\b",
}
_INVENTED = re.compile(
    r"\b(?:brier(?: score)?|r-?squared|r²|rmse|mape|log[- ]?loss|f1(?: score)?|precision|recall|accuracy(?: ratio)?"
    r"|information value|vif|cap curve|kl divergence|kullback|wasserstein|jensen[- ]shannon|lgd|ead|expected loss"
    r"|rwa|approval rate|delinquency rate|loss rate|charge[- ]off rate|net interest margin|value at risk|loss given default"
    r"|exposure at default|odds ratio|herfindahl|hhi|loan terms?|sub-?grades?|installments?|addr_state"
    r"|(?:borrower )?state mix|public records?|revolving balances?|open accounts|total accounts|inquiries"
    r"|delinquenc(?:y|ies)|employment titles?)\b",
    _FLAGS,
)
_KNOWN_ACRONYMS = {
    "PSI", "CSI", "KS", "AUC", "PD", "PDS", "HL", "RAG", "ROC", "DTI", "FICO", "RED", "AMBER", "GREEN",
    "OOT", "GINI", "N/A", "NA", "US", "LC", "II", "AI", "ALL", "O/E", "OK",
}  # fmt: skip
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")

_MONTHS = {
    m: i + 1
    for i, ms in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    for m in (ms,)
}
_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_ISO_MONTH = re.compile(r"\b(20\d\d)[-/](0[1-9]|1[0-2])(?:-\d\d)?\b")
_NAMED_MONTH = re.compile(rf"\b({_MONTH_NAMES})\.?,?\s+(20\d\d)\b", _FLAGS)
_BARE_MONTH = re.compile(
    r"\b(january|february|march|april|june|july|august|september|october|november|december)\b", _FLAGS
)
_YEAR = re.compile(r"\b(20\d\d)\b")

_NUMBER = re.compile(
    r"(?<![\w.=/-])([-+−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:[eE]([-+−]?\d+))?"
    r"(\s*(?:%|per ?cent\b|percentage points?\b|pp\b))?"
)
_BANDS = re.compile(
    r"\b(?:score )?bands?\s*=?\s*(\d+(?:\s*(?:,|and|&|to|-|–|or)\s*(?:band\s*)?\d+)*)", _FLAGS
)
_COUNT_WORDS = {
    w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())
}
_COUNT = re.compile(
    r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:of the \d+ |of \d+ )?(?:(red|amber|green) |headline |monitored |score |more )?"
    r"(alerts?|breaches|features?|characteristics|metrics|bands|variables)\b",
    _FLAGS,
)
_CONSTANT_UNIT = re.compile(r"^\s*-?\s*(?:months?|years?|bins|deciles|days|month)\b", _FLAGS)

_RAG_WORD = re.compile(r"\b(red|amber|green)\b", _FLAGS)
_RAG_SET = re.compile(r"\b(amber|red)\s*(?:/|or|and)\s*(amber|red)\b", _FLAGS)
_THRESHOLD_CMP = re.compile(
    r"\b(above|over|exceed\w*|beyond|cross\w*|breach\w*|past|through|below|under|within|at)\s+"
    r"(?:the |its |their |both )?(?:respective )?(red|amber)\s+(?:thresholds?|limits?|levels?|triggers?|lines?)",
    _FLAGS,
)
_THRESHOLD_REF = re.compile(
    r"\b(red|amber|green)\s+(?:thresholds?|limits?|levels?|triggers?|lines?|zone)", _FLAGS
)
_WITHIN = re.compile(r"\bwithin (?:tolerance|limits|thresholds|appetite)\b", _FLAGS)
_BREACH = re.compile(r"\bbreach(?:es|ed)?\b|\bbreaching\b", _FLAGS)
_BASELINE_CMP = re.compile(
    r"\b(above|over|below|under)\s+(?:the |its )?(?:development |dev )?baseline\b", _FLAGS
)
_NUM_CMP = re.compile(
    r"(?:\b(above|over|exceed(?:s|ed|ing)?|beyond|more than|greater than|higher than|below|under|less than|lower than)"
    r"|(<=|>=|<|>|≤|≥))\s*(?:the |a |its )?(?:level of |value of )?$",
    _FLAGS,
)
_UP = r"rose|risen|rises|rise|rising|increased|increases|increase|increasing|climbed|grew|jumped|edged up|went up|ticked up"
_DOWN = r"fell|fallen|falls|falling|decreased|decreases|decrease|decreasing|declined|declines|decline|declining|dropped|drops|eased|slipped|ticked down|edged down"
_WORSE = (
    r"deteriorated|deteriorating|deteriorates|deterioration|worsened|worsening|weakened|weakening|degraded"
)
_BETTER = r"improved|improving|improves|improvement|strengthened|recovered|recovering"
_FLAT = r"stable|unchanged|flat|steady|held steady|broadly stable|little changed"
_DIRECTION = re.compile(rf"\b({_UP}|{_DOWN}|{_WORSE}|{_BETTER}|{_FLAT})\b", _FLAGS)
_DIR_CLASS = [("up", _UP), ("down", _DOWN), ("worse", _WORSE), ("better", _BETTER), ("flat", _FLAT)]
_UNAVAILABLE = re.compile(
    r"\b(?:not (?:yet )?(?:available|observable|computable|computed|assessable|measurable|assessed)"
    r"|unavailable|n/a|cannot (?:yet )?be (?:computed|assessed|measured)|not yet matured|immature)\b",
    _FLAGS,
)
_NO_ALERTS = re.compile(
    r"\bno (?:headline )?(?:(red|amber) )?(?:alerts?|breaches|warnings)\b(?! (?:other|besides|apart))"
    r"|\ball (?:headline )?(?:metrics|indicators|checks)\b[^.;]*\b(?:green|within (?:tolerance|limits))\b",
    _FLAGS,
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|neither|nor|unlikely|rather than|ruled out|isn't|aren't|didn't|without)\b", _FLAGS
)
_DRIVER_CUE = re.compile(
    r"\b(?:driv(?:en|er|ers|ing|es)|due to|caused|causing|because|reflect\w*|attribut\w*|stem\w*|owing to"
    r"|linked to|associated with|explain\w*|contribut\w*|likely source|consistent with)\b",
    _FLAGS,
)
_PREV_CONTEXT = re.compile(
    r"\bfrom\b|\bprevious\b|\bprior\b|\blast month\b|\bearlier\b|\bwas\b|\bpreviously\b", _FLAGS
)
_THRESHOLD_CONTEXT = re.compile(
    r"threshold|limit|trigger|\bamber\b|\bred\b|above|below|exceed|cross|breach|tolerance|away from", _FLAGS
)
_BASELINE_CONTEXT = re.compile(r"\bbaseline\b|\bdevelopment\b|\bdev\b|\bat launch\b", _FLAGS)
_APPROX = re.compile(
    r"\b(?:about|around|approximately|roughly|nearly|almost|some|close to)\s*$|~\s*$", _FLAGS
)
_CLAUSE_SPLIT = re.compile(
    r";|,\s*(?:while|whereas|but|and|although|though)\b|\bwhile\b|\bwhereas\b|\bbut\b", _FLAGS
)


@dataclass
class _Mention:
    start: int
    end: int
    keys: list[tuple[str, str | None]]  # (metric, segment); segment None = unspecified CSI / bands


@dataclass
class _Item:
    start: int
    end: int
    kind: str  # number | rag | direction | cmp | baseline_cmp | unavailable
    text: str
    data: dict


def _overlaps(spans: list[tuple[int, int]], s: int, e: int) -> bool:
    return any(s < b and a < e for a, b in spans)


def _sentences(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"[^\n]+", text):
        line, base = m.group(0), m.start()
        start = 0
        for sm in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(])", line):
            out.append((base + start, line[start : sm.start()]))
            start = sm.end()
        out.append((base + start, line[start:]))
    return [(o, s) for o, s in out if s.strip()]


def _clauses(sentence: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in _CLAUSE_SPLIT.finditer(sentence):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(sentence)))
    return bounds


# --------------------------------------------------------------------------- checker


class Checker:
    """Verifies drafts for one month. ``check(text)`` returns a :class:`CheckResult`."""

    def __init__(self, facts: MonthFacts):
        self.facts = facts
        self.allowed_months = {facts.month} | ({facts.prev_month} if facts.prev_month else set())

    # ---- public
    def check(self, text: str) -> CheckResult:
        result = CheckResult()
        text = text.replace("−", "-").replace("–", "-").replace("**", "")
        driver_para = self._driver_paragraph_span(text)
        mentioned: set[tuple[str, str | None]] = set()
        for offset, sentence in _sentences(text):
            in_driver_para = driver_para is not None and driver_para[0] <= offset < driver_para[1]
            self._check_sentence(sentence, result, mentioned, in_driver_para)
        self._check_omissions(text, mentioned, result)
        return result

    # ---- sentence-level
    def _check_sentence(self, s: str, result: CheckResult, mentioned: set, in_driver_para: bool) -> None:
        excluded: list[tuple[int, int]] = []
        self._check_months(s, result, excluded)
        band_mentions = self._band_mentions(s, excluded)
        self._check_counts(s, result, excluded)
        self._check_invented(s, result)
        mentions = self._mentions(s) + band_mentions
        mentions.sort(key=lambda m: m.start)
        for m in mentions:
            mentioned.update(m.keys)
        self._check_no_alert_claims(s, result)
        self._check_drivers(s, mentions, result, in_driver_para)
        items = self._items(s, excluded)
        clauses = _clauses(s)
        groups = self._group(s, mentions)
        self._check_band_breaches(s, groups, result)
        for item in items:
            target = self._attach(item, groups, clauses)
            self._verify(item, target, s, clauses, result)

    def _check_band_breaches(self, s: str, groups: list[_Mention], result: CheckResult) -> None:
        """'The binomial test flags bands 8 and 9': every named band must actually fail the test."""
        if not re.search(r"\b(?:fail\w*|flag\w*|breach\w*|reject\w*)\b", s, _FLAGS):
            return
        for g in groups:
            for metric, segment in g.keys:
                if metric != "binomial_pvalue" or segment is None:
                    continue
                f = self.facts.get(metric, segment)
                result.n_claims += 1
                if f is None or f.rag not in ("AMBER", "RED"):
                    result.issues.append(
                        Issue("wrong_status", segment.replace("=", " "), "this band passes the binomial test")
                    )

    def _check_months(self, s: str, result: CheckResult, excluded: list) -> None:
        found: list[tuple[str, str]] = []
        for m in _ISO_MONTH.finditer(s):
            found.append((f"{m.group(1)}-{m.group(2)}", m.group(0)))
            excluded.append(m.span())
        for m in _NAMED_MONTH.finditer(s):
            if _overlaps(excluded, *m.span()):
                continue
            num = _MONTHS[m.group(1).lower()[:3]]
            found.append((f"{m.group(2)}-{num:02d}", m.group(0)))
            excluded.append(m.span())
        for month, claim in found:
            result.n_claims += 1
            if month not in self.allowed_months:
                result.issues.append(
                    Issue("wrong_month", claim, f"this commentary covers {self._months_text()} only")
                )
        allowed_names = {self._month_name(m) for m in self.allowed_months}
        for m in _BARE_MONTH.finditer(s):
            if _overlaps(excluded, *m.span()):
                continue
            result.n_claims += 1
            if m.group(1).lower() not in allowed_names:
                result.issues.append(
                    Issue("wrong_month", m.group(0), f"this commentary covers {self._months_text()} only")
                )
        allowed_years = {m[:4] for m in self.allowed_months}
        for m in _YEAR.finditer(s):
            if _overlaps(excluded, *m.span()):
                continue
            excluded.append(m.span())
            result.n_claims += 1
            dev_ok = m.group(1) in ("2012", "2013") and _BASELINE_CONTEXT.search(s)
            if m.group(1) not in allowed_years and not dev_ok:
                result.issues.append(
                    Issue("wrong_month", m.group(0), f"this commentary covers {self._months_text()} only")
                )

    def _months_text(self) -> str:
        return " and ".join(sorted(self.allowed_months))

    @staticmethod
    def _month_name(month: str) -> str:
        names = "january february march april may june july august september october november december"
        return names.split()[int(month[5:7]) - 1]

    def _band_mentions(self, s: str, excluded: list) -> list[_Mention]:
        out = []
        for m in _BANDS.finditer(s):
            excluded.append(m.span())
            nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
            if re.search(r"\d\s*(?:to|-)\s*\d", m.group(1)) and len(nums) == 2:
                nums = list(range(nums[0], nums[1] + 1))
            out.append(_Mention(m.start(), m.end(), [("band", f"band={n}") for n in nums]))
        return out

    def _check_counts(self, s: str, result: CheckResult, excluded: list) -> None:
        for m in _COUNT.finditer(s):
            if _overlaps(excluded, *m.span(1)):
                continue
            excluded.append(m.span(1))
            raw, rag, noun = m.group(1).lower(), (m.group(2) or "").upper(), m.group(3).lower()
            n = int(raw) if raw.isdigit() else _COUNT_WORDS[raw]
            trailing = re.match(
                r"(?:\s*(?:are|is|were|was|remain|remains|stay|stays|now|in|at|flagged|showing))*\s*(red|amber|green)\b",
                s[m.end() : m.end() + 40],
                _FLAGS,
            )
            if not rag and trailing:
                rag = trailing.group(1).upper()
            ok = self._count_options(noun, rag, s)
            result.n_claims += 1
            if n not in ok:
                result.issues.append(
                    Issue("wrong_number", m.group(0), f"the results support a count of {sorted(ok)} here")
                )

    def _count_options(self, noun: str, rag: str, s: str) -> set[int]:
        f = self.facts
        alerts = f.alerts()
        csi = f.csi()
        if noun.startswith(("alert", "breach")):
            if rag:
                return {sum(a.rag == rag for a in alerts)}
            return {len(alerts), sum(a.rag == "RED" for a in alerts), sum(a.rag == "AMBER" for a in alerts)}
        if noun.startswith("band"):
            bands = f.band_breaches()
            return (
                {len(bands), 10, sum(b.rag == "RED" for b in bands)}
                if not rag
                else {sum(b.rag == rag for b in bands)}
            )
        # features / characteristics / metrics / variables
        if rag:
            opts = {sum(c.rag == rag for c in csi)}
            if noun.startswith("metric"):
                opts.add(sum(a.rag == rag for a in alerts))
            return opts
        opts = {len(csi), sum(c.rag in ("AMBER", "RED") for c in csi), min(3, len(csi))}
        if noun.startswith("metric"):
            opts |= {len(alerts), sum(a.metric != "csi" for a in alerts)}
        return opts

    def _check_invented(self, s: str, result: CheckResult) -> None:
        for m in _INVENTED.finditer(s):
            result.issues.append(
                Issue("invented_metric", m.group(0), "this metric or feature is not monitored for this model")
            )
        for m in _ACRONYM.finditer(s):
            if m.group(0) not in _KNOWN_ACRONYMS and not re.fullmatch(r"20\d\d", m.group(0)):
                result.issues.append(
                    Issue(
                        "invented_metric", m.group(0), "unknown term: not a metric in the monitoring results"
                    )
                )

    def _mentions(self, s: str) -> list[_Mention]:
        spans: list[tuple[int, int]] = []
        out: list[_Mention] = []
        for metric, segment, pat in _METRIC_PATTERNS:
            for m in re.finditer(pat, s, _FLAGS):
                if _overlaps(spans, *m.span()):
                    continue
                spans.append(m.span())
                out.append(_Mention(m.start(), m.end(), [(metric, segment)]))
        for feature, pat in _FEATURE_PATTERNS.items():
            for m in re.finditer(pat, s, _FLAGS):
                if _overlaps(spans, *m.span()):
                    continue
                spans.append(m.span())
                out.append(_Mention(m.start(), m.end(), [("csi", feature)]))
        # A generic "CSI" next to a named feature is about that feature.
        features = [m for m in out if m.keys[0][0] == "csi" and m.keys[0][1] is not None]
        generic = [m for m in out if m.keys == [("csi", None)]]
        for g in generic:
            near = [f for f in features if abs(f.start - g.end) < 40 or abs(g.start - f.end) < 40]
            if near:
                out.remove(g)
        # Calibration next to bands is about those bands.
        return out

    def _group(self, s: str, mentions: list[_Mention]) -> list[_Mention]:
        """Merge mentions joined only by 'and' / ',' / '&' (e.g. 'Gini and KS fell')."""
        groups: list[_Mention] = []
        for m in mentions:
            if groups:
                gap = s[groups[-1].end : m.start]
                if re.fullmatch(
                    r"\s*(?:,|and|&|as well as|/|, and|nor|or)?\s*(?:the |for |of )?\s*", gap, _FLAGS
                ):
                    last = groups[-1]
                    groups[-1] = _Mention(last.start, m.end, last.keys + m.keys)
                    continue
            groups.append(_Mention(m.start, m.end, list(m.keys)))
        # A band list right after calibration / binomial: attach the bands to that metric.
        merged: list[_Mention] = []
        for g in groups:
            if (
                merged
                and g.keys[0][0] == "band"
                and merged[-1].keys[0][0] in ("calibration_ratio", "binomial_pvalue")
            ):
                metric = merged[-1].keys[0][0]
                merged[-1] = _Mention(merged[-1].start, g.end, [(metric, k[1]) for k in g.keys])
                continue
            if (
                merged
                and merged[-1].keys[0][0] == "band"
                and g.keys[0][0] in ("calibration_ratio", "binomial_pvalue")
            ):
                if re.fullmatch(
                    r"\s*(?:'s|,)?\s*(?:fail\w*|on|in|of|for|the|by)?\s*(?:the )?\s*",
                    s[merged[-1].end : g.start],
                ):
                    metric = g.keys[0][0]
                    merged[-1] = _Mention(merged[-1].start, g.end, [(metric, k[1]) for k in merged[-1].keys])
                    continue
            merged.append(g)
        for g in merged:
            g.keys = [("binomial_pvalue", k[1]) if k[0] == "band" else k for k in g.keys]
        return merged

    def _items(self, s: str, excluded: list) -> list[_Item]:
        items: list[_Item] = []
        taken: list[tuple[int, int]] = list(excluded)
        for m in _THRESHOLD_CMP.finditer(s):
            items.append(
                _Item(
                    m.start(),
                    m.end(),
                    "cmp",
                    m.group(0),
                    {"op": m.group(1).lower(), "level": m.group(2).upper()},
                )
            )
            taken.append(m.span())
        for m in _THRESHOLD_REF.finditer(s):
            taken.append(m.span(1))
        for m in _BASELINE_CMP.finditer(s):
            items.append(_Item(m.start(), m.end(), "baseline_cmp", m.group(0), {"op": m.group(1).lower()}))
        for m in _RAG_SET.finditer(s):
            if _overlaps(taken, *m.span()):
                continue
            items.append(
                _Item(
                    m.start(), m.end(), "rag", m.group(0), {"rags": {m.group(1).upper(), m.group(2).upper()}}
                )
            )
            taken.append(m.span())
        for m in _RAG_WORD.finditer(s):
            if _overlaps(taken, *m.span()):
                continue
            items.append(_Item(m.start(), m.end(), "rag", m.group(0), {"rags": {m.group(1).upper()}}))
        for m in _WITHIN.finditer(s):
            items.append(_Item(m.start(), m.end(), "rag", m.group(0), {"rags": {"GREEN"}}))
        for m in _BREACH.finditer(s):
            if _overlaps(taken, *m.span()) or _NEGATION.search(s[max(0, m.start() - 20) : m.start()]):
                continue
            items.append(_Item(m.start(), m.end(), "rag", m.group(0), {"rags": {"AMBER", "RED"}}))
        for m in _DIRECTION.finditer(s):
            word = m.group(1).lower()
            cls = next(c for c, pat in _DIR_CLASS if re.fullmatch(pat, word))
            if _NEGATION.search(s[max(0, m.start() - 12) : m.start()]):
                continue
            items.append(_Item(m.start(), m.end(), "direction", m.group(0), {"cls": cls}))
        for m in _UNAVAILABLE.finditer(s):
            items.append(_Item(m.start(), m.end(), "unavailable", m.group(0), {}))
        for m in _NUMBER.finditer(s):
            if _overlaps(taken, *m.span()):
                continue
            if _CONSTANT_UNIT.match(s[m.end() :]):
                continue
            sign, whole, frac, exp, pct = m.group(1), m.group(2), m.group(3) or "", m.group(4), m.group(5)
            power = int(exp.replace("−", "-")) if exp else 0
            value = float(whole.replace(",", "") + frac) * 10.0**power
            cmp = _NUM_CMP.search(s[max(0, m.start() - 40) : m.start()])
            items.append(
                _Item(
                    m.start(),
                    m.end(),
                    "number",
                    m.group(0).strip(),
                    {
                        "value": -value if sign in ("-", "−") else value,
                        "tol": 0.5 * 10.0 ** (power - max(len(frac) - 1, 0)) + 1e-12,
                        "percent": bool(pct),
                        "approx": bool(_APPROX.search(s[max(0, m.start() - 20) : m.start()])),
                        "cmp": (cmp.group(1) or cmp.group(2)) if cmp else None,
                    },
                )
            )
        return sorted(items, key=lambda i: i.start)

    @staticmethod
    def _attach(item: _Item, groups: list[_Mention], clauses: list[tuple[int, int]]) -> _Mention | None:
        clause = next((c for c in clauses if c[0] <= item.start <= c[1]), clauses[-1])
        before = [g for g in groups if g.end <= item.start and g.start >= clause[0]]
        if before:
            return before[-1]
        after = [g for g in groups if g.start >= item.end and g.end <= clause[1]]
        if after:
            return after[0]
        before = [g for g in groups if g.end <= item.start]
        if before:
            return before[-1]
        after = [g for g in groups if g.start >= item.end]
        return after[0] if after else None

    # ---- verification
    def _facts_for(self, target: _Mention | None) -> list[MetricFact]:
        if target is None:
            return []
        out: list[MetricFact] = []
        for metric, segment in target.keys:
            if metric == "csi" and segment is None:
                out.extend(self.facts.csi())
                continue
            if segment is None:
                out.extend(f for (m, _), f in self.facts.metrics.items() if m == metric)
                continue
            f = self.facts.get(metric, segment)
            if f is not None:
                out.append(f)
        return out

    def _verify(self, item: _Item, target: _Mention | None, s: str, clauses, result: CheckResult) -> None:
        result.n_claims += 1
        facts = self._facts_for(target)
        clause = next((s[a:b] for a, b in clauses if a <= item.start <= b), s)
        if item.kind == "number":
            if target is not None and all(k[0] == "overall_rag" for k in target.keys):
                facts = []  # the overall status is a label, not a number: treat as unattributed
            self._verify_number(item, facts, clause, s, result)
        elif item.kind == "rag":
            self._verify_rag(item, facts, target, s, result)
        elif item.kind == "cmp":
            self._verify_threshold_cmp(item, facts, result)
        elif item.kind == "baseline_cmp":
            for f in facts:
                if f.baseline is not None and f.available:
                    above = f.value > f.baseline
                    if (item.data["op"] in ("above", "over")) != above:
                        result.issues.append(
                            Issue(
                                "wrong_direction",
                                item.text,
                                f"{_name(f)} is {_fmt(f.value)} vs baseline {_fmt(f.baseline)}",
                            )
                        )
        elif item.kind == "direction":
            self._verify_direction(item, facts, clause, result)
        elif item.kind == "unavailable":
            for f in facts:
                if f.available and f.metric not in ("binomial_pvalue",):
                    result.issues.append(
                        Issue(
                            "contradiction",
                            item.text,
                            f"{_name(f)} is available this month ({_fmt(f.value)})",
                        )
                    )

    def _verify_number(
        self, item: _Item, facts: list[MetricFact], clause: str, s: str, result: CheckResult
    ) -> None:
        d = item.data
        if d["cmp"]:
            cmp_word = d["cmp"].lower()
            above = cmp_word.startswith(("above", "over", "exceed", "beyond", "more", "greater", "higher", ">", "≥"))
            claimed = (
                d["value"] / 100
                if d["percent"] and all(f.metric not in ("volume",) for f in facts)
                else d["value"]
            )
            avail = [f for f in facts if f.available]
            for f in avail:
                truth = f.value
                if (above and not truth > claimed) or (not above and not truth < claimed):
                    result.issues.append(
                        Issue("wrong_number", f"{cmp_word} {item.text}", f"{_name(f)} is {_fmt(truth)}")
                    )
            if avail or not facts:
                return
        cands = self._candidates(facts, clause, d)
        if not facts:
            cands = self._all_values()
        if any(_matches(d, x) for x in cands):
            return
        if facts and not any(f.available for f in facts):
            result.issues.append(
                Issue(
                    "unavailable_metric",
                    item.text,
                    f"{_name(facts[0])} is not available in {self.facts.month}",
                )
            )
            return
        if not facts:
            result.issues.append(
                Issue("unsupported_number", item.text, "this number does not match any monitoring result")
            )
            return
        truth = ", ".join(f"{_name(f)} {_fmt(f.value)}" for f in facts[:3])
        result.issues.append(Issue("wrong_number", item.text, f"the results show {truth}"))

    def _candidates(self, facts: list[MetricFact], clause: str, d: dict) -> list[float]:
        cands: list[float] = []
        prev_ctx = _PREV_CONTEXT.search(clause) or (self.facts.prev_month and self.facts.prev_month in clause)
        thr_ctx = _THRESHOLD_CONTEXT.search(clause)
        base_ctx = _BASELINE_CONTEXT.search(clause)
        for f in facts:
            if f.value is not None and not math.isnan(f.value):
                cands.append(f.value)
                if f.metric == "calibration_ratio":
                    cands.append(f.value - 1.0)  # "under-predicts by 32%"
            if prev_ctx and f.prev is not None:
                cands.append(f.prev)
            if (
                re.search(r"\bby\b|\bchange\b|\bincrease of\b|\bdecrease of\b", clause, _FLAGS)
                and f.delta is not None
            ):
                cands.append(f.delta)
                if f.prev:
                    cands.append(f.delta / f.prev)
            if thr_ctx:
                if f.metric == "calibration_ratio":
                    cands += [0.15, 0.30, 0.85, 1.15, 0.70, 1.30, 1.0]
                else:
                    cands += [x for x in (f.amber, f.red) if x is not None]
            if base_ctx and f.baseline is not None:
                cands.append(f.baseline)
                if f.available:
                    cands.append((f.baseline - f.value) / f.baseline)
        return cands

    def _all_values(self) -> list[float]:
        vals: list[float] = []
        for f in self.facts.metrics.values():
            vals += [x for x in (f.value, f.amber, f.red, f.baseline) if x is not None and not math.isnan(x)]
        return vals

    def _verify_rag(self, item: _Item, facts: list[MetricFact], target, s: str, result: CheckResult) -> None:
        rags = item.data["rags"]
        before = s[max(0, item.start - 25) : item.start].lower()
        after = s[item.end : item.end + 30].lower()
        use_prev = bool(
            re.search(r"\bfrom\s*$|\bpreviously\s*$|\bwas\s*$|\blast month\b.*$", before)
            or re.match(r"\s*(?:in|the|last)?\s*(?:the )?(?:previous|prior|last) month", after)
        )
        if not facts:
            overall = self.facts.get("overall_rag")
            facts = [overall] if overall else []
        if target is not None and any(k[0] == "csi" and k[1] is None for k in target.keys) and not use_prev:
            if not any(f.rag in rags for f in facts):
                result.issues.append(Issue("wrong_status", item.text, "no CSI feature has this status"))
            return
        for f in facts:
            truth = f.prev_rag if use_prev else f.rag
            if truth is None:
                result.issues.append(
                    Issue("wrong_status", item.text, f"there is no previous month for {_name(f)}")
                )
            elif truth not in rags:
                when = "last month" if use_prev else self.facts.month
                if truth == "NA":
                    detail = f"{_name(f)} has no RAG status in {when}"
                else:
                    detail = f"{_name(f)} is {truth} in {when}"
                result.issues.append(Issue("wrong_status", item.text, detail))

    def _verify_threshold_cmp(self, item: _Item, facts: list[MetricFact], result: CheckResult) -> None:
        op, level = item.data["op"], item.data["level"]
        if op == "at":
            return
        for f in facts:
            if f.rag not in ("GREEN", "AMBER", "RED"):
                result.issues.append(
                    Issue("wrong_status", item.text, f"{_name(f)} has no RAG status this month")
                )
                continue
            crossing = op.startswith(("exceed", "cross", "breach", "past", "through", "beyond"))
            upward = op in ("above", "over")
            worse_side = crossing or (upward if f.direction == "higher_worse" else op in ("below", "under"))
            if level == "RED":
                ok = {"RED"} if worse_side else {"GREEN", "AMBER"}
            else:
                ok = {"AMBER", "RED"} if worse_side else {"GREEN"}
            if f.rag not in ok:
                result.issues.append(
                    Issue("wrong_status", item.text, f"{_name(f)} is {f.rag} ({_fmt(f.value)})")
                )

    def _verify_direction(
        self, item: _Item, facts: list[MetricFact], clause: str, result: CheckResult
    ) -> None:
        cls = item.data["cls"]
        vs_baseline = bool(_BASELINE_CONTEXT.search(clause))
        for f in facts:
            if not f.available:
                result.issues.append(
                    Issue("unavailable_metric", item.text, f"{_name(f)} is not available this month")
                )
                continue
            ref = f.baseline if vs_baseline and f.baseline is not None else f.prev
            if ref is None:
                if vs_baseline:
                    continue  # PSI/CSI are measured against development by definition
                result.issues.append(
                    Issue("wrong_direction", item.text, f"no previous month to compare {_name(f)} with")
                )
                continue
            delta = f.value - ref
            if cls == "up":
                ok = delta > 0
            elif cls == "down":
                ok = delta < 0
            elif cls == "flat":
                ok = abs(delta) <= max(0.005, 0.02 * abs(ref))
            else:
                now, before = f.badness(f.value), f.badness(ref)
                if now is None or before is None:
                    continue
                ok = now > before if cls == "worse" else now < before
            if not ok:
                result.issues.append(
                    Issue(
                        "wrong_direction",
                        item.text,
                        f"{_name(f)} moved from {_fmt(ref)} to {_fmt(f.value)}"
                        + (" (development baseline)" if ref is f.baseline else f" ({self.facts.prev_month})"),
                    )
                )

    def _check_no_alert_claims(self, s: str, result: CheckResult) -> None:
        for m in _NO_ALERTS.finditer(s):
            alerts = self.facts.alerts()
            level = (m.group(1) or "").upper()
            relevant = [a for a in alerts if not level or a.rag == level]
            result.n_claims += 1
            if relevant:
                result.issues.append(
                    Issue(
                        "contradiction",
                        m.group(0),
                        f"{len(relevant)} headline alerts fired in {self.facts.month}",
                    )
                )

    def _driver_paragraph_span(self, text: str) -> tuple[int, int] | None:
        m = re.search(r"likely drivers?\s*:", text, _FLAGS)
        if not m:
            return None
        end = re.search(r"\n\s*\n|recommended actions?\s*:", text[m.end() :], _FLAGS)
        return (m.start(), m.end() + (end.start() if end else len(text)))

    def _check_drivers(self, s: str, mentions: list[_Mention], result: CheckResult, in_para: bool) -> None:
        if not (in_para or _DRIVER_CUE.search(s)):
            return
        allowed = self.facts.driver_features()
        for m in mentions:
            for metric, feature in m.keys:
                if metric != "csi" or feature is None or feature in allowed:
                    continue
                if _NEGATION.search(s[max(0, m.start - 30) : m.start]) or _NEGATION.search(
                    s[m.end : m.end + 25]
                ):
                    continue
                f = self.facts.get("csi", feature)
                rag = f.rag if f else "unknown"
                result.issues.append(
                    Issue(
                        "unsupported_driver",
                        s[m.start : m.end],
                        f"{FEATURE_LABELS.get(feature, feature)} CSI is {rag} and not among the top drifting features",
                    )
                )

    def _check_omissions(self, text: str, mentioned: set, result: CheckResult) -> None:
        f = self.facts
        if f.overall_rag in ("GREEN", "AMBER", "RED") and not re.search(
            rf"\b{f.overall_rag}\b", text, _FLAGS
        ):
            result.issues.append(
                Issue("omission", "overall status", f"the overall status ({f.overall_rag}) is not stated")
            )
        for a in f.alerts():
            if a.rag != "RED" or a.metric == "csi":
                continue
            if (a.metric, "ALL") not in mentioned:
                result.issues.append(
                    Issue("omission", _name(a), f"the RED {_name(a)} alert is not mentioned")
                )
        red_csi = [c for c in f.csi() if c.rag == "RED"]
        if red_csi and ("csi", red_csi[0].segment) not in mentioned:
            result.issues.append(
                Issue(
                    "omission",
                    _name(red_csi[0]),
                    f"the most drifted feature ({_name(red_csi[0])}, RED) is not mentioned",
                )
            )


def _name(f: MetricFact) -> str:
    if f.metric == "csi":
        return f"CSI {FEATURE_LABELS.get(f.segment, f.segment)}"
    label = f.metric.replace("_", " ")
    return label if f.segment == "ALL" else f"{label} ({f.segment})"


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.4g}"


def _matches(d: dict, truth: float) -> bool:
    claimed = abs(d["value"])
    tol = d["tol"]
    targets = [abs(truth) * 100] if d["percent"] else [abs(truth)]
    if d["percent"] and abs(truth) > 1.5:
        targets.append(abs(truth))  # already on a percent-like scale
    for t in targets:
        allowed = max(tol, 0.02 * t) if d["approx"] else tol
        if abs(claimed - t) <= allowed:
            return True
    return False


def check(text: str, facts: MonthFacts) -> CheckResult:
    return Checker(facts).check(text)
