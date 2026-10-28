"""One month's monitoring results as a typed fact sheet: the single source of truth for the commentary.

The LLM prompt is rendered from these facts and the checker verifies the draft against the same
facts, so the model never sees a number the checker cannot check. Facts come only from
``data/monitoring/results/month=<M>.parquet`` (this month and the previous one); nothing is re-computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from common import paths
from common.config import CSI_FEATURES, monitor_months
from common.schemas import validate_results
from monitoring.config import load_thresholds

#: Headline ALL-segment metrics, in the order they are presented (metric, human label).
SCALAR_METRICS: tuple[tuple[str, str], ...] = (
    ("psi_score", "Score PSI"),
    ("gini", "Gini"),
    ("ks", "KS"),
    ("auc", "AUC"),
    ("calibration_ratio", "Calibration ratio (observed / predicted default rate)"),
    ("hl_pvalue", "Hosmer-Lemeshow p-value"),
    ("mean_pd", "Mean predicted PD"),
    ("default_rate", "Observed default rate"),
    ("volume", "Volume (loans)"),
)
#: Metrics whose RAG drives ``overall_rag`` (CSI features are headline too).
HEADLINE_METRICS = ("psi_score", "gini", "ks", "calibration_ratio", "hl_pvalue")
#: Plain-English names for the CSI features, used in the prompt and by the checker.
FEATURE_LABELS: dict[str, str] = {
    "loan_amnt": "loan amount",
    "int_rate": "interest rate",
    "annual_inc": "annual income",
    "dti": "debt-to-income (DTI)",
    "fico": "FICO",
    "revol_util": "revolving utilisation",
    "credit_history_years": "credit history length",
    "emp_length_years": "employment length",
    "grade": "grade",
    "home_ownership": "home ownership",
    "verification_status": "income verification status",
    "purpose": "loan purpose",
}
RAG_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2}


def _num(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


@dataclass(frozen=True)
class MetricFact:
    """One (metric, segment) row for the month, with last month's value for direction claims."""

    metric: str
    segment: str
    value: float | None
    rag: str
    direction: str  # higher_worse | lower_worse
    amber: float | None = None
    red: float | None = None
    prev: float | None = None
    prev_rag: str | None = None
    baseline: float | None = None  # dev baseline (gini / ks only)

    @property
    def available(self) -> bool:
        return self.value is not None and not math.isnan(self.value)

    @property
    def delta(self) -> float | None:
        if self.value is None or self.prev is None:
            return None
        return self.value - self.prev

    def badness(self, value: float | None) -> float | None:
        """A number that grows as the metric gets worse (for improved / deteriorated claims)."""
        if value is None:
            return None
        if self.metric == "calibration_ratio":
            return abs(value - 1.0)
        if self.metric == "auc":
            return -value  # no threshold row, but higher AUC is better
        if self.metric in ("volume", "mean_pd", "default_rate"):
            return None  # context metrics: "better" / "worse" is not defined
        return -value if self.direction == "lower_worse" else value


@dataclass
class MonthFacts:
    month: str
    prev_month: str | None
    model_version: str
    overall_rag: str
    prev_overall_rag: str | None
    outcomes_observable: bool
    metrics: dict[tuple[str, str], MetricFact] = field(default_factory=dict)

    def get(self, metric: str, segment: str = "ALL") -> MetricFact | None:
        return self.metrics.get((metric, segment))

    def csi(self) -> list[MetricFact]:
        """CSI facts sorted from most to least drifted."""
        rows = [f for (m, _), f in self.metrics.items() if m == "csi" and f.available]
        return sorted(rows, key=lambda f: -(f.value or 0.0))

    def alerts(self) -> list[MetricFact]:
        """Headline AMBER/RED rows (model-level metrics and CSI features), worst first."""
        rows = [
            f
            for f in self.metrics.values()
            if f.rag in ("AMBER", "RED")
            and (f.metric == "csi" or (f.segment == "ALL" and f.metric in HEADLINE_METRICS))
        ]
        return sorted(rows, key=lambda f: (-RAG_RANK[f.rag], f.metric != "psi_score", f.metric, f.segment))

    def band_breaches(self) -> list[MetricFact]:
        """Score bands whose binomial calibration test is AMBER/RED (drill-down, not headline)."""
        rows = [
            f
            for (m, s), f in self.metrics.items()
            if m == "binomial_pvalue" and s.startswith("band=") and f.rag in ("AMBER", "RED")
        ]
        return sorted(rows, key=lambda f: int(f.segment.removeprefix("band=")))

    def driver_features(self, top_n: int = 3) -> set[str]:
        """Features a draft may name as a likely driver: CSI AMBER/RED or among the top-n by CSI."""
        drifted = {f.segment for f in self.csi() if f.rag in ("AMBER", "RED")}
        return drifted | {f.segment for f in self.csi()[:top_n]}


def previous_month(month: str) -> str | None:
    months = monitor_months()
    if month in months:
        i = months.index(month)
        return months[i - 1] if i > 0 else None
    return (pd.Period(month, freq="M") - 1).strftime("%Y-%m")


def _baselines() -> dict[str, float]:
    """Dev Gini/KS baseline implied by the relative-drop thresholds (amber = baseline * (1 - rel_amber))."""
    return {m: (load_thresholds().for_metric(m) or {}).get("rel_amber", float("nan")) for m in ("gini", "ks")}


def build_facts(current: pd.DataFrame, previous: pd.DataFrame | None = None) -> MonthFacts:
    """Build the fact sheet from one month's long results table (and optionally the month before)."""
    month = str(current["month"].iloc[0])
    prev_by_key: dict[tuple[str, str], pd.Series] = {}
    if previous is not None and not previous.empty:
        prev_by_key = {(r.metric, r.segment): r for r in previous.itertuples(index=False)}
    rel_amber = _baselines()
    metrics: dict[tuple[str, str], MetricFact] = {}
    for r in current.itertuples(index=False):
        key = (str(r.metric), str(r.segment))
        prev = prev_by_key.get(key)
        amber, red = _num(r.amber_threshold), _num(r.red_threshold)
        baseline = None
        if r.metric in ("gini", "ks") and amber is not None and not math.isnan(rel_amber[r.metric]):
            baseline = amber / (1.0 - rel_amber[r.metric])
        metrics[key] = MetricFact(
            metric=key[0],
            segment=key[1],
            value=_num(r.value),
            rag=str(r.rag),
            direction=str(r.direction),
            amber=amber,
            red=red,
            prev=_num(prev.value) if prev is not None else None,
            prev_rag=str(prev.rag) if prev is not None else None,
            baseline=baseline,
        )
    overall = metrics[("overall_rag", "ALL")]
    gini = metrics.get(("gini", "ALL"))
    return MonthFacts(
        month=month,
        prev_month=str(previous["month"].iloc[0]) if previous is not None and not previous.empty else None,
        model_version=str(current["model_version"].iloc[0]),
        overall_rag=overall.rag,
        prev_overall_rag=overall.prev_rag,
        outcomes_observable=bool(gini is not None and gini.available),
        metrics=metrics,
    )


def load_month(month: str) -> pd.DataFrame | None:
    path = paths.monitoring_path(month)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    validate_results(frame)
    return frame


def load_facts(month: str) -> MonthFacts:
    current = load_month(month)
    if current is None:
        raise FileNotFoundError(f"no monitoring results for {month}: run `python -m monitoring.run` first")
    prev = previous_month(month)
    return build_facts(current, load_month(prev) if prev else None)


def fmt(value: float | None, metric: str) -> str:
    """The precision a fact is shown at in the prompt and the template."""
    if value is None or math.isnan(value):
        return "n/a"
    if metric == "volume":
        return f"{value:,.0f}"
    if metric in ("hl_pvalue", "binomial_pvalue"):
        return f"{value:.3g}" if value >= 0.001 else "< 0.001"
    return f"{value:.3f}"


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


__all__ = [
    "CSI_FEATURES",
    "FEATURE_LABELS",
    "HEADLINE_METRICS",
    "MetricFact",
    "MonthFacts",
    "SCALAR_METRICS",
    "build_facts",
    "feature_label",
    "fmt",
    "load_facts",
    "previous_month",
]
