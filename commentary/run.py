"""Draft, verify and publish one month's commentary. ``python -m commentary.run --month YYYY-MM``

Flow: fact sheet -> LLM draft -> checker. A failing draft is retried with the checker's feedback up
to ``max_retries`` times; if none passes, the deterministic template is published instead. The
outcome (published text, path taken, every attempt with its issues, model id and settings, timings)
is written to ``common.paths.commentary_path(month)`` and rendered by ``reporting.report``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from commentary.checker import check
from commentary.facts import MonthFacts, load_facts
from commentary.llm import DEFAULT_MODEL_ID, Drafter, HFDrafter, HFSettings
from commentary.prompt import build_messages
from commentary.template import template_commentary
from common import paths
from common.config import monitor_months

MAX_RETRIES = 2
#: How the published text was produced.
PATHS = ("llm_first_pass", "llm_after_retry", "template_fallback", "template_only")


@dataclass
class Attempt:
    draft: str
    passed: bool
    issues: list[dict]
    n_claims: int
    seconds: float


@dataclass
class Commentary:
    month: str
    text: str
    path: str
    n_claims: int
    attempts: list[Attempt] = field(default_factory=list)
    model: dict = field(default_factory=dict)
    template_issues: list[dict] = field(default_factory=list)
    generation_seconds: float = 0.0
    generated_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _issues(result) -> list[dict]:
    return [asdict(i) for i in result.issues]


def draft_commentary(
    facts: MonthFacts, drafter: Drafter | None, max_retries: int = MAX_RETRIES
) -> Commentary:
    """Run the draft -> check -> retry -> fallback loop for one month."""
    start = time.perf_counter()
    attempts: list[Attempt] = []
    history: list[tuple[str, list[str]]] = []
    published: tuple[str, str, int] | None = None
    if drafter is not None:
        for i in range(max_retries + 1):
            t0 = time.perf_counter()
            draft = drafter.generate(build_messages(facts, history), attempt=i)
            gen = time.perf_counter() - t0
            result = check(draft, facts)
            attempts.append(Attempt(draft, result.passed, _issues(result), result.n_claims, round(gen, 2)))
            if result.passed:
                published = (draft, "llm_first_pass" if i == 0 else "llm_after_retry", result.n_claims)
                break
            history.append((draft, result.feedback()))
    template_issues: list[dict] = []
    if published is None:
        text = template_commentary(facts)
        result = check(text, facts)
        template_issues = _issues(result)  # expected empty; a non-empty list is a template bug, kept visible
        published = (text, "template_only" if drafter is None else "template_fallback", result.n_claims)
    text, path, n_claims = published
    return Commentary(
        month=facts.month,
        text=text,
        path=path,
        n_claims=n_claims,
        attempts=attempts,
        model={"name": drafter.name, **drafter.settings()} if drafter is not None else {"name": "template"},
        template_issues=template_issues,
        generation_seconds=round(time.perf_counter() - start, 2),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def write_commentary(c: Commentary) -> str:
    out = paths.ensure_parent(paths.commentary_path(c.month))
    out.write_text(c.to_json(), encoding="utf-8")
    return str(out)


def load_commentary(month: str) -> dict | None:
    path = paths.commentary_path(month)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def make_drafter(kind: str, model_id: str, quantization: str) -> Drafter | None:
    if kind == "template":
        return None
    return HFDrafter(HFSettings(model_id=model_id, quantization=quantization))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="commentary.run", description="Draft + verify monthly commentary")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--month", help="vintage YYYY-MM")
    group.add_argument("--all", action="store_true", help="every monitored month with results on disk")
    parser.add_argument("--drafter", choices=("hf", "template"), default="hf")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--quantization", choices=("nf4", "fp16", "fp32"), default="nf4")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args(argv)
    months = [m for m in monitor_months() if paths.monitoring_path(m).exists()] if args.all else [args.month]
    drafter = make_drafter(args.drafter, args.model_id, args.quantization)
    for month in months:
        c = draft_commentary(load_facts(month), drafter, args.max_retries)
        print(
            f"{month}: {c.path} after {len(c.attempts)} LLM attempt(s), "
            f"{c.generation_seconds}s -> {write_commentary(c)}"
        )


if __name__ == "__main__":
    main()
