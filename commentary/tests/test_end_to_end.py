"""End to end on the committed loan fixture: train, score, monitor, draft (scripted), check, report."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from commentary import run as commentary_run
from commentary.facts import load_facts
from commentary.llm import ScriptedDrafter
from commentary.template import template_commentary
from common import paths
from model import score, train
from monitoring.run import run as monitor
from reporting.data import load_results
from reporting.report import write_report

MONTHS = ("2015-05", "2015-06")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("crm")
    mp = pytest.MonkeyPatch()
    mp.setenv("CRM_DATA_DIR", str(root))
    paths.ensure_parent(paths.prepared_loans())
    pd.read_parquet(paths.SAMPLE_LOANS).to_parquet(paths.prepared_loans())
    mp.setattr("sys.argv", ["x"])
    train.main()
    for month in MONTHS:
        mp.setattr("sys.argv", ["x", "--month", month])
        score.main()
        monitor(month)
    yield root
    mp.undo()


def _corrupt(text: str) -> str:
    """Change the first metric value in the draft (a 'wrong number' a model might produce)."""
    m = re.search(r"\b0\.(\d{3})\b", text)
    wrong = f"0.{(int(m.group(1)) + 137) % 1000:03d}"
    return text[: m.start()] + wrong + text[m.end() :]


def test_retry_then_publish_and_render(data_root):
    facts = load_facts(MONTHS[1])
    assert facts.prev_month == MONTHS[0]
    good = template_commentary(facts)
    drafter = ScriptedDrafter([_corrupt(good), good])
    c = commentary_run.draft_commentary(facts, drafter)
    assert c.path == "llm_after_retry"
    assert [a.passed for a in c.attempts] == [False, True]
    assert c.attempts[0].issues[0]["kind"] == "wrong_number"
    # The second prompt carried the checker's feedback.
    assert "The checker rejected your draft" in drafter.calls[1][-1]["content"]
    assert len(drafter.calls[1]) == 2  # system + user: the rejected draft is not replayed
    commentary_run.write_commentary(c)

    html = write_report(load_results(up_to=MONTHS[1]), MONTHS[1]).read_text(encoding="utf-8")
    assert "AI-drafted, verified" in html
    assert "an earlier LLM draft failed the checker" in html
    assert "wrong_number" in html  # rejected draft's finding is disclosed


def test_fallback_when_every_draft_fails(data_root):
    facts = load_facts(MONTHS[1])
    drafter = ScriptedDrafter(["What changed: The Brier score improved to 0.9 in 2019-01."])
    c = commentary_run.draft_commentary(facts, drafter, max_retries=2)
    assert c.path == "template_fallback"
    assert len(c.attempts) == 3 and not any(a.passed for a in c.attempts)
    assert c.text == template_commentary(facts) and c.template_issues == []
    commentary_run.write_commentary(c)
    html = write_report(load_results(up_to=MONTHS[1]), MONTHS[1]).read_text(encoding="utf-8")
    assert "Template fallback" in html and "Template summary, verified" in html


def test_cli_template_mode(data_root, capsys):
    commentary_run.main(["--month", MONTHS[0], "--drafter", "template"])
    saved = commentary_run.load_commentary(MONTHS[0])
    assert saved["path"] == "template_only" and saved["n_claims"] > 10
