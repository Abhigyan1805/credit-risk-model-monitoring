"""The evaluation's building blocks: clean paraphrases pass, every corruption operator is caught."""

from __future__ import annotations

import random

import pytest

from commentary.checker import check
from commentary.evaluate import CORRUPTIONS, alt_phrasing
from commentary.template import template_commentary
from commentary.tests import _facts
from commentary.tests._facts import GOOD_MARCH


@pytest.mark.parametrize("make", [_facts.march, _facts.first_month, _facts.immature])
def test_alternative_phrasing_is_clean(make):
    facts = make()
    result = check(alt_phrasing(facts), facts)
    assert result.passed, result.feedback()


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
@pytest.mark.parametrize("make", [_facts.march, _facts.immature])
def test_corruptions_are_caught(name, make):
    facts = make()
    rng = random.Random(7)
    for text in (template_commentary(facts), GOOD_MARCH if make is _facts.march else None):
        if text is None:
            continue
        bad = CORRUPTIONS[name](text, rng, facts)
        if bad is None:
            continue  # operator not applicable to this month (e.g. no RED alert to drop)
        assert bad != text
        assert not check(bad, facts).passed, (name, bad)
