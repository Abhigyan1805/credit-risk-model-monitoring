"""Checker unit tests: a hand-written correct draft passes; each deliberate corruption is caught."""

from __future__ import annotations

import pytest

from commentary.checker import check
from commentary.template import template_commentary
from commentary.tests import _facts
from commentary.tests._facts import GOOD_MARCH


@pytest.fixture(scope="module")
def march():
    return _facts.march()


def kinds(text, facts):
    return check(text, facts).kinds()


def test_correct_draft_passes(march):
    result = check(GOOD_MARCH, march)
    assert result.passed, result.feedback()
    assert result.n_claims >= 20


@pytest.mark.parametrize("make", [_facts.march, _facts.first_month, _facts.immature])
def test_template_always_passes(make):
    facts = make()
    result = check(template_commentary(facts), facts)
    assert result.passed, result.feedback()


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        ("rose to 0.183", "rose to 0.218", "wrong_number"),
        ("Gini fell to 0.312", "Gini rose to 0.312", "wrong_direction"),
        ("is now RED, up", "is now AMBER, up", "wrong_status"),
        ("in 2015-02", "in 2015-01", "wrong_month"),
        ("FICO is stable.", "FICO is stable. The Brier score is 0.11.", "invented_metric"),
        ("is the likely driver", "and home ownership are the likely driver", "unsupported_driver"),
        ("The binomial test flags bands 8 and 9.", "There are no alerts this month.", "contradiction"),
        ("(default rate 16.3%)", "(default rate 18.3%)", "wrong_number"),
        ("stays AMBER", "stays GREEN", "wrong_status"),
        ("KS is 0.236 (GREEN)", "KS is 0.236 (AMBER)", "wrong_status"),
        ("Gini fell to 0.312 but remains GREEN", "Gini fell below the amber threshold", "wrong_status"),
        ("the interest rate CSI of 0.311 is RED", "the FICO CSI of 0.311 is RED", "wrong_number"),
    ],
)
def test_single_corruption_is_caught(march, old, new, kind):
    assert old in GOOD_MARCH
    found = kinds(GOOD_MARCH.replace(old, new), march)
    assert kind in found, found


def test_rounding_tolerance(march):
    assert check("Overall status is RED. Score PSI is 0.18 (AMBER).", march).kinds() == ["omission"] * 7
    issues = check("Overall status is RED. Score PSI is 0.19 (AMBER).", march).kinds()
    assert "wrong_number" in issues
    assert "wrong_number" not in kinds("The observed default rate is 16%.", march)
    assert "wrong_number" in kinds("The observed default rate is 17%.", march)


def test_omission_of_red_alerts(march):
    found = check("Overall status is RED. Score PSI is 0.183, AMBER.", march)
    details = " ".join(i.detail for i in found.issues if i.kind == "omission")
    assert "calibration ratio" in details and "hl pvalue" in details and "interest rate" in details


def test_threshold_and_numeric_comparisons(march):
    assert "wrong_status" in kinds("Score PSI is above the red threshold.", march)
    assert "wrong_status" not in kinds("Score PSI is above the amber threshold.", march)
    assert "wrong_number" in kinds("Score PSI is above 0.25.", march)
    assert "wrong_number" not in kinds("Score PSI is above 0.10.", march)
    assert "wrong_direction" in kinds("Gini is above its development baseline.", march)


def test_counts(march):
    assert "wrong_number" not in kinds("One feature is RED.", march)
    assert "wrong_number" in kinds("Three features are RED.", march)
    assert "wrong_number" not in kinds("There are 5 headline alerts.", march)


def test_first_month_has_no_direction_claims():
    facts = _facts.first_month()
    found = check("In 2014-01 the overall status is RED and score PSI rose to 0.183.", facts)
    assert "wrong_direction" in found.kinds()


def test_unavailable_outcome_metrics():
    facts = _facts.immature()
    assert "unavailable_metric" in kinds("Gini is 0.312 this month.", facts)
    ok = check(
        "In 2016-04 the overall status is RED. Gini is not available yet. Score PSI rose to 0.281.", facts
    )
    assert "unavailable_metric" not in ok.kinds() and "contradiction" not in ok.kinds()
    assert "contradiction" in kinds("Gini is not available yet.", _facts.march())


def test_unattributed_number_and_unknown_acronym(march):
    assert "unsupported_number" in kinds("Overall status is RED; 0.777 of loans were affected.", march)
    assert "invented_metric" in kinds("The LGD rose this month.", march)


def test_small_p_values(march):
    assert "wrong_number" not in kinds("The Hosmer-Lemeshow p-value is < 0.001 (RED).", march)
    assert "wrong_number" not in kinds("The Hosmer-Lemeshow p-value is 1.2e-4 (RED).", march)
    assert "wrong_number" in kinds("The Hosmer-Lemeshow p-value is 3.2e-4 (RED).", march)
    assert "wrong_number" in kinds("The Hosmer-Lemeshow p-value is > 0.05 (RED).", march)


def test_band_lists_each_keep_their_own_status(march):
    ok = "The binomial test flags score band 8 RED, band 9 AMBER."
    assert "wrong_status" not in kinds(ok, march)
    assert "wrong_status" in kinds("The binomial test flags score band 8 RED, band 9 RED.", march)


def test_missing_paragraph_is_an_omission(march):
    found = check(GOOD_MARCH.split("\n\nRecommended action:")[0], march)
    assert [i.claim for i in found.issues] == ["Recommended action"]


def test_from_to_order(march):
    assert "wrong_number" not in kinds("Score PSI rose from 0.151 to 0.183.", march)
    assert "wrong_number" in kinds("Score PSI rose from 0.183 to 0.151.", march)
    assert "wrong_status" in kinds("The calibration ratio is in the green zone.", march)


def test_other_features_green(march):
    assert "wrong_status" not in kinds("All other features are GREEN.", march)


def test_audited_false_alarms_stay_fixed(march):
    # Found by auditing development-run drafts: negation, plain "accuracy", "recalibration ratio".
    assert "wrong_direction" not in kinds(
        "The overall status is RED, indicating no significant improvement.", march
    )
    assert "invented_metric" not in kinds("Interest rate drift may be affecting its accuracy.", march)
    assert "unsupported_number" not in kinds(
        "The recalibration ratio rose to 1.381 from 1.290, which is RED.", march
    )
    found = check("Interest rate and FICO have CSI values above 0.10.", march)
    assert found.kinds().count("wrong_number") == 1
