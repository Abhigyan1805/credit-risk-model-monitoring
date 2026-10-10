"""Hand-computed tests for discrimination metrics."""

import numpy as np
import pytest

from monitoring.discrimination import auc, gini, ks

# Worked example: labels [0,0,1,1], scores [0.1,0.4,0.35,0.8].
# Sorted scores give default ranks 2 and 4 -> AUC = (2+4 - 3)/(2*2) = 0.75.
HAND_LABELS = [0, 0, 1, 1]
HAND_SCORES = [0.1, 0.4, 0.35, 0.8]


def test_auc_hand_computed():
    assert auc(HAND_LABELS, HAND_SCORES) == pytest.approx(0.75)


def test_gini_hand_computed():
    assert gini(HAND_LABELS, HAND_SCORES) == pytest.approx(0.5)


def test_ks_hand_computed():
    # sorted y = [0,1,0,1]; max |cum_pos - cum_neg| = 0.5
    assert ks(HAND_LABELS, HAND_SCORES) == pytest.approx(0.5)


def test_auc_perfect_separation():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_auc_reverse_separation_is_zero():
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auc_ties_count_as_half():
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_ks_perfect_separation_is_one():
    assert ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_gini_equals_two_auc_minus_one():
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.3, 2000)
    s = np.where(y == 1, rng.normal(1, 1, 2000), rng.normal(0, 1, 2000))
    assert gini(y, s) == pytest.approx(2 * auc(y, s) - 1)


def test_auc_matches_bruteforce_roc():
    rng = np.random.default_rng(12)
    y = rng.binomial(1, 0.4, 500)
    s = rng.normal(0, 1, 500) + 0.6 * y
    pos = s[y == 1]
    neg = s[y == 0]
    brute = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert auc(y, s) == pytest.approx(brute)


def test_rejects_single_class():
    with pytest.raises(ValueError):
        auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        ks([0, 1], [0.1, 0.2, 0.3])
