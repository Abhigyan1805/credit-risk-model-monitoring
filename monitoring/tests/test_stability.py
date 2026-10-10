"""Hand-computed tests for PSI / CSI."""

import numpy as np
import pytest

from monitoring.stability import (
    csi,
    csi_categorical,
    psi,
    psi_from_counts,
    psi_from_proportions,
    quantile_bin_edges,
)

# Worked example (textbook PSI):
#   expected = [0.20, 0.30, 0.50], actual = [0.10, 0.40, 0.50]
#   bin 0: (0.10-0.20)*ln(0.10/0.20) = -0.10 * -0.693147 = 0.0693147
#   bin 1: (0.40-0.30)*ln(0.40/0.30) =  0.10 *  0.287682 = 0.0287682
#   bin 2: (0.50-0.50)*ln(1)         =  0
#   total = 0.0980829
HAND_PSI = 0.0693147 + 0.0287682


def test_psi_hand_computed_proportions():
    value = psi_from_proportions([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
    assert value == pytest.approx(HAND_PSI, abs=1e-6)


def test_psi_identical_distributions_is_zero():
    props = [0.1, 0.2, 0.3, 0.4]
    assert psi_from_proportions(props, props) == pytest.approx(0.0, abs=1e-12)


def test_psi_from_counts_matches_proportions():
    counts = psi_from_counts([20, 30, 50], [10, 40, 50])
    assert counts == pytest.approx(HAND_PSI, abs=1e-6)


def test_psi_symmetric_in_swap():
    a = psi_from_proportions([0.2, 0.3, 0.5], [0.1, 0.4, 0.5])
    b = psi_from_proportions([0.1, 0.4, 0.5], [0.2, 0.3, 0.5])
    assert a == pytest.approx(b, abs=1e-12)


def test_psi_empty_bin_is_finite():
    value = psi_from_proportions([0.5, 0.5, 0.0], [0.5, 0.4, 0.1])
    assert np.isfinite(value)


def test_psi_must_align_bins():
    with pytest.raises(ValueError):
        psi_from_proportions([0.5, 0.5], [0.3, 0.3, 0.4])


def test_quantile_edges_open_ended():
    edges = quantile_bin_edges(np.arange(100), n_bins=10)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert len(edges) == 11


def test_psi_on_raw_scores_identical_is_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=2000)
    assert psi(sample, sample, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_psi_detects_location_shift_beyond_amber():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.0, 1, 5000)
    assert psi(baseline, shifted, n_bins=10) > 0.25


def test_frozen_edges_are_reused():
    rng = np.random.default_rng(2)
    baseline = rng.normal(0, 1, 2000)
    edges = quantile_bin_edges(baseline, n_bins=10)
    same = psi(baseline, baseline, bin_edges=edges)
    assert same == pytest.approx(0.0, abs=1e-12)


def test_csi_numeric_matches_psi():
    rng = np.random.default_rng(3)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(0.2, 1, 1000)
    assert csi(baseline, current) == pytest.approx(psi(baseline, current))


def test_csi_categorical_hand_values():
    # categories A,B,C,D; baseline proportions differ from current
    baseline = ["A"] * 20 + ["B"] * 30 + ["C"] * 50
    current = ["A"] * 10 + ["B"] * 40 + ["C"] * 50
    assert csi_categorical(baseline, current) == pytest.approx(HAND_PSI, abs=1e-6)


def test_csi_categorical_new_category_counts():
    baseline = ["A"] * 50 + ["B"] * 50
    current = ["A"] * 50 + ["B"] * 40 + ["C"] * 10
    value = csi_categorical(baseline, current)
    assert value > 0
    assert np.isfinite(value)
