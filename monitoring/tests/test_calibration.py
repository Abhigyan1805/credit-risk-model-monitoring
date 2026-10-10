"""Hand-computed tests for calibration metrics."""

import numpy as np
import pytest
from scipy.stats import chi2

from monitoring.calibration import calibration_bands, hosmer_lemeshow


def _blocks():
    """Three 100-loan bands with predicted PD 0.1 / 0.2 / 0.3."""
    scores = np.concatenate(
        [np.full(100, 0.1), np.full(100, 0.2), np.full(100, 0.3)]
    )
    labels = np.zeros(300, dtype=int)
    labels[0:20] = 1        # band 0: 20 defaults vs 10 expected  -> component 100/9
    labels[100:120] = 1     # band 1: 20 defaults vs 20 expected  -> 0
    labels[200:230] = 1     # band 2: 30 defaults vs 30 expected  -> 0
    return labels, scores


def test_hosmer_lemeshow_hand_computed():
    labels, scores = _blocks()
    result = hosmer_lemeshow(labels, scores, n_bands=3)
    # HL = (20-10)^2 / (100*0.1*0.9) = 11.1111, dof = 3-2 = 1
    assert result["statistic"] == pytest.approx(100.0 / 9.0, abs=1e-9)
    assert result["dof"] == 1
    assert result["p_value"] == pytest.approx(float(chi2.sf(100.0 / 9.0, 1)), abs=1e-12)
    assert result["p_value"] < 0.01


def test_hosmer_lemeshow_perfect_calibration():
    # observed equals expected in every band -> statistic 0, p = 1
    scores = np.concatenate(
        [np.full(100, 0.1), np.full(100, 0.2), np.full(100, 0.3)]
    )
    labels = np.zeros(300, dtype=int)
    labels[0:10] = 1
    labels[100:120] = 1
    labels[200:230] = 1
    result = hosmer_lemeshow(labels, scores, n_bands=3)
    assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-12)


def test_hosmer_lemeshow_requires_three_bands():
    labels, scores = _blocks()
    with pytest.raises(ValueError):
        hosmer_lemeshow(labels, scores, n_bands=2)


def test_hosmer_lemeshow_rejects_degenerate_band():
    scores = np.zeros(100)
    labels = np.zeros(100, dtype=int)
    labels[:10] = 1
    with pytest.raises(ValueError):
        hosmer_lemeshow(labels, scores, n_bands=3)


def test_calibration_bands_columns_and_rates():
    labels, scores = _blocks()
    bands = calibration_bands(labels, scores, n_bands=3)
    assert list(bands["n"]) == [100, 100, 100]
    assert bands["observed_defaults"].tolist() == [20, 20, 30]
    assert bands["observed_default_rate"].tolist() == pytest.approx([0.2, 0.2, 0.3])
    assert bands["mean_predicted_pd"].tolist() == pytest.approx([0.1, 0.2, 0.3])
    # band 0 is badly miscalibrated -> tiny binomial p-value
    assert bands.loc[0, "binomial_pvalue"] < 0.05
    # bands 1 and 2 match expectation -> not significant
    assert bands.loc[1, "binomial_pvalue"] > 0.5
    assert bands.loc[2, "binomial_pvalue"] > 0.5


def test_calibration_bands_deviation_sign():
    labels, scores = _blocks()
    bands = calibration_bands(labels, scores, n_bands=3)
    assert bands.loc[0, "deviation"] == pytest.approx(0.1)
    assert bands.loc[1, "deviation"] == pytest.approx(0.0)


def test_calibration_bands_n_bands_too_many():
    with pytest.raises(ValueError):
        calibration_bands([0, 1], [0.2, 0.8], n_bands=5)


def test_well_calibrated_model_has_high_hl_pvalue():
    rng = np.random.default_rng(20)
    p = rng.uniform(0.05, 0.5, 4000)
    y = rng.binomial(1, p)
    result = hosmer_lemeshow(y, p, n_bands=10)
    assert result["p_value"] > 0.01


def test_miscalibrated_model_has_low_hl_pvalue():
    rng = np.random.default_rng(21)
    p = rng.uniform(0.1, 0.3, 4000)
    # true default rate is double the prediction
    y = rng.binomial(1, np.clip(2 * p, 0, 1))
    result = hosmer_lemeshow(y, p, n_bands=10)
    assert result["p_value"] < 0.001
