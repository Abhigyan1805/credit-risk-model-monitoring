"""monitoring component. See docs/CONTRACTS.md for the interface this package must implement.

This package turns raw monthly model outputs (scores, observed defaults and
model features) into the monitoring evidence a bank reviews every month:

* stability      -- population stability index (PSI) and characteristic
                    stability index (CSI) against the development reference;
* discrimination -- AUC, Gini and KS on seasoned vintages that now have
                    observed outcomes;
* calibration    -- predicted vs observed default rates by score band, an
                    exact binomial test per band and an overall
                    Hosmer-Lemeshow test;
* alerting       -- traffic-light (GREEN/AMBER/RED) thresholds held in
                    ``config/thresholds.yaml``;
* results table  -- the long ``RESULT_COLUMNS`` table defined in
                    ``docs/CONTRACTS.md`` (written by ``monitoring.run``).

Everything is pure numpy / pandas / scipy so it runs on a small sample and is
easy to explain line by line in an interview.
"""

from monitoring.calibration import calibration_bands, hosmer_lemeshow
from monitoring.discrimination import auc, gini, ks
from monitoring.stability import (
    csi,
    csi_categorical,
    psi,
    psi_from_counts,
    psi_from_proportions,
)

__all__ = [
    "psi",
    "psi_from_counts",
    "psi_from_proportions",
    "csi",
    "csi_categorical",
    "auc",
    "gini",
    "ks",
    "calibration_bands",
    "hosmer_lemeshow",
]

__version__ = "0.1.0"
