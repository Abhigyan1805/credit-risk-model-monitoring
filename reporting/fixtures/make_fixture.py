"""Generate a deterministic synthetic monitoring-results fixture.

This lets every reporting test and the demo report run without the real
LendingClub dataset (which builder #1 runs once). The fixture deliberately
contains a healthy early period and a deteriorating late period so that all
three RAG colours appear and every chart has something to show.

Run directly to (re)write the committed fixture::

    python -m reporting.fixtures.make_fixture
"""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timezone

from reporting.rag import DEFAULT_THRESHOLDS, classify_breaches

#: Ten risk grades, safest first. PD increases with the grade.
BANDS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_N_BANDS = len(BANDS)

FEATURES = [
    "annual_inc",
    "dti",
    "fico_range_low",
    "int_rate",
    "loan_amnt",
    "term",
    "home_ownership",
    "emp_length",
    "purpose",
    "verification_status",
]

#: Development ("reference") model metrics used to judge discrimination drift.
REFERENCE_METRICS = {"auc": 0.720, "gini": 0.440, "ks": 0.335}

START_YEAR, START_MONTH = 2017, 1
N_MONTHS = 18


def _month_label(offset: int) -> str:
    month_index = (START_MONTH - 1) + offset
    year = START_YEAR + month_index // 12
    month = month_index % 12 + 1
    return f"{year:04d}-{month:02d}"


def _band_bounds(band_index: int) -> tuple[float, float]:
    width = 1.0 / _N_BANDS
    return round(band_index * width, 3), round((band_index + 1) * width, 3)


def _expected_band_shares() -> list[float]:
    # A smooth, high-weight-to-good-grades reference distribution.
    weights = [math.exp(-((i - 2.0) ** 2) / 6.0) for i in range(_N_BANDS)]
    total = sum(weights)
    return [w / total for w in weights]


def build_fixture(seed: int = 20261012) -> dict:
    random.seed(seed)
    expected_shares = _expected_band_shares()

    periods: list[dict] = []
    score_distribution: list[dict] = []
    calibration_by_band: list[dict] = []
    feature_csi_rows: list[dict] = []

    for offset in range(N_MONTHS):
        period = _month_label(offset)
        # Deterioration ramps up only in the final third of the timeline.
        drift = max(0.0, (offset - 11) / 6.0)

        n_obs = int(900 + 40 * offset + random.randint(-60, 60))
        default_rate = round(0.048 + 0.004 * offset + 0.03 * drift + random.uniform(-0.003, 0.003), 4)

        psi = round(0.03 + 0.006 * offset + 0.20 * drift ** 2 + random.uniform(-0.004, 0.004), 4)
        csi = round(0.035 + 0.005 * offset + 0.18 * drift ** 2 + random.uniform(-0.004, 0.004), 4)
        auc = round(0.720 - 0.0015 * offset - 0.045 * drift + random.uniform(-0.003, 0.003), 4)
        gini = round(2 * auc - 1, 4)
        ks = round(max(0.0, auc - 0.385) + random.uniform(-0.004, 0.004), 4)
        calibration_gap = round(0.008 + 0.001 * offset + 0.035 * drift + random.uniform(-0.002, 0.002), 4)

        metrics = {
            "psi": psi,
            "csi": csi,
            "auc": auc,
            "gini": gini,
            "ks": ks,
            "calibration_gap": calibration_gap,
        }
        periods.append(
            {
                "period": period,
                "n_obs": n_obs,
                "default_rate": default_rate,
                "metrics": metrics,
            }
        )

        # --- score distribution (population drift towards riskier bands) -----
        raw = list(expected_shares)
        for i in range(_N_BANDS):
            raw[i] *= 1.0 + (i - 4) * 0.10 * drift
        raw_total = sum(raw)
        shares = [r / raw_total for r in raw]
        for i, band in enumerate(BANDS):
            lo, hi = _band_bounds(i)
            count = int(round(shares[i] * n_obs))
            score_distribution.append(
                {
                    "period": period,
                    "band": band,
                    "score_min": lo,
                    "score_max": hi,
                    "count": count,
                    "share": round(shares[i], 5),
                    "expected_share": round(expected_shares[i], 5),
                }
            )

            # --- calibration by band (predicted vs observed PD) --------------
            predicted_pd = round(_band_bounds(i)[0] + 0.03 + 0.015 * i, 4)
            observed_pd = round(predicted_pd * (1.0 + 0.10 * drift) + random.uniform(-0.002, 0.002), 4)
            calibration_by_band.append(
                {
                    "period": period,
                    "band": band,
                    "count": count,
                    "predicted_pd": predicted_pd,
                    "observed_pd": observed_pd,
                }
            )

        # --- feature CSI ------------------------------------------------------
        for j, feature in enumerate(FEATURES):
            base = 0.02 + 0.004 * (j % 4)
            csi_val = round(base + 0.01 * offset * 0.15 + 0.16 * drift ** 2 * (1 + 0.1 * j) + random.uniform(-0.003, 0.003), 4)
            feature_csi_rows.append({"period": period, "feature": feature, "csi": csi_val})

    payload = {
        "model": {
            "name": "LendingClub PD scorecard",
            "version": "v1.0.0",
            "target": "loan default (charged-off) within 36 months",
            "reference_period": _month_label(0),
            "reference_metrics": REFERENCE_METRICS,
        },
        "generated_at": datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc).isoformat(),
        "periods": periods,
        "score_distribution": score_distribution,
        "calibration_by_band": calibration_by_band,
        "feature_csi": feature_csi_rows,
        "thresholds": DEFAULT_THRESHOLDS,
    }

    # Derive breaches from the period metrics so the fixture stays consistent.
    breaches = []
    for period in periods:
        for breach in classify_breaches(period["metrics"], REFERENCE_METRICS):
            breaches.append({"period": period["period"], **breach})
    payload["breaches"] = breaches
    return payload


def main() -> None:
    out_path = os.path.join(os.path.dirname(__file__), "synthetic_monitoring_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(build_fixture(), fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
