"""Command-line demo for the monitoring engine.

Runs the full pipeline on a small **synthetic** loan book so the engine can be
exercised (and reviewed) without the full LendingClub file. Writes the three
output tables as CSV into an output directory.

Usage::

    python -m monitoring --out-dir out/monitoring --months 6

The synthetic book deliberately drifts in the later months so the demo lights
up amber/red alerts, which is useful when explaining the system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.runner import ModelSnapshot, run_monitoring


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_book(
    n_baseline: int = 3000,
    n_per_month: int = 500,
    n_months: int = 6,
    seed: int = 7,
) -> tuple[ModelSnapshot, dict[str, ModelSnapshot]]:
    """Generate a baseline and drifted monthly snapshots.

    The true relationship between features and default is fixed; a "model
    score" is that linear predictor plus noise. From month 4 the feature
    distributions and the intercept shift (population drift + a harsher
    economy), producing measurable PSI/CSI and calibration changes.
    """
    rng = np.random.default_rng(seed)

    def draw(n: int, drift: float, purpose_shift: float):
        fico = rng.normal(700 - 25 * drift, 40, n).clip(580, 820)
        dti = rng.normal(18 + 4 * drift, 6, n).clip(1, 40)
        loan_amnt = rng.normal(15000 + 2000 * drift, 5000, n).clip(1000, 40000)
        shift = float(np.clip(purpose_shift, 0.0, 1.0))
        purpose = rng.choice(
            ["debt", "car", "home", "other"],
            n,
            p=[0.5 - 0.2 * shift, 0.2 + 0.2 * shift, 0.2, 0.1],
        )
        intercept = -1.1 + 0.9 * drift
        logit = (
            intercept
            - 0.02 * (fico - 700)
            + 0.06 * (dti - 18)
            + 0.00003 * loan_amnt
            + 0.4 * (purpose == "other")
        )
        default = rng.binomial(1, _sigmoid(logit))
        score = _sigmoid(logit + rng.normal(0, 0.3, n))
        return ModelSnapshot(
            scores=score,
            labels=default,
            features={
                "fico": fico,
                "dti": dti,
                "loan_amnt": loan_amnt,
                "purpose": purpose,
            },
        )

    baseline = draw(n_baseline, drift=0.0, purpose_shift=0.0)
    months: dict[str, ModelSnapshot] = {}
    for m in range(1, n_months + 1):
        drift = 0.0 if m <= 3 else 0.15 * (m - 3)
        months[f"2024-{m:02d}"] = draw(
            n_per_month, drift=drift, purpose_shift=min(0.5, 0.15 * (m - 3))
        )
    return baseline, months


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the monitoring engine on synthetic data")
    parser.add_argument("--out-dir", default="out/monitoring", help="where to write CSVs")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--n-baseline", type=int, default=3000)
    parser.add_argument("--n-per-month", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    baseline, months = make_synthetic_book(
        n_baseline=args.n_baseline,
        n_per_month=args.n_per_month,
        n_months=args.months,
        seed=args.seed,
    )
    outputs = run_monitoring(baseline, months)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    summary = outputs["monthly_summary"]
    cols = [c for c in ("month", "psi_score", "gini", "ks", "hl_pvalue", "overall_rag") if c in summary]
    print("\nMonthly summary:")
    print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
