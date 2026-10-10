"""Idempotent building blocks the DAG calls. ``python -m orchestration.steps <step> [--month YYYY-MM]``.

Heavy one-off steps (prepare data, train the model) run only when their output is missing, so every
monthly DAG run is safe to repeat and a backfill over many months does the setup exactly once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from common import paths


def _run(*module_args: str) -> None:
    print("+", sys.executable, "-m", *module_args, flush=True)
    subprocess.run([sys.executable, "-m", *module_args], check=True)


def ensure_prepared() -> None:
    if paths.prepared_loans().exists():
        print(f"prepared data present: {paths.prepared_loans()}")
        return
    if os.environ.get("CRM_USE_SAMPLE") == "1":
        _run("ingest.prepare", "--from-sample")
    else:
        _run("ingest.download")
        _run("ingest.prepare")


def ensure_model() -> None:
    if (paths.model_dir() / "metadata.json").exists() and paths.scored_path("dev").exists():
        print("model and dev reference present")
        return
    _run("model.train")


STEPS = {
    "ensure_prepared": lambda m: ensure_prepared(),
    "ensure_model": lambda m: ensure_model(),
    "score": lambda m: _run("model.score", "--month", m),
    "monitor": lambda m: _run("monitoring.run", "--month", m),
    "report": lambda m: _run("reporting.report", "--month", m),
    "extracts": lambda m: _run("reporting.extracts"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--month")
    args = ap.parse_args()
    if args.step in {"score", "monitor", "report"} and not args.month:
        ap.error("--month required")
    STEPS[args.step](args.month)


if __name__ == "__main__":
    main()
