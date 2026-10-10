"""Command-line entry point for a reporting run.

Examples
--------
Generate everything from the committed synthetic fixture::

    python -m reporting.cli --fixture --out-dir out/reporting

Generate from a real monitoring-results file, CSV extracts only::

    python -m reporting.cli --input results/monitoring_2026_10.json \\
        --out-dir out/reporting --formats extracts

Write a fully offline report (plotly.js embedded, no CDN)::

    python -m reporting.cli --fixture --out-dir out/reporting --inline-js
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from reporting.pipeline import DEFAULT_FORMATS, generate_outputs
from reporting.schema import load_results

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "synthetic_monitoring_results.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reporting",
        description="Generate the monitoring report, Tableau extracts and fallback dashboard.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", "-i", help="Path to a monitoring-results JSON file.")
    source.add_argument(
        "--fixture",
        action="store_true",
        help="Use the committed synthetic monitoring-results fixture (no dataset needed).",
    )
    parser.add_argument("--out-dir", "-o", required=True, help="Directory for generated artefacts.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=list(DEFAULT_FORMATS),
        default=list(DEFAULT_FORMATS),
        help="Which artefacts to generate (default: all).",
    )
    parser.add_argument(
        "--inline-js",
        action="store_true",
        help="Embed plotly.js in the HTML for a fully offline file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = _FIXTURE if args.fixture else args.input
    results = load_results(source)
    result = generate_outputs(results, args.out_dir, formats=args.formats, inline_js=args.inline_js)
    print(json.dumps(result["files"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
