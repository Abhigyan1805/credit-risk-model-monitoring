"""Reporting package for the credit-risk model monitoring system.

This package turns the monitoring-results produced by the monitoring stage
into:

* tidy, Tableau-ready CSV extracts (see :mod:`reporting.extracts`),
* an auto-generated per-run HTML monitoring report (see
  :mod:`reporting.report`), and
* a fallback interactive HTML dashboard (see :mod:`reporting.dashboard`).

The design goal is explainability: every number in the outputs is derived from
the monitoring results by small, unit-tested, pure-Python functions, and every
threshold used to colour a status is a named constant a reviewer can point at.

Public entry point for a run: :func:`reporting.pipeline.generate_outputs`.
"""

__all__ = ["generate_outputs"]


def generate_outputs(*args, **kwargs):
    """Lazy proxy for :func:`reporting.pipeline.generate_outputs`.

    Imported lazily so lightweight submodules (tables, extracts) can be used
    without pulling in the charting/rendering dependencies.
    """
    from reporting.pipeline import generate_outputs as _generate

    return _generate(*args, **kwargs)
