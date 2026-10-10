"""Tests for CSV extract writing and the data dictionary."""

import csv
import json
import os

import pytest

from reporting.extracts import (
    DATA_DICTIONARY,
    EXTRACT_FILES,
    write_csv,
    write_data_dictionary,
    write_extracts,
)
from reporting.schema import normalize_results

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "synthetic_monitoring_results.json"
)


@pytest.fixture()
def results():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        return normalize_results(json.load(fh))


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_write_csv_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "x.csv")
    write_csv(path, ["a", "b"], [{"a": 1, "b": 2.5}, {"a": None, "b": 3}])
    rows = _read_csv(path)
    assert rows[0]["a"] == "1"
    assert rows[1]["a"] == ""
    assert float(rows[0]["b"]) == 2.5


def test_write_extracts_creates_all_files(results, tmp_path):
    written = write_extracts(results, str(tmp_path))
    for name in EXTRACT_FILES:
        assert os.path.exists(written[name]), name
    assert os.path.exists(written["data_dictionary_csv"])
    assert os.path.exists(written["data_dictionary_md"])
    fact_rows = _read_csv(written["fact"])
    assert len(fact_rows) == len(results.periods)
    assert fact_rows[0]["period"]


def test_extracts_are_utf8_and_not_empty(results, tmp_path):
    written = write_extracts(results, str(tmp_path))
    for path in written.values():
        assert os.path.getsize(path) > 0
        with open(path, "r", encoding="utf-8") as fh:
            assert fh.read(1)


def test_data_dictionary_covers_every_column(results, tmp_path):
    from reporting.tables import build_all_tables

    tables = build_all_tables(results)
    written = write_extracts(results, str(tmp_path))
    dictionary = _read_csv(written["data_dictionary_csv"])
    documented = {(r["table"], r["column"]) for r in dictionary}
    for table, (columns, _) in tables.items():
        for column in columns:
            assert (table, column) in documented, f"{table}.{column} undocumented"


def test_data_dictionary_markdown(tmp_path):
    csv_path, md_path = write_data_dictionary(str(tmp_path))
    with open(md_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    assert "# Data dictionary" in content
    assert "`monitoring_fact.csv`" not in content  # keyed by logical table name
    assert "`rag_overall`" in content
    assert DATA_DICTIONARY["fact"]["psi"][1]
