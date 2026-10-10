"""Tests for writing the CSV extracts and the data dictionary."""

import csv

import pandas as pd

from reporting import extracts
from reporting.data import load_scored_bands
from reporting.tables import build_extracts


def _read(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_write_extracts_creates_expected_files(results, crm_data):
    bands = load_scored_bands(extracts.available_months())
    written = extracts.write_extracts(results, bands)
    for name in extracts.EXTRACT_FILES:
        assert written[name].exists(), name
    assert written["data_dictionary_csv"].exists()
    assert written["data_dictionary_md"].exists()


def test_extract_headers_match_contract(results, crm_data):
    bands = load_scored_bands(extracts.available_months())
    written = extracts.write_extracts(results, bands)
    tables = build_extracts(results, bands)
    for name, path in written.items():
        if name.startswith("data_dictionary"):
            continue
        rows = _read(path)
        assert list(rows[0].keys()) == list(tables[name].columns)


def test_data_dictionary_documents_every_column(results, crm_data):
    bands = load_scored_bands(extracts.available_months())
    written = extracts.write_extracts(results, bands)
    dictionary = _read(written["data_dictionary_csv"])
    documented = {(r["table"], r["column"]) for r in dictionary}
    for table, df in build_extracts(results, bands).items():
        for column in df.columns:
            assert (table, column) in documented, f"{table}.{column} undocumented"


def test_data_dictionary_markdown(crm_data, tmp_path):
    _, md_path = extracts.write_data_dictionary(tmp_path)
    content = md_path.read_text(encoding="utf-8")
    assert "# Data dictionary" in content
    assert "`month_date`" in content


def test_main_writes_extracts(crm_data, capsys):
    extracts.main([])
    out = capsys.readouterr().out
    assert "monitoring_results:" in out
    assert (crm_data / "tableau" / "monthly_summary.csv").exists()


def test_month_dates_are_iso(results, crm_data):
    bands = load_scored_bands(extracts.available_months())
    written = extracts.write_extracts(results, bands)
    long_rows = _read(written["monitoring_results"])
    assert long_rows[0]["month_date"].endswith("-01")
    assert pd.Timestamp(long_rows[0]["month_date"]).day == 1
