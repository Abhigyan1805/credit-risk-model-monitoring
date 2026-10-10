"""Tests for the RAG helpers."""

import pytest

from reporting.rag import is_breach, rag_color, rag_from_value, worst_rag


def test_higher_worse_thresholds():
    assert rag_from_value(0.05, 0.10, 0.25, "higher_worse") == "GREEN"
    assert rag_from_value(0.10, 0.10, 0.25, "higher_worse") == "AMBER"
    assert rag_from_value(0.20, 0.10, 0.25, "higher_worse") == "AMBER"
    assert rag_from_value(0.25, 0.10, 0.25, "higher_worse") == "RED"


def test_lower_worse_thresholds():
    assert rag_from_value(0.45, 0.40, 0.30, "lower_worse") == "GREEN"
    assert rag_from_value(0.35, 0.40, 0.30, "lower_worse") == "AMBER"
    assert rag_from_value(0.25, 0.40, 0.30, "lower_worse") == "RED"


def test_missing_value_is_na():
    assert rag_from_value(None, 0.1, 0.25, "higher_worse") == "NA"


def test_unknown_direction_raises():
    with pytest.raises(ValueError):
        rag_from_value(0.5, 0.1, 0.25, "sideways")


def test_worst_rag():
    assert worst_rag(["GREEN", "AMBER", "GREEN"]) == "AMBER"
    assert worst_rag(["GREEN", "RED", "AMBER"]) == "RED"
    assert worst_rag(["NA", "NA"]) == "NA"
    assert worst_rag([]) == "NA"
    # NA never outranks a real status.
    assert worst_rag(["NA", "GREEN"]) == "GREEN"


def test_is_breach_and_color():
    assert is_breach("AMBER") and is_breach("RED")
    assert not is_breach("GREEN") and not is_breach("NA")
    assert rag_color("RED") == "#c0392b"
    assert rag_color("garbage") == "#6b7280"
