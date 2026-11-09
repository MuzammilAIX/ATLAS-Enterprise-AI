"""Tests for app.ingest.extractor — pure logic, no network."""

import pytest

from app.ingest.extractor import extract_text


def test_extract_txt_from_bytes():
    data = b"Hello, ATLAS."
    assert extract_text(data, filename="test.txt") == "Hello, ATLAS."


def test_extract_txt_from_latin1_bytes():
    data = b"caf\xe9"  # 'café' in latin-1
    out = extract_text(data, filename="test.txt")
    assert "caf" in out


def test_extract_unknown_falls_back_to_txt():
    data = b"plain text content"
    out = extract_text(data, filename="unknown.xyz")
    assert "plain text" in out


def test_extract_detects_by_content_type():
    data = b"plain"
    out = extract_text(data, content_type="text/plain")
    assert out == "plain"


def test_extract_returns_string():
    # Minimum sanity — output is always str
    assert isinstance(extract_text(b"abc", filename="x.txt"), str)