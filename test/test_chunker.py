"""Tests for app.ingest.chunker — pure logic, no external deps."""

import pytest

from app.ingest.chunker import chunk_text


def test_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []

def test_short_text_single_chunk():
    text = "Hello world."
    assert chunk_text(text, chunk_size=100, overlap=0) == [text]

def test_splits_long_text():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 4
    for c in chunks:
        assert len(c) <= 500


def test_chunks_cover_original():
    text = "abcdefghij " * 100  # 1200 chars
    chunks = chunk_text(text, chunk_size=300, overlap=20)
    # Every word should appear at least once
    for word in ["abcdefghij"]:
        joined = " ".join(chunks)
        assert word in joined


def test_prefers_paragraph_boundary():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    assert len(chunks) >= 2
    # Breaks should be at \n\n boundaries when possible
    assert any(c.endswith("paragraph.") or c.endswith("\n\n") for c in chunks[:2])


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("x" * 100, chunk_size=10, overlap=-1)
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("x" * 100, chunk_size=10, overlap=10)