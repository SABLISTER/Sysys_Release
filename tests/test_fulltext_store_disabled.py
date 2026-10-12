"""Tests for fulltext_store memory/control paths."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from process.fulltext_store import update_fulltext_store


def test_update_fulltext_store_can_disable_paperqa(tmp_path):
    text_path = tmp_path / "paper.txt"
    source_path = tmp_path / "paper.pdf"
    text_path.write_text("extracted text", encoding="utf-8")
    source_path.write_bytes(b"%PDF original bytes")

    rows, metadata = update_fulltext_store(
        [{"path": str(text_path), "source_path": str(source_path), "source_format": "pdf"}],
        tmp_path / "store",
        workers=1,
        paperqa_enabled=False,
    )

    assert metadata["archived_originals"] == 1
    paperqa = metadata["paperqa"]
    assert paperqa["available"] is False
    assert paperqa["skipped"] is True
    assert paperqa["reason"] == "disabled"
