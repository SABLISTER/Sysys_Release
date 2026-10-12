"""Regression tests for full-text retrieval success counting."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from gui.workers import _fulltext_result_success


def test_fulltext_result_success_accepts_explicit_success_flag():
    assert _fulltext_result_success({"success": True, "text": ""}) is True


def test_fulltext_result_success_infers_success_from_extracted_text():
    result = {"text": "Extracted article body", "format": "pdf"}

    assert _fulltext_result_success(result) is True


def test_fulltext_result_success_rejects_empty_or_failed_results():
    assert _fulltext_result_success({"success": False, "text": "   "}) is False
    assert _fulltext_result_success({"format": "error"}) is False
