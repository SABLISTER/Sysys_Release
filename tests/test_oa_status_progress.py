"""Regression tests for OA lookup progress reporting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stages import s7_fulltext


class _NoNetworkRequests:
    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("cached OA lookup should not hit the network")


def test_check_oa_status_accepts_progress_label_for_cached_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _NoNetworkRequests)
    cache_path = tmp_path / "oa_status.json"
    cache_path.write_text(
        json.dumps({
            "10.1234/example": {
                "is_oa": True,
                "oa_url": "https://example.test/paper.pdf",
                "host_type": "repository",
            }
        }),
        encoding="utf-8",
    )
    progress = []

    rows = s7_fulltext.check_oa_status(
        [{"doi": "10.1234/example", "title": "Example"}],
        cache_path=cache_path,
        progress_callback=progress.append,
        progress_label="Checking corpus open-access availability",
    )

    assert rows[0]["is_oa"] is True
    assert rows[0]["oa_url"] == "https://example.test/paper.pdf"
    assert progress == [{
        "message": "Checking corpus open-access availability: 1/1 checked",
        "current": 1,
        "total": 1,
    }]


def test_check_oa_status_preserves_string_progress_without_label(tmp_path, monkeypatch):
    monkeypatch.setattr(s7_fulltext, "_import_requests", lambda: _NoNetworkRequests)
    cache_path = tmp_path / "oa_status.json"
    cache_path.write_text(
        json.dumps({"10.1234/example": {"is_oa": False, "oa_url": "", "host_type": ""}}),
        encoding="utf-8",
    )
    progress = []

    s7_fulltext.check_oa_status(
        [{"doi": "10.1234/example", "title": "Example"}],
        cache_path=cache_path,
        progress_callback=progress.append,
    )

    assert progress == ["  OA check [1/1] — 0 queried, 1 cached"]
