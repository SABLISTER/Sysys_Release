"""Tests for the ninth fix batch: Stage 2b memory footprint.

The problem: Stage 2b read every downloaded full text into ``downloaded_map``,
built a second full copy of the corpus (``enriched``), and then serialized the
whole thing to one in-memory string via ``write_json_atomic`` — three
corpus-sized allocations live at once.

The fix: ``downloaded_map`` holds only file locations, each full text is read
inside a generator immediately before its record is serialized, and the new
``atomic_write_json_array`` primitive streams the records into an atomic JSON
array so peak memory is one document rather than all of them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.io import atomic_write_json_array
import stages.s2b_fulltext as s2b


# ── atomic_write_json_array ───────────────────────────────────────────────


def test_array_round_trip_and_count(tmp_path: Path):
    target = tmp_path / "out.json"
    records = ({"i": i, "text": "x" * i} for i in range(5))

    count = atomic_write_json_array(target, records)

    assert count == 5
    assert json.loads(target.read_text()) == [
        {"i": i, "text": "x" * i} for i in range(5)
    ]


def test_array_empty_iterable_writes_valid_json(tmp_path: Path):
    target = tmp_path / "out.json"
    assert atomic_write_json_array(target, iter(())) == 0
    assert json.loads(target.read_text()) == []


def test_array_crash_mid_stream_keeps_previous_file(tmp_path: Path):
    target = tmp_path / "out.json"
    target.write_text('[{"old": true}]')

    def _explodes():
        yield {"new": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write_json_array(target, _explodes())

    assert json.loads(target.read_text()) == [{"old": True}]


def test_array_preserves_existing_mode(tmp_path: Path):
    target = tmp_path / "out.json"
    target.write_text("[]")
    os.chmod(target, 0o600)

    atomic_write_json_array(target, [{"a": 1}])

    assert (target.stat().st_mode & 0o777) == 0o600
    assert json.loads(target.read_text()) == [{"a": 1}]


# ── Stage 2b streaming enrichment ─────────────────────────────────────────


def _make_corpus(tmp_path: Path):
    """Three papers: one with readable full text, one whose downloaded file is
    missing from disk, one never downloaded."""
    from core.config import Config

    cfg = Config(project_root=tmp_path)
    cfg.corpus_dir.mkdir(parents=True, exist_ok=True)

    corpus = [
        {"pmid": "111", "title": "Readable", "abstract": "A1"},
        {"pmid": "222", "title": "Unreadable", "abstract": "A2"},
        {"pmid": "333", "title": "Never downloaded", "abstract": "A3"},
    ]
    (cfg.corpus_dir / "corpus.json").write_text(json.dumps(corpus))

    download_dir = cfg.data_dir / "fulltext_corpus"
    download_dir.mkdir(parents=True, exist_ok=True)
    text_path = download_dir / "111.txt"
    text_path.write_text("Full text one.")

    rows = [
        {"key": "111", "title": "Readable", "status": "downloaded",
         "path": str(text_path), "source_format": "html",
         "source_url": "https://example.org/111"},
        {"key": "222", "title": "Unreadable", "status": "downloaded",
         "path": str(download_dir / "missing.txt"), "source_format": "pdf",
         "source_url": "https://example.org/222"},
    ]
    return cfg, download_dir, rows


def _run_stage(cfg, rows, monkeypatch, cancel_event=None):
    monkeypatch.setattr(s2b, "_download_fulltext",
                        lambda *a, **k: list(rows))
    import process.fulltext_store as store

    monkeypatch.setattr(
        store, "update_fulltext_store",
        lambda downloaded, *a, **k: (downloaded, {
            "archive_dir": "", "archived_originals": 0,
            "paperqa": {"available": False},
        }),
    )
    return s2b.run(cfg, cancel_event=cancel_event)


def test_streaming_enrichment_reads_text_from_disk(tmp_path, monkeypatch):
    cfg, _download_dir, rows = _make_corpus(tmp_path)

    result = _run_stage(cfg, rows, monkeypatch)

    records = {r["pmid"]: r for r in
               json.loads((cfg.corpus_dir / "corpus_fulltext.json").read_text())}
    assert len(records) == 3
    assert records["111"]["text_source"] == "full_text"
    assert records["111"]["full_text"] == "Full text one."
    assert records["111"]["full_text_chars"] == len("Full text one.")
    # The unreadable download falls back to the abstract instead of crashing.
    assert records["222"]["text_source"] == "abstract"
    assert "full_text" not in records["222"]
    assert records["333"]["text_source"] == "abstract"

    assert result["paper_count"] == 3
    assert result["fulltext_count"] == 1
    assert not result.get("cancelled")


def test_download_log_merge_keeps_prior_entries(tmp_path, monkeypatch):
    cfg, download_dir, rows = _make_corpus(tmp_path)
    log_path = download_dir / "download_log.json"
    log_path.write_text(json.dumps([
        {"key": "999", "title": "Checked before", "status": "no_usable_text"},
    ]))

    _run_stage(cfg, rows, monkeypatch)

    by_key = {row["key"]: row for row in json.loads(log_path.read_text())}
    # The pre-existing negative-result entry survives the merge, so the next
    # run still skips the paper instead of re-downloading it.
    assert by_key["999"]["status"] == "no_usable_text"
    assert by_key["111"]["status"] == "downloaded"
    assert by_key["222"]["status"] == "downloaded"


class _CancelAfter:
    """is_set() returns False for the first `n` polls, then True forever."""

    def __init__(self, n: int):
        self.remaining = n

    def is_set(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return False
        return True


def test_cancel_during_enrichment_still_covers_whole_corpus(tmp_path, monkeypatch):
    cfg, _download_dir, rows = _make_corpus(tmp_path)
    # Two polls happen before enrichment (post-OA and post-download); the
    # third poll is the first downloaded paper inside the generator.
    cancel = _CancelAfter(2)

    result = _run_stage(cfg, rows, monkeypatch, cancel_event=cancel)

    assert result["cancelled"] is True
    records = json.loads((cfg.corpus_dir / "corpus_fulltext.json").read_text())
    # The output still covers the full corpus — remaining papers fall back to
    # their abstracts, matching what a partial run produced before.
    assert len(records) == 3
    assert all(r["text_source"] == "abstract" for r in records)
    assert result["fulltext_count"] == 0
