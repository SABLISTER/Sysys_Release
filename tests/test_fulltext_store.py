"""Tests for full-text storage and archival helpers."""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from acquire.fulltext import reusable_download_record
from process.fulltext_store import archive_original_sources


def test_archive_original_sources_compresses_source_and_keeps_text(tmp_path):
    text_path = tmp_path / "paper.txt"
    source_path = tmp_path / "paper.pdf"
    text_path.write_text("extracted text", encoding="utf-8")
    source_path.write_bytes(b"%PDF original bytes")

    rows = archive_original_sources(
        [{"path": str(text_path), "source_path": str(source_path), "source_format": "pdf"}],
        tmp_path / "source_archive",
        workers=3,
    )

    row = rows[0]
    archive_path = Path(row["source_archive_path"])
    assert text_path.exists()
    assert not source_path.exists()
    assert archive_path.exists()
    assert row["source_path"] == ""
    with gzip.open(archive_path, "rb") as f:
        assert f.read() == b"%PDF original bytes"


def test_archive_original_sources_ignores_blank_source_path(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text("extracted text", encoding="utf-8")

    rows = archive_original_sources(
        [{"path": str(text_path), "source_path": "", "source_format": "text"}],
        tmp_path / "source_archive",
        workers=3,
    )

    assert rows == [{"path": str(text_path), "source_path": "", "source_format": "text"}]
    assert text_path.exists()
    assert not (tmp_path / "source_archive").exists()


def test_reusable_download_record_accepts_archived_source(tmp_path):
    text_path = tmp_path / "paper.txt"
    archive_path = tmp_path / "paper.pdf.gz"
    text_path.write_text("extracted text", encoding="utf-8")
    archive_path.write_bytes(b"compressed")

    assert reusable_download_record({
        "path": str(text_path),
        "source_path": str(tmp_path / "paper.pdf"),
        "source_archive_path": str(archive_path),
    })


def test_add_rows_to_paperqa_skips_pickle_when_nothing_pending(tmp_path):
    import asyncio

    from process.fulltext_store import _add_rows_to_paperqa, _row_key

    rows = [{"doi": "10.1/a"}, {"doi": "10.1/b"}]
    manifest = {"indexed": [_row_key(r) for r in rows]}
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    # Bytes that fail at unpickle time prove the fast path never loads the
    # pickle when every row is already indexed.
    (db_dir / "paperqa_docs.pkl").write_bytes(b"not a pickle")

    result = asyncio.run(_add_rows_to_paperqa(rows, db_dir, manifest))

    assert result["added"] == 0
    assert result["skipped"] == len(rows)
    assert result["indexed_total"] == len(rows)
    assert manifest["indexed"] == [_row_key(r) for r in rows]


def test_missing_pickle_invalidates_stale_manifest_early_exit(tmp_path):
    """When the pickle is gone but the manifest claims rows are indexed,
    the early-exit fast path must NOT fire — otherwise the store is never
    rebuilt and later runs keep hitting the same empty fast path."""
    import asyncio
    from unittest.mock import patch

    from process.fulltext_store import _add_rows_to_paperqa, _row_key

    text_file = tmp_path / "paper.txt"
    text_file.write_text("some extracted text", encoding="utf-8")

    rows = [{"doi": "10.1/a", "title": "Paper A", "path": str(text_file)}]
    manifest = {"indexed": [_row_key(r) for r in rows]}
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    # No paperqa_docs.pkl on disk — simulates a deleted/lost pickle.

    async def fake_aadd(self, *a, **kw):
        pass

    with patch("paperqa.Docs.aadd", fake_aadd):
        result = asyncio.run(_add_rows_to_paperqa(rows, db_dir, manifest))

    assert result["added"] == 1, (
        "Row should be re-indexed when pickle is missing"
    )
    assert (db_dir / "paperqa_docs.pkl").exists()


def test_missing_pickle_invalidates_stale_manifest_loop_path(tmp_path):
    """When the pickle is gone and new + old rows are both present, old rows
    must not be skipped in the loop because their keys were in the manifest."""
    import asyncio
    from unittest.mock import patch

    from process.fulltext_store import _add_rows_to_paperqa, _row_key

    text_a = tmp_path / "a.txt"
    text_b = tmp_path / "b.txt"
    text_a.write_text("text a", encoding="utf-8")
    text_b.write_text("text b", encoding="utf-8")

    old_row = {"doi": "10.1/a", "title": "Old", "path": str(text_a)}
    new_row = {"doi": "10.1/b", "title": "New", "path": str(text_b)}
    manifest = {"indexed": [_row_key(old_row)]}
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    async def fake_aadd(self, *a, **kw):
        pass

    with patch("paperqa.Docs.aadd", fake_aadd):
        result = asyncio.run(
            _add_rows_to_paperqa([old_row, new_row], db_dir, manifest)
        )

    assert result["added"] == 2, (
        "Both old and new rows must be indexed when the pickle is missing"
    )
    assert "10.1/a" in manifest["indexed"]
    assert "10.1/b" in manifest["indexed"]
