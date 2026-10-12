from __future__ import annotations

import json
import os
from pathlib import Path

from acquire.fulltext import write_json_atomic
from core.io import (
    atomic_append_jsonl,
    atomic_write_json,
    atomic_write_text,
    jsonl_to_json_array,
)


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_atomic_write_json_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o640)

    atomic_write_json(path, {"ok": True})

    assert _mode(path) == 0o640
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_write_text_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text("old", encoding="utf-8")
    os.chmod(path, 0o644)

    atomic_write_text(path, "new")

    assert _mode(path) == 0o644
    assert path.read_text(encoding="utf-8") == "new"


def test_fulltext_write_json_atomic_keeps_ascii_escaping(tmp_path: Path):
    path = tmp_path / "fulltext.json"

    write_json_atomic(path, {"title": "café"})

    assert "caf\\u00e9" in path.read_text(encoding="utf-8")


def test_atomic_append_jsonl_appends_records(tmp_path: Path):
    path = tmp_path / "records.jsonl"

    atomic_append_jsonl(path, {"id": 1})
    atomic_append_jsonl(path, {"id": 2})

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1}
    assert json.loads(lines[1]) == {"id": 2}


def test_atomic_append_jsonl_preserves_existing_mode(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o640)

    atomic_append_jsonl(path, {"id": 1})
    atomic_append_jsonl(path, {"id": 2})

    assert _mode(path) == 0o640


def test_jsonl_to_json_array_streaming_conversion(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    json_path = tmp_path / "records.json"

    atomic_append_jsonl(jsonl_path, {"id": 1, "value": "a"})
    atomic_append_jsonl(jsonl_path, {"id": 2, "value": "b"})

    jsonl_to_json_array(jsonl_path, json_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data == [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]


def test_jsonl_to_json_array_skips_crash_truncated_line(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    json_path = tmp_path / "records.json"
    jsonl_path.write_text(
        '{"id": 1}\n{"id": 2}\n{"id":',
        encoding="utf-8",
    )

    jsonl_to_json_array(jsonl_path, json_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {"id": 1},
        {"id": 2},
    ]


def test_jsonl_to_json_array_keeps_latest_active_record(tmp_path: Path):
    jsonl_path = tmp_path / "records.jsonl"
    json_path = tmp_path / "records.json"
    for record in (
        {"id": "a", "status": "failed"},
        {"id": "b", "status": "ok"},
        {"id": "a", "status": "ok"},
        {"id": "old", "status": "ok"},
    ):
        atomic_append_jsonl(jsonl_path, record)

    jsonl_to_json_array(
        jsonl_path,
        json_path,
        record_key=lambda row: row["id"],
        include_keys={"a", "b"},
    )

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert {row["id"]: row["status"] for row in rows} == {
        "a": "ok",
        "b": "ok",
    }


def test_jsonl_resume_survives_truncated_multibyte_character(tmp_path):
    from core.io import iter_jsonl_records
    source = tmp_path / 'checkpoint.jsonl'
    source.write_bytes(b'{"id": 1}\n{"title": "\xc3')
    atomic_append_jsonl(source, {'id': 2})
    assert list(iter_jsonl_records(source)) == [{'id': 1}, {'id': 2}]
    target = tmp_path / 'final.json'
    jsonl_to_json_array(source, target)
    assert json.loads(target.read_text()) == [{'id': 1}, {'id': 2}]
