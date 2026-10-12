from __future__ import annotations

import json
from pathlib import Path

from core.checkpoint import CheckpointManager


def test_checkpoint_manager_loads_current_format(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps({"processed_count": 2, "total": 3, "data": [{"a": 1}, {"b": 2}]}),
        encoding="utf-8",
    )

    data, index = CheckpointManager(tmp_path, "checkpoint.json").load()

    assert data == [{"a": 1}, {"b": 2}]
    assert index == 2


def test_checkpoint_manager_loads_legacy_extraction_format(tmp_path: Path):
    path = tmp_path / "extract_checkpoint.json"
    path.write_text(
        json.dumps({"extracted": [{"title": "A"}], "next_index": 1}),
        encoding="utf-8",
    )

    data, index = CheckpointManager(tmp_path, "extract_checkpoint.json").load()

    assert data == [{"title": "A"}]
    assert index == 1


def test_checkpoint_manager_loads_legacy_list_format(tmp_path: Path):
    path = tmp_path / "interrater_checkpoint.json"
    path.write_text(json.dumps([{"title": "A"}, {"title": "B"}]), encoding="utf-8")

    data, index = CheckpointManager(tmp_path, "interrater_checkpoint.json").load()

    assert data == [{"title": "A"}, {"title": "B"}]
    assert index == 2


def test_checkpoint_manager_rejects_unknown_format(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps("bad"), encoding="utf-8")

    data, index = CheckpointManager(tmp_path, "checkpoint.json").load()

    assert data == []
    assert index == 0


def test_checkpoint_manager_force_save_creates_directory(tmp_path: Path):
    checkpoint_dir = tmp_path / "nested" / "checkpoints"
    ckpt = CheckpointManager(checkpoint_dir, "checkpoint.json")

    assert ckpt.force_save([{"title": "A"}], 2)
    assert (checkpoint_dir / "checkpoint.json").exists()
