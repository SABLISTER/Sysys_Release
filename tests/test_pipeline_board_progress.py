"""Tests for pipeline-board disk summaries and snapshot merging."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QApplication

from core.config import Config
from gui.pipeline_board import (
    PENDING_SUMMARY,
    PipelineBoardDialog,
    collect_stage_progress,
    empty_stage_progress,
    merge_disk_progress,
    normalize_status,
)


def _cfg(tmp_path: Path) -> Config:
    return Config(project_root=tmp_path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_normalize_status_maps_error_to_failed():
    assert normalize_status("error") == "failed"
    assert normalize_status("done") == "done"
    assert normalize_status("unknown") == "pending"


def test_collect_stage_progress_empty_config_is_all_pending(tmp_path):
    progress = collect_stage_progress(_cfg(tmp_path))
    assert len(progress) == 11
    for num in range(1, 12):
        assert progress[num]["status"] == "pending"
        assert progress[num]["summary"] == PENDING_SUMMARY


def test_collect_stage_progress_corpus_only(tmp_path):
    cfg = _cfg(tmp_path)
    _write_json(
        cfg.corpus_dir / "corpus.json",
        [{"title": "Paper A"}, {"title": "Paper B"}],
    )
    progress = collect_stage_progress(cfg)
    assert progress[1]["status"] == "done"
    assert progress[1]["summary"] == "2 papers"
    for num in range(2, 12):
        assert progress[num]["status"] == "pending"
        assert progress[num]["summary"] == PENDING_SUMMARY


def test_collect_stage_progress_score_extract_audit(tmp_path):
    cfg = _cfg(tmp_path)
    _write_json(cfg.corpus_dir / "corpus.json", [{"title": "A"}] * 10)
    _write_json(cfg.claims_dir / "relevant.json", [{"title": "A"}] * 4)
    _write_json(cfg.claims_dir / "scored_all.json", [{"title": "A"}] * 10)
    _write_json(cfg.claims_dir / "claims.json", [{"title": "A"}] * 4)
    _write_json(cfg.claims_dir / "claims_filtered.json", [{"title": "A"}] * 3)

    progress = collect_stage_progress(cfg)
    assert progress[1]["summary"] == "10 papers"
    assert progress[2]["status"] == "done"
    assert progress[2]["summary"] == "4 relevant of 10 scored"
    assert progress[3]["status"] == "done"
    assert progress[3]["summary"] == "4 papers extracted"
    assert progress[5]["status"] == "done"
    assert progress[5]["summary"] == "3 kept"
    assert progress[4]["status"] == "pending"


def test_collect_stage_progress_unreadable_file_stays_pending(tmp_path):
    cfg = _cfg(tmp_path)
    path = cfg.corpus_dir / "corpus.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    progress = collect_stage_progress(cfg)
    assert progress[1]["status"] == "pending"
    assert progress[1]["summary"] == PENDING_SUMMARY


def test_merge_disk_progress_skips_running_stage():
    current = empty_stage_progress()
    current[2] = {"status": "running", "summary": "In progress…"}
    disk = empty_stage_progress()
    disk[1] = {"status": "done", "summary": "12 papers"}
    disk[2] = {"status": "done", "summary": "4 relevant papers"}
    merged = merge_disk_progress(current, disk)
    assert merged[1]["status"] == "done"
    assert merged[1]["summary"] == "12 papers"
    assert merged[2]["status"] == "running"
    assert merged[2]["summary"] == "In progress…"
    assert merged[3]["status"] == "pending"


def test_collect_stage_progress_validate_and_synthesize(tmp_path):
    cfg = _cfg(tmp_path)
    _write_json(
        cfg.data_dir / "method_review" / "method_profiles_validated.json",
        [
            {"method_profile": {"needs_manual_review": False}},
            {"method_profile": {"needs_manual_review": True}},
        ],
    )
    _write_json(cfg.clusters_dir / "clusters.json", [{"label": "A"}, {"label": "B"}])
    (cfg.clusters_dir / "abstract_draft.md").write_text("Draft.", encoding="utf-8")
    progress = collect_stage_progress(cfg)
    assert progress[4]["status"] == "done"
    assert progress[4]["summary"] == "2 profiles, 1 flagged"
    assert progress[6]["status"] == "done"
    assert progress[6]["summary"] == "2 clusters; abstract saved"


def test_apply_progress_skips_running_row():
    _app = QApplication.instance() or QApplication([])
    dialog = PipelineBoardDialog()
    dialog.update_stage(3, "running", summary="In progress…")
    dialog.apply_progress({
        1: {"status": "done", "summary": "9 papers"},
        3: {"status": "done", "summary": "4 papers extracted"},
    })
    assert dialog._stage_data[1]["status"] == "done"
    assert dialog._stage_data[1]["summary"] == "9 papers"
    assert dialog._stage_data[3]["status"] == "running"
    assert dialog._stage_data[3]["summary"] == "In progress…"
    dialog.close()
