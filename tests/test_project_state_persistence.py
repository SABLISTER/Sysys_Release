"""Tests for Search-tab project state persistence."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QSettings

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.models import HypothesisComponent
from core.config import Config
from gui.main_window import MainWindow
from gui.project_manager import ProjectManager


@pytest.fixture(autouse=True)
def preserve_recent_projects_setting():
    settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
    previous = settings.value("projects/recent", []) or []
    yield
    settings.setValue("projects/recent", previous)


def test_project_state_persists_components_and_weighted_boolean_query(tmp_path):
    components = [
        HypothesisComponent(
            id="term_autism",
            label="autism",
            description="Search term from question: autism",
            keywords=["autism"],
            weight=1.75,
        ),
        HypothesisComponent(
            id="phrase_white_matter",
            label="white matter",
            description="Phrase from question: white matter",
            keywords=["white matter"],
            weight=0.9,
        ),
    ]
    window_state = {
        "search_query": "autism white matter",
        "weighted_boolean_query": 'autism AND ("white matter")',
        "hypothesis_components": ProjectManager.serialize_hypothesis_components(components),
    }

    ProjectManager.save_project(tmp_path, config=None, window_state=window_state)

    loaded = ProjectManager.load_project(tmp_path)["window_state"]
    restored = ProjectManager.deserialize_hypothesis_components(
        loaded["hypothesis_components"]
    )
    assert loaded["weighted_boolean_query"] == 'autism AND ("white matter")'
    assert [component.id for component in restored] == ["term_autism", "phrase_white_matter"]
    assert restored[0].weight == 1.75
    assert restored[1].keywords == ["white matter"]


def test_config_project_root_dot_resolves_to_config_folder(tmp_path):
    config_path = tmp_path / "research_config.yaml"
    config_path.write_text("project_root: .\nllm_provider: ollama\n", encoding="utf-8")

    cfg = Config.load_config(config_path)

    assert cfg.project_root == tmp_path.resolve()
    assert cfg.data_dir == tmp_path.resolve() / "data"


def test_fulltext_controls_load_from_config_and_restore_from_project_state(tmp_path):
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(
        """{
  "project_root": ".",
  "fulltext_download_workers": 4,
  "fulltext_streaming": false,
  "fulltext_download_batch_size": 21,
  "paperqa_index_enabled": false,
  "paperqa_index_batch_size": 34
}
""",
        encoding="utf-8",
    )
    loaded = Config.load_config(config_path)

    assert loaded.fulltext_download_workers == 4
    assert loaded.fulltext_streaming is False
    assert loaded.fulltext_download_batch_size == 21
    assert loaded.paperqa_index_enabled is False
    assert loaded.paperqa_index_batch_size == 34

    ProjectManager.save_project(tmp_path, loaded, window_state={})
    state = ProjectManager.load_project(tmp_path)

    class _Window:
        pipeline_config = Config(project_root=tmp_path)

    window = _Window()
    assert MainWindow._restore_fulltext_project_config(cast(MainWindow, window), state) is True
    assert window.pipeline_config.fulltext_download_workers == 4
    assert window.pipeline_config.fulltext_streaming is False
    assert window.pipeline_config.fulltext_download_batch_size == 21
    assert window.pipeline_config.paperqa_index_enabled is False
    assert window.pipeline_config.paperqa_index_batch_size == 34


def test_project_backup_restore_round_trip(tmp_path):
    project = tmp_path / "review_project"
    ProjectManager._create_skeleton(project)
    (project / "data" / "corpus" / "corpus.json").write_text(
        '[{"title": "Paper"}]',
        encoding="utf-8",
    )

    backup = ProjectManager.backup_project(project, tmp_path / "drive_backups")
    restored = ProjectManager.restore_project(backup, tmp_path / "restored_project")

    assert backup.suffix == ".zip"
    assert (restored / "research_config.yaml").exists()
    assert (restored / "data" / "corpus" / "corpus.json").exists()
    assert ProjectManager.validate_project(restored)["valid"]


def test_incremental_auto_backup_copies_only_changed_essential_files(tmp_path):
    project = tmp_path / "review_project"
    ProjectManager._create_skeleton(project)
    corpus_path = project / "data" / "corpus" / "corpus.json"
    corpus_path.write_text('[{"title": "Paper"}]', encoding="utf-8")
    (project / "data" / "fulltext_corpus" / "source_archive").mkdir(parents=True)
    (project / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf").write_bytes(b"pdf")

    destination = tmp_path / "drive"
    first = ProjectManager.sync_project_backup(project, destination)
    second = ProjectManager.sync_project_backup(project, destination)
    corpus_path.write_text('[{"title": "Updated Paper"}]', encoding="utf-8")
    third = ProjectManager.sync_project_backup(project, destination)
    backup_root = Path(first["backup_root"])

    assert first["copied"] >= 2
    assert second["copied"] == 0
    assert third["copied"] == 1
    assert (backup_root / "research_config.yaml").exists()
    assert (backup_root / "data" / "corpus" / "corpus.json").exists()
    assert (backup_root / "data" / "corpus" / "corpus.json").read_text(
        encoding="utf-8"
    ) == '[{"title": "Updated Paper"}]'
    assert not (backup_root / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf").exists()


def test_auto_backup_restore_from_synced_root_without_fulltext_payloads(tmp_path):
    project = tmp_path / "review_project"
    ProjectManager._create_skeleton(project)
    (project / "data" / "corpus" / "corpus.json").write_text(
        '[{"title": "Paper"}]',
        encoding="utf-8",
    )
    pdf_path = project / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"pdf")

    drive_root = tmp_path / "drive" / ProjectManager.AUTO_BACKUP_FOLDER_NAME
    result = ProjectManager.sync_project_backup(project, drive_root)
    restored = ProjectManager.restore_project(drive_root, tmp_path / "restored_project")

    assert ProjectManager.resolve_backup_source(drive_root) == Path(result["backup_root"])
    assert (restored / "research_config.yaml").exists()
    assert (restored / "data" / "corpus" / "corpus.json").exists()
    assert not (restored / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf").exists()
    assert ProjectManager.validate_project(restored)["valid"]


def test_incremental_auto_backup_can_include_fulltext(tmp_path):
    project = tmp_path / "review_project"
    ProjectManager._create_skeleton(project)
    pdf_path = project / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"pdf")

    result = ProjectManager.sync_project_backup(
        project,
        tmp_path / "drive",
        include_fulltext=True,
    )

    backup_root = Path(result["backup_root"])
    assert (backup_root / "data" / "fulltext_corpus" / "source_archive" / "paper.pdf").exists()


def test_backups_reject_project_folder_as_destination(tmp_path):
    project = tmp_path / "review_project"
    ProjectManager._create_skeleton(project)

    with pytest.raises(ValueError, match="project folder"):
        ProjectManager.backup_project(project, project)

    with pytest.raises(ValueError, match="project folder"):
        ProjectManager.sync_project_backup(project, project)
