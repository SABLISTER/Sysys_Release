"""GUI contract tests for editable human relevance review rows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
)

from gui.main_window import MainWindow


def _cell(table: QTableWidget, row: int, col: int) -> QTableWidgetItem:
    """Return the item at (row, col), asserting the caller's invariant that
    the table was already populated so the cell is never empty."""
    item = table.item(row, col)
    assert item is not None
    return item


class _ReviewTableHarness:
    def __init__(self):
        self.human_review_table = QTableWidget(0, 7)
        self.human_review_edit_cells = QCheckBox()
        self.human_review_status = QLabel()
        self._human_review_rows: list[dict] = []
        self._human_review_artifact: dict | None = None
        self._loaded_human_review_path = None
        self.scored_path: Path | None = None

    _set_human_relevance_editing = MainWindow._set_human_relevance_editing
    _populate_human_relevance_table = MainWindow._populate_human_relevance_table
    _collect_human_relevance_rows = MainWindow._collect_human_relevance_rows
    _load_json_list = MainWindow._load_json_list
    _load_existing_human_relevance_review = (
        MainWindow._load_existing_human_relevance_review
    )
    _refresh_human_relevance_model_scores = (
        MainWindow._refresh_human_relevance_model_scores
    )

    def _scored_relevance_path(self):
        return self.scored_path

    def _log(self, *_args, **_kwargs):
        pass


def test_human_review_cells_are_correctable_and_verification_is_collected():
    _app = QApplication.instance() or QApplication([])
    harness = _ReviewTableHarness()
    harness._populate_human_relevance_table([
        {
            "article_id": "old-id",
            "title": "Original title",
            "year": 2022,
            "abstract": "Original abstract",
            "model_score": 2,
            "human_score": 4,
            "human_verified": False,
        }
    ])

    assert not (
        _cell(harness.human_review_table, 0, 2).flags()
        & Qt.ItemFlag.ItemIsEditable
    )

    harness.human_review_edit_cells.setChecked(True)
    harness._set_human_relevance_editing(True)
    assert (
        _cell(harness.human_review_table, 0, 2).flags()
        & Qt.ItemFlag.ItemIsEditable
    )

    _cell(harness.human_review_table, 0, 1).setCheckState(Qt.CheckState.Checked)
    _cell(harness.human_review_table, 0, 2).setText("Corrected title")
    _cell(harness.human_review_table, 0, 3).setText("2024")
    _cell(harness.human_review_table, 0, 4).setText("4")
    _cell(harness.human_review_table, 0, 5).setText("corrected-id")
    _cell(harness.human_review_table, 0, 6).setText("Corrected abstract")

    row = harness._collect_human_relevance_rows()[0]

    assert row["human_verified"] is True
    assert row["title"] == "Corrected title"
    assert row["year"] == 2024
    assert row["model_score"] == 4
    assert row["model_score_human_corrected"] is True
    assert row["article_id"] == "corrected-id"
    assert row["abstract"] == "Corrected abstract"


def test_refresh_preserves_a_human_corrected_brt_score(tmp_path):
    _app = QApplication.instance() or QApplication([])
    scored_path = tmp_path / "scored_all.json"
    scored_path.write_text(
        json.dumps([{"id": "paper-a", "title": "A", "relevance_score": 1}]),
        encoding="utf-8",
    )
    harness = _ReviewTableHarness()
    harness.scored_path = scored_path
    harness._populate_human_relevance_table([
        {
            "article_id": "paper-a",
            "title": "A",
            "model_score": 4,
            "model_score_human_corrected": True,
            "human_verified": True,
        }
    ])

    harness._refresh_human_relevance_model_scores()

    assert harness._human_review_rows[0]["model_score"] == 4
    assert _cell(harness.human_review_table, 0, 4).text() == "4"


def test_saved_review_is_restored_for_the_current_project(tmp_path):
    _app = QApplication.instance() or QApplication([])
    review_path = tmp_path / "validation" / "human_relevance_review.json"
    review_path.parent.mkdir()
    review_path.write_text(
        json.dumps({
            "reviewed_count": 1,
            "verified_count": 1,
            "records": [{
                "article_id": "paper-a",
                "title": "A",
                "human_score": 5,
                "human_verified": True,
            }],
        }),
        encoding="utf-8",
    )
    harness = _ReviewTableHarness()

    harness._load_existing_human_relevance_review(
        SimpleNamespace(data_dir=tmp_path)
    )

    assert harness._human_review_artifact is not None
    assert harness._human_review_rows[0]["human_verified"] is True
    assert "1 human verified" in harness.human_review_status.text()
