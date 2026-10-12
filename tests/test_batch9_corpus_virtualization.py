"""Tests for the ninth fix batch: corpus browser virtualization.

The browser previously created a ``QTableWidgetItem`` for every cell of the
corpus (7 per paper) and re-created all of them on every stage-relevance
refresh. It now uses a ``QTableView`` over ``CorpusTableModel`` +
``CorpusFilterProxyModel``: one tuple of display strings per paper, with
delegates instantiated only for visible rows — and numerically correct
sorting for the #, Year, and Score columns, which string sorting got wrong.
"""
from __future__ import annotations

import csv
import os
from typing import cast
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTableView,
)

from gui.corpus_table_model import CorpusFilterProxyModel, CorpusTableModel
from gui.main_window import MainWindow


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


class _BrowserStub:
    """Just the corpus-browser slice of MainWindow, borrowing its methods."""

    _populate_corpus_browser = MainWindow._populate_corpus_browser
    _filter_corpus_table = MainWindow._filter_corpus_table
    _corpus_paper_for_proxy_row = MainWindow._corpus_paper_for_proxy_row
    _show_corpus_abstract = MainWindow._show_corpus_abstract
    _open_corpus_doi = MainWindow._open_corpus_doi
    _export_corpus_visible = MainWindow._export_corpus_visible
    _corpus_paper_key = MainWindow._corpus_paper_key
    _set_corpus_stage_relevance = MainWindow._set_corpus_stage_relevance

    def __init__(self):
        _app()
        self._corpus_model = CorpusTableModel()
        self._corpus_proxy = CorpusFilterProxyModel()
        self._corpus_proxy.setSourceModel(self._corpus_model)
        self._corpus_table = QTableView()
        self._corpus_table.setModel(self._corpus_proxy)
        self._corpus_table.setSortingEnabled(True)
        self._corpus_filter = QLineEdit()
        self._corpus_relevance_filter = QComboBox()
        self._corpus_relevance_filter.addItem("All papers", "all")
        self._corpus_relevance_filter.addItem("Stage 2 relevant", "stage2")
        self._corpus_abstract = QPlainTextEdit()
        self._corpus_data: list[dict] = []
        self._corpus_stage_relevance: dict[str, set[str]] = {}
        self._corpus_stage_records: dict[str, dict[str, dict]] = {}
        self.logged: list[str] = []

    def _log(self, message, *args, **kwargs):
        self.logged.append(str(message))


def _corpus(n: int = 12) -> list[dict]:
    return [
        {
            "title": f"Study {i}",
            "authors": [f"Author{i}A", f"Author{i}B", f"Author{i}C", "Extra"],
            "year": 2000 + i,
            "source": "pubmed",
            "doi": f"10.1000/paper{i}",
            "abstract": f"Abstract body {i}",
            "relevance_score": i / 10,
        }
        for i in range(n)
    ]


def test_model_rows_and_score_fallback():
    stub = _BrowserStub()
    papers = _corpus(3)
    del papers[1]["relevance_score"]  # falls back to the stage-2 record
    key = stub._corpus_paper_key(papers[1])
    stub._corpus_stage_records["stage2"] = {key: {"relevance_score": 0.77}}

    stub._populate_corpus_browser(papers)

    model = stub._corpus_model
    assert model.rowCount() == 3
    row0 = [model.index(0, c).data() for c in range(7)]
    assert row0 == ["1", "Study 0", "Author0A, Author0B, Author0C et al.",
                    "2000", "pubmed", "0.00", "10.1000/paper0"]
    assert model.index(1, 5).data() == "0.77"


def test_numeric_sorting_not_string_sorting():
    stub = _BrowserStub()
    stub._populate_corpus_browser(_corpus(105))

    stub._corpus_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    # String sorting would rank "99" above "105"; numeric sorting must not.
    assert stub._corpus_proxy.index(0, 0).data() == "105"

    stub._corpus_table.sortByColumn(5, Qt.SortOrder.DescendingOrder)
    assert stub._corpus_proxy.index(0, 1).data() == "Study 104"


def test_text_filter_matches_title_authors_year():
    stub = _BrowserStub()
    stub._populate_corpus_browser(_corpus(12))

    stub._filter_corpus_table("study 3")  # case-insensitive title match
    assert stub._corpus_proxy.rowCount() == 1

    stub._filter_corpus_table("Author7B")  # author match
    assert stub._corpus_proxy.rowCount() == 1

    stub._filter_corpus_table("2011")  # year match
    assert stub._corpus_proxy.rowCount() == 1

    stub._filter_corpus_table("")
    assert stub._corpus_proxy.rowCount() == 12


def test_stage_relevance_filter():
    stub = _BrowserStub()
    papers = _corpus(10)
    stub._populate_corpus_browser(papers)

    stub._set_corpus_stage_relevance("stage2", papers[:4])
    stub._corpus_relevance_filter.setCurrentIndex(1)
    stub._filter_corpus_table("")
    assert stub._corpus_proxy.rowCount() == 4

    stub._corpus_relevance_filter.setCurrentIndex(0)
    stub._filter_corpus_table("")
    assert stub._corpus_proxy.rowCount() == 10


def test_abstract_lookup_survives_sort_and_filter():
    stub = _BrowserStub()
    stub._populate_corpus_browser(_corpus(12))
    stub._corpus_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    stub._filter_corpus_table("study 1")  # Study 1, 10, 11

    index = stub._corpus_proxy.index(0, 0)  # highest # among the matches
    stub._show_corpus_abstract(index)
    text = stub._corpus_abstract.toPlainText()
    assert text.startswith("Study 11")
    assert "Abstract body 11" in text


def test_open_doi_uses_selected_visible_row():
    stub = _BrowserStub()
    stub._populate_corpus_browser(_corpus(5))
    stub._filter_corpus_table("study 3")
    stub._corpus_table.setCurrentIndex(stub._corpus_proxy.index(0, 0))

    with patch("webbrowser.open") as opened:
        stub._open_corpus_doi()
    opened.assert_called_once_with("https://doi.org/10.1000/paper3")


def test_export_visible_respects_filter(tmp_path):
    stub = _BrowserStub()
    stub._populate_corpus_browser(_corpus(12))
    stub._filter_corpus_table("study 3")

    out = tmp_path / "visible.csv"
    with patch("gui.main_window.QFileDialog.getSaveFileName",
               return_value=(str(out), "")):
        stub._export_corpus_visible()

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["#", "Title", "Authors", "Year", "Source", "Score", "DOI"]
    assert len(rows) == 2
    assert rows[1][1] == "Study 3"


@pytest.mark.parametrize("has_begin", [False, True])
def test_filter_without_complete_modern_qt_api(monkeypatch, has_begin):
    """Older supported Qt releases must update both filters and retain sorting."""
    if not has_begin:
        monkeypatch.setattr(CorpusFilterProxyModel, "beginFilterChange", None,
                            raising=False)
    monkeypatch.setattr(CorpusFilterProxyModel, "endFilterChange", None,
                        raising=False)
    stub = _BrowserStub()
    papers = _corpus(12)
    stub._populate_corpus_browser(papers)
    stub._corpus_table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    proxy = stub._corpus_proxy
    assert proxy.rowCount() == 12

    proxy.set_text_filter("Study 1")
    assert proxy.rowCount() == 3
    assert proxy.index(0, 1).data() == "Study 11"
    proxy.set_stage_keys({stub._corpus_paper_key(papers[1])})
    assert proxy.rowCount() == 1
    assert proxy.index(0, 1).data() == "Study 1"
    proxy.set_stage_keys(set())
    assert proxy.rowCount() == 0
    proxy.set_stage_keys(None)
    proxy.set_text_filter("")
    assert proxy.rowCount() == 12
    assert proxy.index(0, 1).data() == "Study 11"


def test_zero_scores_and_nullable_metadata_remain_visible_and_filterable():
    stub = _BrowserStub()
    papers = [{'title': 'Zero score', 'relevance_score': 0},
              {'title': None, 'source': None, 'doi': None, 'year': None}]
    stub._populate_corpus_browser(papers)
    assert stub._corpus_model.index(0, 5).data() == '0'
    assert stub._corpus_model.index(1, 1).data() == ''
    stub._filter_corpus_table('zero')
    assert stub._corpus_proxy.rowCount() == 1
    stub._corpus_proxy.sort(5, Qt.SortOrder.AscendingOrder)
    assert stub._corpus_proxy.index(0, 5).data() == '0'
