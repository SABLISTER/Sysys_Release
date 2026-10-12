"""Tests for the SystS citation guide dialog."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from PySide6.QtWidgets import QApplication

from gui.dialogs import CITATION_GUIDE_TEXT, CitationGuideDialog


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_citation_guide_lists_meta_analysis_method_references():
    expected = [
        "Page MJ",
        "DerSimonian R, Laird N",
        "Egger M, Davey Smith G",
        "Duval S, Tweedie R",
        "Rosenthal R",
        "@article{dersimonian_laird_1986",
        "@software{systs_hypothesis_consensus_analyzer",
    ]

    for text in expected:
        assert text in CITATION_GUIDE_TEXT


def test_citation_guide_dialog_shows_copyable_text():
    _app()

    dialog = CitationGuideDialog()

    assert dialog.windowTitle() == "Citation Guide"
    assert dialog.citation_text.toPlainText() == CITATION_GUIDE_TEXT
    assert dialog.citation_text.isReadOnly()
