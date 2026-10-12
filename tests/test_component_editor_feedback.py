"""Tests for keyword-analysis feedback in the component editor dialog."""
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

from core.models import HypothesisComponent
from gui.dialogs import (
    ComponentEditor,
    _parse_keyword_analysis_metrics,
    _split_keyword_analysis_description,
)


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_split_keyword_analysis_description_removes_suffix_from_editable_text():
    base, suffix = _split_keyword_analysis_description(
        "Search term from question: autism | Keyword analysis: "
        "utility=0.530, coverage=0.80, relevance=0.17, confidence=0.11, direction=0.51"
    )

    assert base == "Search term from question: autism"
    assert suffix.startswith(" | Keyword analysis:")
    assert "utility=0.530" in suffix


def test_parse_keyword_analysis_metrics_extracts_display_values():
    metrics = _parse_keyword_analysis_metrics(
        " | Keyword analysis: utility=0.530, coverage=0.80, "
        "relevance=0.17, confidence=0.11, direction=-0.51"
    )

    assert metrics == {
        "utility": "0.530",
        "coverage": "0.80",
        "relevance": "0.17",
        "confidence": "0.11",
        "direction": "-0.51",
    }


def test_component_editor_save_preserves_keyword_analysis_suffix():
    _app()
    component = HypothesisComponent(
        id="term_autism",
        label="autism",
        description=(
            "Search term from question: autism | Keyword analysis: "
            "utility=0.530, coverage=0.80, relevance=0.17, confidence=0.11, direction=0.51"
        ),
        keywords=["autism"],
        weight=1.295,
    )
    dialog = ComponentEditor(component)
    dialog.desc_edit.setPlainText("Updated description")
    dialog.keywords_edit.setPlainText("autism, cortex")

    dialog._save()

    assert component.description.startswith("Updated description | Keyword analysis:")
    assert "utility=0.530" in component.description
    assert component.keywords == ["autism", "cortex"]
