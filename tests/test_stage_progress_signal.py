"""Regression tests for the PipelineStageWorker.progress signal boundary.

Stages such as ``s2b_fulltext`` and ``s7_fulltext.check_oa_status`` report
progress as a structured dict (``{"message", "current", "total"}``).  Those
payloads have to survive the Qt signal hop into the GUI thread; a mismatched
signal signature drops them silently (shiboken prints ``_pythonToCppCopy``
to stderr and delivers nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from PySide6.QtWidgets import QApplication

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from gui import stage_runner
from gui.workers import PipelineStageWorker
from stages.s2b_fulltext import _emit_progress


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_dict_progress_payload_survives_the_signal_hop(monkeypatch, capfd):
    async def stage(config, cancel_event=None, progress_callback=None, **kwargs):
        _emit_progress(
            progress_callback,
            "Downloading 7/42: Smith et al.",
            7,
            42,
        )
        return {}

    monkeypatch.setitem(
        stage_runner._STAGE_REGISTRY,
        "structured_progress_signal_test",
        (stage, True),
    )

    app = _app()
    worker = PipelineStageWorker(stage_name="structured_progress_signal_test")
    received: list[str] = []
    counts: list[tuple[int, int]] = []
    worker.progress.connect(received.append)
    worker.progress_counts.connect(
        lambda current, total: counts.append((current, total))
    )

    worker.run()
    app.processEvents()

    assert received, "structured progress payload was dropped at the signal boundary"
    assert received == ["Downloading 7/42: Smith et al."]
    assert counts == [(7, 42)]
    assert "_pythonToCppCopy" not in capfd.readouterr().err


def test_string_progress_payload_still_works():
    _app()
    worker = PipelineStageWorker(stage_name="corpus_fulltext")
    received: list = []
    worker.progress.connect(received.append)

    worker.progress.emit("Phase 1: Identifying high-value targets...")

    assert received == ["Phase 1: Identifying high-value targets..."]
