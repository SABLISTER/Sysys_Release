"""Regression tests for Python progress payloads crossing Qt signals."""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from PySide6.QtCore import QCoreApplication

from gui import stage_runner
from gui.main_window import MainWindow
from gui.workers import PipelineStageWorker, _progress_message


def test_progress_message_extracts_text_from_structured_payload():
    payload = {
        "message": "Checking corpus open-access availability: 1/1 checked",
        "current": 1,
        "total": 1,
    }

    assert _progress_message(payload) == payload["message"]
    assert _progress_message("Downloading PDF") == "Downloading PDF"
    assert _progress_message({"message": 0}) == "0"
    assert _progress_message({"message": None}) == ""
    assert _progress_message(None) == ""


def test_pipeline_worker_emits_structured_progress_as_text(monkeypatch, capfd):
    async def stage(config, cancel_event=None, progress_callback=None, **kwargs):
        assert progress_callback is not None
        progress_callback({
            "message": "Processing downloaded PDF 1/1",
            "current": 1,
            "total": 1,
        })
        return {"processed": 1}

    monkeypatch.setitem(
        stage_runner._STAGE_REGISTRY,
        "structured_progress_test",
        (stage, True),
    )

    app = QCoreApplication.instance() or QCoreApplication([])
    worker = PipelineStageWorker(stage_name="structured_progress_test")
    messages: list[str] = []
    counts: list[tuple[int, int]] = []
    worker.progress.connect(messages.append)
    worker.progress_counts.connect(
        lambda current, total: counts.append((current, total))
    )

    worker.run()
    app.processEvents()

    assert messages == ["Processing downloaded PDF 1/1"]
    assert counts == [(1, 1)]
    assert "_pythonToCppCopy" not in capfd.readouterr().err


def test_search_count_handler_updates_progress_bar():
    class ProgressBar:
        def __init__(self):
            self.range = None
            self.value = None

        def setRange(self, minimum, maximum):
            self.range = (minimum, maximum)

        def setValue(self, value):
            self.value = value

    progress_bar = ProgressBar()
    window = SimpleNamespace(search_progress=progress_bar)

    MainWindow._on_sysys_search_counts(cast(MainWindow, window), 3, 10)

    assert progress_bar.range == (0, 10)
    assert progress_bar.value == 3
