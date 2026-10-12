"""Tests for the Run-All-Stages queue and pipeline worker guard logic."""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from gui.main_window import MainWindow


class _FakeWorker:
    def __init__(self, running: bool = False):
        self._running = running

    def isRunning(self) -> bool:
        return self._running


class DummyWindow:
    def __init__(self):
        self._logs: list[tuple[str, str]] = []
        self._pipeline_worker: _FakeWorker | None = None
        self._search_worker: _FakeWorker | None = None
        self._run_all_queue: list[str] = []
        self._run_all_active: bool = False
        self._pipeline_progress: dict = {}
        self._pipeline_board = SimpleNamespace(
            update_stage=lambda num, status, elapsed_sec=None, summary=None: None,
            apply_progress=lambda progress: None,
        )
        self._started: list[str] = []
        # Bind the MainWindow state-machine helpers to this dummy instance
        self._run_all_stages_advance = MainWindow._run_all_stages_advance.__get__(
            self, DummyWindow
        )
        self._set_pipeline_stage = MainWindow._set_pipeline_stage.__get__(
            self, DummyWindow
        )

    def _log(self, msg: str, level: str = "INFO") -> None:
        self._logs.append((level, msg))

    def _start_sysys_search(self) -> None:
        self._started.append("search")
        self._search_worker = _FakeWorker(running=True)

    def _start_sysys_score(self) -> None:
        self._started.append("score")
        self._pipeline_worker = _FakeWorker(running=True)

    def _start_extract(self) -> None:
        self._started.append("extract")

    def _start_validate(self) -> None:
        self._started.append("validate")

    def _start_audit(self) -> None:
        self._started.append("audit")

    def _start_synthesize(self) -> None:
        self._started.append("synthesize")

    def _start_sysys_fulltext(self) -> None:
        self._started.append("fulltext")

    def _start_interrater(self) -> None:
        self._started.append("interrater")

    def _start_robustness(self) -> None:
        self._started.append("robustness")

    def _start_verify(self) -> None:
        self._started.append("verify")


def test_pipeline_stage_guard_blocks_when_pipeline_worker_running():
    win = DummyWindow()
    win._pipeline_worker = _FakeWorker(running=True)
    assert MainWindow._pipeline_stage_guard(cast(MainWindow, win), "Stage X") is False
    assert any("another pipeline stage is running" in m for _, m in win._logs)


def test_pipeline_stage_guard_blocks_when_search_worker_running():
    win = DummyWindow()
    win._search_worker = _FakeWorker(running=True)
    assert MainWindow._pipeline_stage_guard(cast(MainWindow, win), "Stage X") is False
    assert any("Stage 1 search is still running" in m for _, m in win._logs)


def test_pipeline_stage_guard_allows_when_no_worker_running():
    win = DummyWindow()
    assert MainWindow._pipeline_stage_guard(cast(MainWindow, win), "Stage X") is True


def test_run_all_stages_initializes_queue_and_starts_search():
    win = DummyWindow()
    MainWindow._run_all_stages(cast(MainWindow, win))
    assert win._run_all_active is True
    assert win._run_all_queue == [
        "score", "extract", "validate", "audit",
        "synthesize", "fulltext", "interrater", "robustness", "verify",
    ]
    assert win._started == ["search"]


def test_run_all_stages_advance_calls_next_stage_and_pops_queue():
    win = DummyWindow()
    win._run_all_active = True
    win._run_all_queue = ["score", "extract"]
    MainWindow._run_all_stages_advance(cast(MainWindow, win))
    assert win._started == ["score"]
    assert win._run_all_queue == ["extract"]
    assert win._run_all_active is True


def test_run_all_stages_advance_finishes_when_queue_empty():
    win = DummyWindow()
    win._run_all_active = True
    win._run_all_queue = []
    MainWindow._run_all_stages_advance(cast(MainWindow, win))
    assert win._started == []
    assert win._run_all_active is False


def test_run_all_stages_advance_aborts_when_start_does_not_run_worker():
    """A stage that fails to start a worker should abort the chain."""
    win = DummyWindow()
    win._run_all_active = True
    win._run_all_queue = ["extract"]
    # Remove _start_extract so stage_start_map returns None? Actually the mapping
    # is static; easier to make the start method a no-op that leaves worker None.
    win._start_extract = lambda: None  # type: ignore[method-assign]
    MainWindow._run_all_stages_advance(cast(MainWindow, win))
    assert win._run_all_active is False
    assert win._run_all_queue == []
