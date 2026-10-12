"""Real Qt worker/project round trip; only the HTTP boundary is replaced."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_stage7_interruption_reopen_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(root),
               MPLCONFIGDIR=str(tmp_path / "matplotlib"),
               XDG_CACHE_HOME=str(tmp_path / "cache"),
               XDG_CONFIG_HOME=str(tmp_path / "settings"))
    # Generous timeout: the subprocess builds two full MainWindows and runs
    # two real stage-7 worker passes; under parallel-suite load 45s was flaky.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_contract(root):
    import json
    import threading
    import time
    from collections import Counter

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from core.io import atomic_append_jsonl
    from gui.main_window import MainWindow
    from gui.project_manager import ProjectManager
    from stages import s7_fulltext

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(root / "settings"))
    # Suppress first-launch modal questions, preserving real UI construction.
    MainWindow._show_welcome_if_first_run = lambda self: None
    MainWindow._maybe_prompt_auto_backup = lambda self, reason: None
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    reached_second = threading.Event()
    release_second = threading.Event()
    downloads = []
    lookups = []

    class Response:
        status_code = 200
        headers = {"content-type": "text/html"}
        content = (b"<html><body>Trial results: N = 120. Cohen's d = 0.85, "
                   b"95% CI [0.45, 1.25]. p < 0.001.</body></html>")

        def __init__(self, doi):
            self.doi = doi

        def raise_for_status(self):
            pass

        def json(self):
            return {"is_oa": True, "best_oa_location": {
                "url_for_landing_page": f"https://example.test/{self.doi}"}}

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, **kwargs):
            if "api.unpaywall.org" in url:
                doi = url.split("/v2/", 1)[1].split("?", 1)[0]
                lookups.append(doi)
            else:
                doi = url.removeprefix("https://example.test/")
                downloads.append(doi)
                if doi == "10.1234/b" and downloads.count(doi) == 1:
                    reached_second.set()
                    assert release_second.wait(10), "GUI never released HTTP request"
            return Response(doi)

    class Requests:
        pass

    Requests.Session = Session
    s7_fulltext._import_requests = lambda: Requests

    def wait_until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            app.processEvents()
            assert time.monotonic() < deadline, "Timed out waiting for Qt worker"
            time.sleep(0.005)
        app.processEvents()

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    project = root / "review"
    ProjectManager._create_skeleton(project)
    claims = project / "data" / "claims" / "selected.json"
    claims.write_text(json.dumps([
        {"doi": f"10.1234/{letter}", "title": f"Trial {letter}",
         "methodology": "randomized controlled trial"}
        for letter in ("a", "b", "c", "a")
    ]))
    ft = project / "data" / "fulltext"
    win = MainWindow()
    windows = [win]
    workers = []
    errors, completed, cancelled, progress = [], [], [], []
    try:
        win._open_project_path(project)
        win.sysys_ft_corpus.setText(str(claims))
        win.sysys_ft_workers.setValue(2)
        win.sysys_ft_batch_size.setValue(7)
        win.sysys_ft_paperqa_enabled.setChecked(False)
        win.sysys_ft_paperqa_batch_size.setValue(3)
        win.sysys_ft_resume.setChecked(True)
        win.sysys_nl_query.setText("Stage 7 contract question")
        win.sysys_ft_btn.click()
        worker = win._pipeline_worker
        workers.append(worker)
        worker.error.connect(lambda *args: errors.append(args))
        worker.finished.connect(lambda *args: completed.append(args))
        worker.cancelled.connect(lambda: cancelled.append(True))
        worker.progress.connect(progress.append)
        wait_until(reached_second.is_set)
        assert not win.sysys_ft_btn.isEnabled()
        assert win.sysys_ft_cancel_btn.isEnabled()
        log_path = ft / "download_log.jsonl"
        metrics_path = ft / "auto_extracted_metrics.jsonl"
        before = [p.read_bytes() for p in (log_path, metrics_path)]
        assert [r["doi"] for r in rows(log_path)] == ["10.1234/a"]
        assert rows(metrics_path)[0]["metrics"]["sample_sizes"][0]["value"] == 120
        win.sysys_ft_cancel_btn.click()
        release_second.set()
        wait_until(lambda: not worker.isRunning() and bool(cancelled))
        assert not errors and not completed
        assert [p.read_bytes() for p in (log_path, metrics_path)] == before
        assert not (ft / "download_log.json").exists()
        assert not (ft / "auto_extracted_metrics.json").exists()
        assert win.sysys_ft_btn.isEnabled()
        assert not win.sysys_ft_cancel_btn.isEnabled()
        win.close()
        state = ProjectManager.load_project(project)
        assert state["window_state"]["stage7"] == {
            "claims_path": str(claims), "workers": 2, "resume": True}

        # Duplicate records, a stale paper from another selection, and a torn
        # trailing write must never leak into the active run's final outputs.
        for path in (log_path, metrics_path):
            record = rows(path)[0]
            atomic_append_jsonl(path, record)
            atomic_append_jsonl(path, {**record, "doi": "10.1234/stale"})
            with path.open("a") as handle:
                handle.write('{"doi":')

        reopened = MainWindow()
        windows.append(reopened)
        reopened.sysys_ft_resume.setChecked(False)
        reopened._open_project_path(project)
        assert reopened.pipeline_config.project_root == project
        assert reopened.pipeline_config.data_dir == project / "data"
        assert reopened.sysys_nl_query.text() == "Stage 7 contract question"
        assert reopened.sysys_ft_corpus.text() == str(claims)
        assert reopened.sysys_ft_workers.value() == 2
        assert reopened.sysys_ft_streaming.isChecked()
        assert reopened.sysys_ft_batch_size.value() == 7
        assert not reopened.sysys_ft_paperqa_enabled.isChecked()
        assert reopened.sysys_ft_paperqa_batch_size.value() == 3
        assert reopened.pipeline_config.fulltext_download_batch_size == 7
        assert reopened.pipeline_config.paperqa_index_enabled is False
        assert reopened.sysys_ft_resume.isChecked()
        assert reopened.sysys_ft_btn.isEnabled()
        assert not reopened.sysys_ft_cancel_btn.isEnabled()
        reopened.sysys_ft_btn.click()
        worker = reopened._pipeline_worker
        workers.append(worker)
        worker.error.connect(lambda *args: errors.append(args))
        worker.finished.connect(lambda *args: completed.append(args))
        wait_until(lambda: not worker.isRunning() and bool(completed or errors))
        assert not errors
        assert Counter(downloads) == {"10.1234/a": 1, "10.1234/b": 2, "10.1234/c": 1}
        assert Counter(lookups) == {f"10.1234/{x}": 1 for x in "abc"}
        assert completed[0][1]["downloaded"] == 3
        assert completed[0][1]["metrics_extracted"] == 3
        for name in ("download_log", "auto_extracted_metrics"):
            final = json.loads((ft / f"{name}.json").read_text())
            assert Counter(r["doi"] for r in final) == {f"10.1234/{x}": 1 for x in "abc"}
        assert reopened.sysys_ft_btn.isEnabled()
        assert not reopened.sysys_ft_cancel_btn.isEnabled()
        assert progress
    finally:
        release_second.set()
        for worker in workers:
            worker.cancel()
            assert worker.wait(10000)
        for window in windows:
            window.close()
        app.processEvents()


if __name__ == "__main__":
    _exercise_contract(Path(sys.argv[1]))
