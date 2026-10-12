"""Real GUI controls must hand the selected LM Studio settings to each worker."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_lmstudio_gui_stage_controls(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)], cwd=tmp_path,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(repo),
                 MPLCONFIGDIR=str(tmp_path / "mpl"), XDG_CACHE_HOME=str(tmp_path / "cache")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise(root):
    import json
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from core.config import Config
    from gui.main_window import MainWindow
    from gui.workers import PipelineStageWorker

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / "settings"))
    MainWindow._show_welcome_if_first_run = lambda self: None
    MainWindow._maybe_prompt_auto_backup = lambda self, reason: None
    app = QApplication([])
    window = MainWindow()
    cfg = Config(project_root=root, llm_provider="custom", custom_api_format="lmstudio", custom_model="global-model",
                 custom_base_url="http://lmstudio.test:12345", custom_api_key="gui-key")
    cfg.ensure_dirs()
    selected = root / "selected.json"
    selected.write_text(json.dumps([{"title": "Selected paper", "abstract": "Trial evidence"}]))
    window.pipeline_config = cfg
    window._sync_settings_to_tabs(force=True)
    window._apply_score_advanced_mode(True)
    # The parent window/tab is not visible. That must not switch scoring to BERT.
    assert not window.score_advanced_box.isVisible()
    assert not window.score_advanced_box.isHidden()
    window.stat_extract_llm_context.setChecked(True)
    started = []
    # Inspect actual constructed workers; stage HTTP execution has separate tests.
    PipelineStageWorker.start = lambda self: started.append(self)
    try:
        for prefix, launcher, field, stage in [
            ("score", "_start_sysys_score", "score_corpus_path", "score"),
            ("stat_extract", "_start_statistical_extract", "stat_extract_input", "statistical_extract"),
            ("stat_extract", "_start_statistical_extract", "stat_extract_input", "statistical_extract"),
            ("extract", "_start_extract", "extract_corpus", "extract"),
            ("validate", "_start_validate", "validate_claims", "validate"),
            ("synth", "_start_synthesize", "synth_input", "synthesize"),
            ("interrater", "_start_interrater", "interrater_corpus", "interrater"),
        ]:
            assert getattr(window, prefix + "_provider").currentText() == "custom"
            getattr(window, prefix + "_model").setText(prefix + "-model")
            getattr(window, field).setText(str(selected))
            before = len(started)
            window.stat_extract_full_llm.setChecked(before == 2)
            getattr(window, launcher)()
            assert len(started) == before + 1, launcher
            worker = started[-1]
            assert worker.stage_name == stage
            assert worker.config.llm_provider == "custom"
            # Stage 8 keeps the primary's config and sends the second rater
            # separately, so agreement isn't mislabeled as self-consistency.
            assert worker.config.llm_model == ("score-model" if stage == "interrater" else prefix + "-model")
            assert worker.config.custom_base_url == cfg.custom_base_url
            assert worker.config.custom_api_key == "gui-key"
            assert cfg.custom_model == "global-model"
            if stage == "score":
                assert worker.config.relevance_model == ""
                assert worker._kwargs["corpus_path"] == selected
            if stage == "statistical_extract":
                assert worker._kwargs["use_llm_context_check"] is True
                assert worker._kwargs["llm_full_statistical_extraction"] is (before == 2)
            if stage == "interrater":
                assert worker._kwargs["primary_model"] == "score-model"
                assert worker._kwargs["second_provider"] == "custom"
                assert worker._kwargs["second_model"] == "interrater-model"

        # This launcher must not replace an active worker/shared LLM config.
        class BusyWorker:
            def isRunning(self):
                return True

        busy = BusyWorker()
        window._pipeline_worker = busy
        before = len(started)
        window._start_statistical_extract()
        assert window._pipeline_worker is busy
        assert len(started) == before
    finally:
        window._pipeline_worker = None
        window.close()
        app.processEvents()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]))
