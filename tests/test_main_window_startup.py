"""Construct the real desktop window in an isolated process and project."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_main_window_constructs_and_processes_events(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        QT_QPA_PLATFORM="offscreen",
        PYTHONPATH=str(repo),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    code = textwrap.dedent("""
        from pathlib import Path
        from unittest.mock import patch
        from PySide6.QtCore import QCoreApplication, QEvent, QSettings
        from PySide6.QtWidgets import QApplication
        from core.config import Config
        from gui.main_window import MainWindow

        root = Path.cwd()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, str(root / 'settings'))

        # Exercise real config loading against a clean project, not user data.
        original_load = Config.load_config
        def load_test_config(cls, config_path=None, *, project_root=None):
            return original_load(config_path, project_root=project_root or root)

        app = QApplication([])
        # Use real functions: MagicMock replacements on this QObject subclass
        # crash Qt's dynamic slot registration before the first menu is built.
        with patch.object(Config, 'load_config', classmethod(load_test_config)), \
             patch.object(MainWindow, '_show_welcome_if_first_run', lambda self: None), \
             patch.object(MainWindow, '_maybe_prompt_auto_backup', lambda self, *a: None):
            window = MainWindow()

        assert window.tabs.count() >= 10
        assert window.menuBar().actions()
        # Use the real stage widgets so test doubles cannot hide renamed controls.
        window.pipeline_config.llm_provider = 'custom'
        window.pipeline_config.custom_model = 'startup-test-model'
        window._sync_settings_to_tabs(force=True)
        for prefix in ('score', 'stat_extract', 'extract', 'validate', 'synth', 'interrater', 'verify'):
            assert getattr(window, prefix + '_provider').currentText() == 'custom'
            assert getattr(window, prefix + '_model').text() == 'startup-test-model'
        window._populate_corpus_browser([{'title': 'Startup corpus', 'year': 2026}])
        window._filter_corpus_table('startup')
        assert window._corpus_proxy.rowCount() == 1
        from core.models import AlignmentResult, AnalysisSession
        window.session = AnalysisSession(
            alignment_results=[
                AlignmentResult(
                    component_id="c1",
                    component_label="White matter",
                    support_score=0.4,
                    supporting_count=2,
                    contradicting_count=1,
                ),
            ],
            total_articles_analyzed=3,
        )
        window._update_alignment_dashboard()
        assert window.alignment_bars_layout.count() == 1
        assert window.dash_supporting_card._value_label.text() == "2"
        assert window.dash_overall_card._value_label.text() == "+0.40"
        window.show()
        app.processEvents()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        print('MainWindow startup and shutdown passed', flush=True)
    """)
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "MainWindow startup and shutdown passed" in completed.stdout
