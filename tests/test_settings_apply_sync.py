"""Tests for Preferences Apply propagation into stage controls."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)

from core.config import Config
from gui.dialogs import SettingsDialog
from gui.main_window import MainWindow


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def _llm_pair(provider: str = "openai", model: str = "old-model") -> tuple[Any, Any]:
    combo: Any = QComboBox()
    combo.addItems(["ollama", "custom", "openai", "anthropic", "brt-bert", "brt-irr"])
    combo.setCurrentText(provider)
    combo._user_customized = True
    edit: Any = QLineEdit()
    edit.setText(model)
    edit._user_customized = True
    return combo, edit


class DummyWindow:
    def __init__(self, config: Config):
        self.pipeline_config = config
        self.sysys_nl_query = QLineEdit()
        self.fulltext_only = QCheckBox()
        self.fetch_fulltext = QCheckBox()

        for prefix in ("score", "stat_extract", "extract", "validate", "synth", "interrater", "verify"):
            combo, edit = _llm_pair()
            setattr(self, f"{prefix}_provider", combo)
            setattr(self, f"{prefix}_model", edit)

        self.score_relevance_model: Any = QComboBox()
        self.score_relevance_model.setEditable(True)
        self.score_relevance_model.addItems(["brt-bert", "housecatbert-scibert"])
        self.score_relevance_model._user_customized = True

        self.score_threshold = QSpinBox()
        self.score_batch = QSpinBox()
        self.extract_max_tokens = QSpinBox()
        self.extract_thinking = QCheckBox()
        self.synth_min_cluster = QSpinBox()
        self.sysys_ft_workers = QSpinBox()
        self.sysys_ft_streaming = QCheckBox()
        self.sysys_ft_batch_size = QSpinBox()
        self.sysys_ft_paperqa_enabled = QCheckBox()
        self.sysys_ft_paperqa_batch_size = QSpinBox()

    def _sync_all_stage_paths(self, config: Config) -> None:
        pass

    def _apply_score_advanced_mode_from_settings(self) -> None:
        pass

    def _log(self, *_args, **_kwargs) -> None:
        pass


def test_preferences_apply_force_propagates_provider_and_model_to_all_stage_rows(tmp_path):
    _app()
    cfg = Config(project_root=tmp_path)
    cfg.llm_provider = "custom"
    cfg.custom_model = "local-review-model"

    window = DummyWindow(cfg)

    MainWindow._sync_settings_to_tabs(cast(MainWindow, window), force=True)

    for prefix in ("score", "stat_extract", "extract", "validate", "synth", "interrater", "verify"):
        combo = getattr(window, f"{prefix}_provider")
        edit = getattr(window, f"{prefix}_model")
        assert combo.currentText() == "custom"
        assert edit.text() == "local-review-model"
        assert combo._user_customized is False
        assert edit._user_customized is False


def test_preferences_custom_key_is_masked_and_updates_runtime_config(tmp_path):
    _app()
    cfg = Config(project_root=tmp_path)
    cfg.custom_api_key = "configured-key"
    window: Any = QWidget()
    window.pipeline_config = cfg

    dialog = SettingsDialog(window)

    assert dialog.settings_custom_key.text() == "configured-key"
    assert dialog.settings_custom_key.echoMode() == QLineEdit.EchoMode.Password

    dialog.settings_custom_key.setText("preferences-key")
    dialog._apply_to_config(cfg)

    assert cfg.custom_api_key == "preferences-key"


def test_fulltext_preferences_populate_apply_and_sync_to_stage_controls(tmp_path):
    _app()
    cfg = Config(project_root=tmp_path)
    cfg.fulltext_download_workers = 5
    cfg.fulltext_streaming = False
    cfg.fulltext_download_batch_size = 17
    cfg.paperqa_index_enabled = False
    cfg.paperqa_index_batch_size = 23

    preferences_host: Any = QWidget()
    preferences_host.pipeline_config = cfg
    dialog = SettingsDialog(preferences_host)

    assert dialog.settings_fulltext_streaming.isChecked() is False
    assert dialog.settings_fulltext_batch_size.value() == 17
    assert dialog.settings_paperqa_index_enabled.isChecked() is False
    assert dialog.settings_paperqa_batch_size.value() == 23

    dialog.settings_fulltext_streaming.setChecked(True)
    dialog.settings_fulltext_batch_size.setValue(29)
    dialog.settings_paperqa_index_enabled.setChecked(True)
    dialog.settings_paperqa_batch_size.setValue(31)
    dialog._apply_to_config(cfg)

    assert cfg.fulltext_streaming is True
    assert cfg.fulltext_download_batch_size == 29
    assert cfg.paperqa_index_enabled is True
    assert cfg.paperqa_index_batch_size == 31

    window = DummyWindow(cfg)
    MainWindow._sync_settings_to_tabs(cast(MainWindow, window), force=True)

    assert window.sysys_ft_workers.value() == 5
    assert window.sysys_ft_streaming.isChecked() is True
    assert window.sysys_ft_batch_size.value() == 29
    assert window.sysys_ft_batch_size.isEnabled() is False
    assert window.sysys_ft_paperqa_enabled.isChecked() is True
    assert window.sysys_ft_paperqa_batch_size.value() == 31

    window.sysys_ft_workers.setValue(7)
    window.sysys_ft_streaming.setChecked(False)
    window.sysys_ft_batch_size.setValue(11)
    window.sysys_ft_paperqa_enabled.setChecked(False)
    window.sysys_ft_paperqa_batch_size.setValue(13)
    MainWindow._sync_fulltext_controls_to_config(cast(MainWindow, window))

    assert cfg.fulltext_download_workers == 7
    assert cfg.fulltext_streaming is False
    assert cfg.fulltext_download_batch_size == 11
    assert cfg.paperqa_index_enabled is False
    assert cfg.paperqa_index_batch_size == 13


def test_preferences_remember_toggle_controls_custom_key_storage(
    monkeypatch, tmp_path
):
    _app()

    class FakeSettings:
        def __init__(self):
            self.values = {}

        def value(self, key, default="", type=None):  # noqa: A002
            value = self.values.get(key, default)
            return type(value) if type is not None else value

        def setValue(self, key, value):
            self.values[key] = value

        def remove(self, key):
            self.values.pop(key, None)

    settings = FakeSettings()
    monkeypatch.setattr("gui.dialogs.QSettings", lambda *args: settings)

    cfg = Config(project_root=tmp_path)
    window: Any = QWidget()
    window.pipeline_config = cfg
    dialog = SettingsDialog(window)
    dialog.settings_custom_key.setText("remembered-key")

    dialog.settings_remember_keys.setChecked(True)
    dialog._save_keys_to_qsettings()
    assert settings.values["keys/custom_api_key"] == "remembered-key"

    dialog.settings_remember_keys.setChecked(False)
    dialog._save_keys_to_qsettings()
    assert "keys/custom_api_key" not in settings.values
