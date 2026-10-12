"""Tests that config save strips API keys from the written JSON."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from gui.dialogs import SettingsDialog


class _FakeWidget:
    def __init__(self, value):
        self._value = value

    def text(self):
        return self._value

    def currentText(self):
        return self._value

    def currentData(self):
        return self._value

    def value(self):
        return self._value


class _FakeMainWindow:
    def __init__(self, tmp_path: Path):
        self.pipeline_config = SimpleNamespace(project_root=tmp_path)
        self._logs: list[tuple] = []

    def _log(self, msg, level="INFO"):
        self._logs.append((level, msg))


def _fake_dialog(tmp_path: Path):
    """Build a minimal object for testing SettingsDialog._on_save_config."""
    dialog = SimpleNamespace()
    dialog.main_window = _FakeMainWindow(tmp_path)
    widgets = {
        "settings_project_root": "",
        "settings_provider": "openai",
        "settings_ollama_model": "",
        "settings_ollama_url": "",
        "settings_custom_format": "openai",
        "settings_custom_url": "",
        "settings_custom_model": "",
        "settings_custom_key": "secret-custom-key",
        "settings_openai_model": "gpt-4o",
        "settings_openai_url": "",
        "settings_anthropic_model": "",
        "settings_openai_key": "secret-openai-key",
        "settings_anthropic_key": "secret-anthropic-key",
        "settings_s2_key": "secret-s2-key",
        "settings_pubmed_key": "secret-pubmed-key",
        "settings_pubmed_email": "test@example.com",
        "settings_elicit_key": "secret-elicit-key",
        "settings_scopus_key": "secret-scopus-key",
        "settings_scopus_insttoken": "secret-insttoken",
        "settings_wos_key": "secret-wos-key",
        "settings_wos_researcher_key": "secret-wos-researcher-key",
    }
    for name, value in widgets.items():
        setattr(dialog, name, _FakeWidget(value))
    for name in (
        "settings_threshold",
        "settings_workers",
        "settings_batch_size",
        "settings_score_max_tokens",
        "settings_stagger_delay",
        "settings_timeout",
    ):
        setattr(dialog, name, _FakeWidget(1))
    dialog._custom_format_from_widget = lambda: SettingsDialog._custom_format_from_widget(
        cast(SettingsDialog, dialog)
    )
    return dialog


def test_on_save_config_strips_api_keys(tmp_path: Path):
    dialog = _fake_dialog(tmp_path)
    save_path = tmp_path / "saved_config.json"
    captured: dict = {}

    def capture_dump(data, f, **kwargs):
        captured.update(data)

    with patch("gui.dialogs.QFileDialog.getSaveFileName", return_value=(str(save_path), "JSON files (*.json)")), \
         patch("gui.dialogs.json.dump", side_effect=capture_dump):
        SettingsDialog._on_save_config(cast(SettingsDialog, dialog))

    assert captured.get("openai_model") == "gpt-4o"
    assert captured.get("pubmed_email") == "test@example.com"

    # No API keys should be present
    assert "openai_api_key" not in captured
    assert "custom_api_key" not in captured
    assert "longcat_api_key" not in captured
    assert "anthropic_api_key" not in captured
    assert "s2_api_key" not in captured
    assert "pubmed_api_key" not in captured
    assert "elicit_api_key" not in captured
    assert "scopus_api_key" not in captured
    assert "scopus_insttoken" not in captured
    assert "wos_api_key" not in captured
    assert "wos_researcher_api_key" not in captured

    # A warning should have been logged listing stripped keys
    warnings = [msg for level, msg in dialog.main_window._logs if level == "WARNING"]
    assert any("openai_api_key" in msg for msg in warnings)
    assert any("custom_api_key" in msg for msg in warnings)


def test_project_save_excludes_credentials_and_preserves_previous_file_on_error(tmp_path, monkeypatch):
    import core.io
    from core.config import Config
    from gui.project_manager import ProjectManager
    from gui.settings_persistence import API_KEY_QSETTINGS_KEYS
    cfg = Config(project_root=tmp_path)
    credential_fields = set(API_KEY_QSETTINGS_KEYS) - {'pubmed_email'}
    credential_fields.update(name for name in vars(cfg) if name.endswith('_api_key'))
    credential_fields.add('future_provider_api_key')
    for field in credential_fields:
        setattr(cfg, field, 'test-value-' + field)
    cfg.pubmed_email = 'researcher@example.test'
    state = {'stage7': {'resume': True}}
    ProjectManager.save_project(tmp_path, cfg, state)
    path = tmp_path / 'project_state.json'
    before = path.read_bytes()
    saved = json.loads(before)
    assert saved['window_state'] == state
    for field in credential_fields:
        assert field not in saved['config']
    assert saved['config']['pubmed_email'] == cfg.pubmed_email
    def fail_replace(*args):
        raise OSError('Simulated disk failure')
    monkeypatch.setattr(core.io, '_replace_preserving_mode', fail_replace)
    ProjectManager.save_project(tmp_path, cfg, {'changed': True})
    assert path.read_bytes() == before
