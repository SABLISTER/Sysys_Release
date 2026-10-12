"""Tests for persisted GUI settings applied to runtime config."""
from __future__ import annotations

from core.config import Config
from gui.settings_persistence import restore_saved_settings


class FakeSettings:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def value(self, key: str, default="", type=str):  # noqa: A002 - mirrors QSettings
        value = self._values.get(key, default)
        return type(value) if type is not None else value


def test_saved_llm_preferences_override_non_empty_config_defaults(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.llm_provider = "openai"
    cfg.openai_model = "gpt-5"

    restore_saved_settings(
        cfg,
        FakeSettings({
            "llm/provider": "lmstudio",
            "llm/lmstudio_model": "local-stats-model",
            "llm/lmstudio_url": "http://127.0.0.1:1234",
        }),
    )

    # Legacy LM Studio preferences migrate onto the custom provider.
    assert cfg.llm_provider == "custom"
    assert cfg.custom_api_format == "lmstudio"
    assert cfg.llm_model == "local-stats-model"
    assert cfg.custom_base_url == "http://127.0.0.1:1234"


def test_saved_api_key_does_not_override_existing_config_value(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.openai_api_key = "env-or-project-key"

    restore_saved_settings(
        cfg,
        FakeSettings({"keys/openai_api_key": "saved-qsettings-key"}),
    )

    assert cfg.openai_api_key == "env-or-project-key"


def test_saved_lmstudio_api_key_fills_empty_config_value(tmp_path):
    cfg = Config(project_root=tmp_path)

    restore_saved_settings(
        cfg,
        FakeSettings({"keys/lmstudio_api_key": "saved-lmstudio-key"}),
    )

    assert cfg.custom_api_key == "saved-lmstudio-key"


def test_renamed_settings_migrate_once_and_preserve_new_values(tmp_path):
    from PySide6.QtCore import QSettings
    from gui.settings_persistence import migrate_legacy_settings
    old = QSettings(str(tmp_path / 'old.ini'), QSettings.Format.IniFormat)
    new = QSettings(str(tmp_path / 'new.ini'), QSettings.Format.IniFormat)
    for key, value in {'llm/provider': 'lmstudio', 'llm/lmstudio_url': 'http://localhost:12345',
                       'llm/lmstudio_model': 'old-model', 'keys/remember_keys': True,
                       'keys/lmstudio_api_key': 'test-secret', 'appearance/font_size': 13,
                       'projects/recent': ['/a/project']}.items():
        old.setValue(key, value)
    new.setValue('llm/lmstudio_model', 'new-model')
    migrate_legacy_settings(new, old)
    assert new.value('llm/provider') == 'lmstudio'
    assert new.value('llm/lmstudio_url') == 'http://localhost:12345'
    assert new.value('llm/lmstudio_model') == 'new-model'
    assert new.value('keys/lmstudio_api_key') == 'test-secret'
    assert new.value('appearance/font_size', type=int) == 13
    assert new.value('projects/recent') == ['/a/project']
    new.remove('keys/lmstudio_api_key')
    migrate_legacy_settings(new, old)
    assert not new.contains('keys/lmstudio_api_key')


def test_settings_migration_honors_new_credential_opt_out(tmp_path):
    from PySide6.QtCore import QSettings
    from gui.settings_persistence import migrate_legacy_settings
    old = QSettings(str(tmp_path / 'old.ini'), QSettings.Format.IniFormat)
    new = QSettings(str(tmp_path / 'new.ini'), QSettings.Format.IniFormat)
    old.setValue('keys/remember_keys', True)
    old.setValue('keys/lmstudio_api_key', 'test-secret')
    old.setValue('llm/provider', 'lmstudio')
    new.setValue('keys/remember_keys', False)
    migrate_legacy_settings(new, old)
    assert not new.contains('keys/lmstudio_api_key')
    assert new.value('llm/provider') == 'lmstudio'
