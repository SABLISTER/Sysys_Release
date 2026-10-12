"""Fetch Models button: stage rows and Settings dialog."""
from __future__ import annotations

from typing import Any, cast

from PySide6.QtWidgets import QApplication, QWidget

from core.config import Config
from gui import model_picker
from process.custom_endpoint import CustomEndpoint, ModelListError


def _app() -> QApplication:
    return cast(QApplication, QApplication.instance() or QApplication([]))


def test_stage_row_fetch_is_custom_only_and_fills_completer(monkeypatch):
    from gui.main_window import _make_llm_backend_row

    _app()
    requested: list[CustomEndpoint] = []
    endpoint = CustomEndpoint("openai", "http://h:1234", "k")
    monkeypatch.setattr(model_picker, "list_models",
                        lambda ep: requested.append(ep) or ["b-model", "a-model"])
    holder = QWidget()
    row, combo, model_edit = _make_llm_backend_row(endpoint_getter=lambda: endpoint)
    holder.setLayout(row)
    button = model_edit._fetch_button

    assert combo.findText("lmstudio") < 0 and combo.findText("longcat") < 0
    assert not button.isEnabled()
    combo.setCurrentText("custom")
    assert button.isEnabled()

    assert button.fetch() == ["b-model", "a-model"]
    assert requested == [endpoint]
    assert model_edit.completer().model().stringList() == ["b-model", "a-model"]

    # Choosing a fetched model counts as a hand edit so project syncs keep it.
    model_edit.completer().activated.emit("a-model")
    assert model_edit._user_customized is True


def test_fetch_failure_shows_message_and_keeps_text(monkeypatch):
    from gui.main_window import _make_llm_backend_row

    _app()
    def fail(_endpoint):
        raise ModelListError("Could not reach http://h:1")
    shown: list[str] = []
    monkeypatch.setattr(model_picker, "list_models", fail)
    monkeypatch.setattr(model_picker.QMessageBox, "warning",
                        lambda _parent, _title, message: shown.append(message))
    _, combo, model_edit = _make_llm_backend_row(
        endpoint_getter=lambda: CustomEndpoint("openai", "http://h:1")
    )
    model_edit.setText("typed-model")

    assert model_edit._fetch_button.fetch() == []
    assert shown == ["Could not reach http://h:1"]
    assert model_edit.text() == "typed-model"


def test_settings_fetch_uses_unsaved_widget_values_and_applies_format(monkeypatch, tmp_path):
    from gui.dialogs import SettingsDialog

    _app()
    cfg = Config(project_root=tmp_path, llm_provider="custom", custom_api_format="lmstudio",
                 custom_base_url="http://saved:1", custom_model="saved-model")
    window: Any = QWidget()
    window.pipeline_config = cfg
    dialog = SettingsDialog(window)

    assert dialog.settings_provider.currentText() == "custom"
    assert dialog.settings_custom_format.currentData() == "lmstudio"
    assert dialog.settings_custom_url.text() == "http://saved:1"

    requested: list[CustomEndpoint] = []
    monkeypatch.setattr(model_picker, "list_models",
                        lambda ep: requested.append(ep) or ["gw-model"])
    dialog.settings_custom_url.setText("https://gateway.test/v1")
    dialog.settings_custom_format.setCurrentIndex(
        dialog.settings_custom_format.findData("anthropic")
    )
    dialog.settings_custom_key.setText("gw-key")
    dialog.settings_custom_fetch.fetch()
    assert requested == [CustomEndpoint("anthropic", "https://gateway.test/v1", "gw-key")]

    dialog.settings_custom_model.setText("gw-model")
    dialog._apply_to_config(cfg)
    assert (cfg.custom_api_format, cfg.custom_base_url, cfg.custom_model, cfg.custom_api_key) == (
        "anthropic", "https://gateway.test/v1", "gw-model", "gw-key"
    )
