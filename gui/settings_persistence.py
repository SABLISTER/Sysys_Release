"""Helpers for applying persisted GUI settings to runtime config objects."""

from __future__ import annotations

from typing import Any


def migrate_legacy_settings(settings=None, legacy=None) -> None:
    """Copy preferences once after the Systes -> Sysys organization rename.

    Existing new-store values (including deliberate empty values) win. A
    completed migration is never repeated, so removed keys stay removed.
    """
    from PySide6.QtCore import QSettings

    if settings is None:
        settings = QSettings("Sysys", "HypothesisConsensusAnalyzer")
    if legacy is None:
        legacy = QSettings("Systes", "HypothesisConsensusAnalyzer")
    marker = "migration/systes_settings_v1"
    if settings.value(marker, False, type=bool):
        return
    remember_key = "keys/remember_keys"
    remember = settings.value(remember_key, legacy.value(remember_key, False, type=bool), type=bool)
    credentials = (
        set(API_KEY_QSETTINGS_KEYS.values()) | LEGACY_CREDENTIAL_QSETTINGS_KEYS
    ) - {"keys/pubmed_email"}
    for key in legacy.allKeys():
        if not settings.contains(key) and (key not in credentials or remember):
            settings.setValue(key, legacy.value(key))
    settings.setValue(marker, True)
    settings.sync()


API_KEY_QSETTINGS_KEYS: dict[str, str] = {
    "openai_api_key": "keys/openai_api_key",
    "custom_api_key": "keys/custom_api_key",
    "anthropic_api_key": "keys/anthropic_api_key",
    "s2_api_key": "keys/s2_api_key",
    "pubmed_api_key": "keys/pubmed_api_key",
    "pubmed_email": "keys/pubmed_email",
    "elicit_api_key": "keys/elicit_api_key",
    "scopus_api_key": "keys/scopus_api_key",
    "scopus_insttoken": "keys/scopus_insttoken",
    "wos_api_key": "keys/wos_api_key",
    "wos_researcher_api_key": "keys/wos_researcher_api_key",
}

LLM_QSETTINGS_KEYS: dict[str, str] = {
    "llm_provider": "llm/provider",
    "ollama_model": "llm/ollama_model",
    "ollama_base_url": "llm/ollama_url",
    "custom_api_format": "llm/custom_format",
    "custom_base_url": "llm/custom_url",
    "custom_model": "llm/custom_model",
    "openai_model": "llm/openai_model",
    "openai_base_url": "llm/openai_url",
    "anthropic_model": "llm/anthropic_model",
}

# Keys written before lmstudio/longcat were folded into the custom provider.
# They are read once through migrate_legacy_llm_fields and removed on save.
LEGACY_LLM_QSETTINGS_KEYS: dict[str, str] = {
    "lmstudio_model": "llm/lmstudio_model",
    "lmstudio_base_url": "llm/lmstudio_url",
    "lmstudio_api_key": "keys/lmstudio_api_key",
    "longcat_model": "llm/longcat_model",
    "longcat_base_url": "llm/longcat_url",
    "longcat_api_key": "keys/longcat_api_key",
}
LEGACY_CREDENTIAL_QSETTINGS_KEYS: set[str] = {
    qs_key for field, qs_key in LEGACY_LLM_QSETTINGS_KEYS.items()
    if field.endswith("_api_key")
}


def _saved_text(settings: Any, key: str) -> str:
    value = settings.value(key, "", type=str)
    return str(value or "").strip()


def restore_saved_settings(config: Any, settings: Any) -> None:
    """Layer saved QSettings values onto a Config-like object.

    API credentials keep the conservative env/config > QSettings priority, but
    LLM provider/model/base-url fields are user preferences and should override
    the non-empty dataclass defaults such as ``openai``.
    """
    from core.config import migrate_legacy_llm_fields

    all_keys = {**API_KEY_QSETTINGS_KEYS, **LLM_QSETTINGS_KEYS, **LEGACY_LLM_QSETTINGS_KEYS}
    saved = {field: _saved_text(settings, qs_key) for field, qs_key in all_keys.items()}
    saved = migrate_legacy_llm_fields({k: v for k, v in saved.items() if v})

    for cfg_field in API_KEY_QSETTINGS_KEYS:
        if not hasattr(config, cfg_field) or not saved.get(cfg_field):
            continue
        current = getattr(config, cfg_field, "")
        if current and str(current).strip():
            continue
        setattr(config, cfg_field, saved[cfg_field])

    for cfg_field in LLM_QSETTINGS_KEYS:
        if hasattr(config, cfg_field) and saved.get(cfg_field):
            setattr(config, cfg_field, saved[cfg_field])
