"""Tests for the ninth fix batch: Stage 8 second-rater wiring.

The bug: ``_start_interrater`` built its config from the Interrater tab's
provider/model widgets, which overwrote ``cfg.llm_provider`` with the SECOND
rater — then handed that same config over as the primary. Cross-model
reliability silently degraded to one model rating itself, and nothing in the
log said so.

The second bug: ``set_config(second_cfg)`` was restored only on the success
path, so any exception inside Stage 8 left the process-global LLM config
pointing at the second rater for every later stage.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

import process.llm as llm
from core.config import Config
from gui.stage_runner import (
    _same_rater,
    resolve_primary_rater,
    resolve_second_rater,
    run_interrater,
)


@pytest.fixture
def cfg():
    c = Config()
    c.llm_provider = "openai"
    c.openai_model = "gpt-4o"
    c.ollama_model = "llama3"
    c.anthropic_model = "claude-sonnet"
    return c


@pytest.fixture(autouse=True)
def _quiet(caplog):
    caplog.set_level(logging.WARNING)


def _run(cfg, **kw):
    """Run the stage with s8.run stubbed out; returns (result, messages)."""
    messages: list[str] = []
    payload = kw.pop("_payload", {})
    side_effect = kw.pop("_side_effect", None)
    target = "stages.s8_interrater.run"
    patcher = (patch(target, side_effect=side_effect) if side_effect
               else patch(target, return_value=payload))
    with patcher:
        result = asyncio.run(
            run_interrater(cfg, progress_callback=messages.append, **kw))
    return result, messages


# ── Second-rater resolution ───────────────────────────────────────────────

def test_second_rater_switches_provider_on_its_own_config(cfg):
    second_cfg, provider, model = resolve_second_rater(cfg, "ollama", None)
    assert (provider, model) == ("ollama", "llama3")
    assert second_cfg.llm_provider == "ollama"
    # The caller's config must not be mutated.
    assert cfg.llm_provider == "openai"


def test_second_rater_model_override_lands_on_the_right_field(cfg):
    second_cfg, provider, model = resolve_second_rater(cfg, "anthropic", "claude-opus")
    assert (provider, model) == ("anthropic", "claude-opus")
    assert second_cfg.anthropic_model == "claude-opus"
    assert cfg.anthropic_model == "claude-sonnet"


def test_second_rater_resolves_a_blank_model_to_a_concrete_name(cfg):
    """s8.run must never be handed None; the report names the model used."""
    _, _, model = resolve_second_rater(cfg, "ollama", "")
    assert model == "llama3"


def test_second_rater_normalizes_provider_case(cfg):
    _, provider, _ = resolve_second_rater(cfg, "OpenAI", None)
    assert provider == "openai"


def test_second_rater_rejects_unknown_provider(cfg):
    """llm.chat falls through to Ollama for unknown providers.

    Without this guard a typo silently rates with a model nobody chose.
    """
    with pytest.raises(ValueError, match="Unsupported second-rater provider"):
        resolve_second_rater(cfg, "gpt5-turbo-supreme", None)


@pytest.mark.parametrize("rater", ["brt-bert", "brt_bert", "brt-irr", "brt_irr"])
def test_cross_encoder_second_rater_leaves_llm_settings_alone(cfg, rater):
    second_cfg, provider, model = resolve_second_rater(cfg, rater, "/ckpt/final")
    assert provider == rater
    assert model == "/ckpt/final"
    assert second_cfg.llm_provider == "openai"


# ── Primary-rater resolution ──────────────────────────────────────────────

def test_primary_falls_back_to_config_when_not_supplied(cfg):
    assert resolve_primary_rater(cfg) == ("openai", "gpt-4o")


def test_primary_blank_model_resolves_like_the_second_rater(cfg):
    """A blank Stage 2 Model field still names gpt-4o, so the two compare."""
    assert resolve_primary_rater(cfg, "openai", "") == ("openai", "gpt-4o")


def test_primary_cross_encoder_is_kept_verbatim(cfg):
    assert resolve_primary_rater(cfg, "brt-bert", "") == ("brt-bert", "")


# ── Same-rater detection ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "primary,second,expected",
    [
        (("openai", "gpt-4o"), ("openai", "gpt-4o"), True),
        (("openai", "gpt-4o"), ("openai", "gpt-4o-mini"), False),
        (("openai", "gpt-4o"), ("ollama", "llama3"), False),
        (("brt-bert", ""), ("openai", "gpt-4o"), False),
        (("", ""), ("openai", "gpt-4o"), False),
        (("openai", "gpt-4o"), ("", ""), False),
    ],
)
def test_same_rater(primary, second, expected):
    assert _same_rater(primary, second) is expected


def test_warns_when_both_raters_are_the_same_model(cfg):
    result, messages = _run(cfg, second_provider="openai", _payload={})
    assert result["same_rater"] is True
    assert any("identical to the primary" in m for m in messages)


def test_no_warning_for_genuinely_distinct_raters(cfg):
    result, messages = _run(cfg, second_provider="ollama", _payload={})
    assert result["same_rater"] is False
    assert not any("identical" in m for m in messages)


def test_blank_model_fields_still_detect_a_same_model_pairing(cfg):
    """Regression: both sides defaulting must not read as two raters."""
    result, _ = _run(cfg, second_provider="openai", second_model=None,
                     primary_provider="openai", primary_model=None,
                     _payload={})
    assert result["same_rater"] is True


# ── Global config handling ────────────────────────────────────────────────

def test_restores_primary_config_after_a_successful_run(cfg):
    llm.set_config(cfg)
    _run(cfg, second_provider="ollama", _payload={})
    assert llm._config is not None
    assert llm._config.llm_provider == "openai"


def test_restores_primary_config_after_a_failed_run(cfg):
    """Without try/finally the GUI would keep rating with the second model."""
    llm.set_config(cfg)
    with pytest.raises(RuntimeError):
        _run(cfg, second_provider="ollama", _side_effect=RuntimeError("boom"))
    assert llm._config is not None
    assert llm._config.llm_provider == "openai"


def test_second_config_is_active_while_stage_8_runs(cfg):
    """The swap must actually take effect, not just be restored."""
    llm.set_config(cfg)
    seen = {}

    def _spy(**kwargs):
        assert llm._config is not None
        seen["provider"] = llm._config.llm_provider
        seen["model"] = llm._config.llm_model
        return {}

    with patch("stages.s8_interrater.run", side_effect=_spy):
        asyncio.run(run_interrater(cfg, second_provider="anthropic",
                                   second_model="claude-opus"))
    assert seen == {"provider": "anthropic", "model": "claude-opus"}
    assert llm._config is not None
    assert llm._config.llm_provider == "openai"


# ── Hand-off to s8.run ────────────────────────────────────────────────────

def test_stage8_receives_resolved_second_rater_and_no_primary_kwargs(cfg):
    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        return {}

    with patch("stages.s8_interrater.run", side_effect=_spy):
        asyncio.run(run_interrater(cfg, second_provider="ollama",
                                   primary_provider="openai"))
    assert seen["second_provider"] == "ollama"
    assert seen["second_model"] == "llama3"
    # s8.run has no primary_* parameters; leaking them would be a TypeError.
    assert "primary_provider" not in seen
    assert "primary_model" not in seen


def test_result_records_both_raters_for_the_report(cfg):
    result, _ = _run(cfg, second_provider="ollama", _payload={"score_kappa": 0.4})
    assert result["primary_provider"] == "openai"
    assert result["primary_model"] == "gpt-4o"
    assert result["second_provider"] == "ollama"
    assert result["second_model"] == "llama3"
    assert result["score_kappa"] == 0.4


def test_result_does_not_clobber_values_stage8_already_set(cfg):
    payload = {"second_provider": "from-s8", "same_rater": True}
    result, _ = _run(cfg, second_provider="ollama", _payload=payload)
    assert result["second_provider"] == "from-s8"
    assert result["same_rater"] is True


def test_non_dict_result_passes_through(cfg):
    result, _ = _run(cfg, second_provider="ollama", _payload=None)
    assert result is None


# ── Research question reaches the LLM second rater ────────────────────────

def _run_s8_capturing_prompts(tmp_path, monkeypatch, cfg):
    """Run the real Stage 8 on one paper; return the scoring prompts sent."""
    import json
    from stages import s8_interrater

    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps([{
        "title": "Ketamine and depression",
        "abstract": "Ketamine reduced depressive symptoms in a randomized trial.",
        "relevance_score": 4,
    }]), encoding="utf-8")

    prompts: list[str] = []

    def fake_chat(model, messages, **kw):
        prompts.append(messages[0]["content"])
        return '{"score": 4, "quoted_span": ""}'

    import process.relevance_filter  # noqa: F401 — bind the real chat before patching
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(s8_interrater, "provider_extract_claims",
                        lambda *a, **k: {})
    s8_interrater.run(claims_path=str(claims), output_dir=str(tmp_path),
                      config=cfg, second_provider="ollama", sample_size=1)
    return [p for p in prompts if "RESEARCH QUESTION:" in p]


def test_llm_second_rater_scores_against_the_configured_question(
        cfg, tmp_path, monkeypatch):
    cfg.project_root = tmp_path
    cfg._resolve_paths()
    question = "Does ketamine reduce treatment-resistant depression?"
    monkeypatch.setattr(cfg, "get_natural_language_question", lambda: question)

    prompts = _run_s8_capturing_prompts(tmp_path, monkeypatch, cfg)

    assert prompts, "second rater never sent a scoring prompt"
    assert f"RESEARCH QUESTION: {question}" in prompts[0]
    assert "the research topic" not in prompts[0]


def test_llm_second_rater_falls_back_to_hypothesis_text(
        cfg, tmp_path, monkeypatch):
    cfg.project_root = tmp_path
    cfg._resolve_paths()
    cfg.hypothesis_text = "Ketamine is an effective antidepressant."
    monkeypatch.setattr(cfg, "get_natural_language_question", lambda: "")

    prompts = _run_s8_capturing_prompts(tmp_path, monkeypatch, cfg)

    assert "RESEARCH QUESTION: Ketamine is an effective antidepressant." in prompts[0]


def test_unparseable_second_rater_reply_counts_as_zero(cfg, tmp_path, monkeypatch):
    import json
    from stages import s8_interrater

    cfg.project_root = tmp_path
    cfg._resolve_paths()
    claims = tmp_path / "claims.json"
    claims.write_text(json.dumps([{"title": "T", "abstract": "Some text.",
                                   "relevance_score": 4}]), encoding="utf-8")
    import process.relevance_filter  # noqa: F401 — bind the real chat before patching
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "I cannot decide.")
    monkeypatch.setattr(s8_interrater, "provider_extract_claims",
                        lambda *a, **k: {})

    result = s8_interrater.run(claims_path=str(claims), output_dir=str(tmp_path),
                               config=cfg, second_provider="ollama", sample_size=1)

    assert "error" not in result
    saved = json.loads((tmp_path / "data" / "validation" /
                        "interrater_results.json").read_text(encoding="utf-8"))
    assert saved[0]["second_score"] == 0
