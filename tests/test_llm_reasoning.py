"""Unit tests for LLM reasoning/thinking helpers."""
from __future__ import annotations

import pytest

from process.llm import (
    _anthropic_thinking,
    _is_reasoning_model,
    _openai_reasoning_effort,
    _status_icon,
    _supports_anthropic_thinking,
    parse_llm_json,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("o1-mini", True),
        ("o3-mini", True),
        ("o4", True),
        ("claude-3-7-sonnet-20250219", True),
        ("claude-4-sonnet-20251001", True),
        ("gpt-4o", False),
        ("claude-3-5-sonnet-20241022", False),
        ("lmstudio-model", False),
        ("", False),
    ],
)
def test_is_reasoning_model(model: str, expected: bool):
    assert _is_reasoning_model(model) is expected


@pytest.mark.parametrize(
    "think,expected",
    [
        (None, None),
        (False, None),
        ("off", None),
        ("false", None),
        (True, "medium"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("on", "medium"),
        ("true", "medium"),
        ("unknown", "medium"),
    ],
)
def test_openai_reasoning_effort(think, expected):
    assert _openai_reasoning_effort(think) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-3-7-sonnet-20250219", True),
        ("claude-4-sonnet-20251001", True),
        ("claude-3-5-sonnet-20241022", False),
        ("gpt-4o", False),
    ],
)
def test_supports_anthropic_thinking(model: str, expected: bool):
    assert _supports_anthropic_thinking(model) is expected


def test_anthropic_thinking_disabled_when_max_tokens_too_small():
    assert _anthropic_thinking(True, 512) is None


def test_anthropic_thinking_enabled_with_reasonable_budget():
    thinking = _anthropic_thinking(True, 4096)
    assert thinking == {"type": "enabled", "budget_tokens": 3276}


def test_anthropic_thinking_clamps_budget_below_max_tokens():
    thinking = _anthropic_thinking(True, 1025)
    assert thinking == {"type": "enabled", "budget_tokens": 1024}


def test_anthropic_thinking_disabled_at_exactly_1024():
    # budget_tokens must be > 1024 AND < max_tokens; at exactly 1024 no
    # valid budget exists, so thinking must be disabled (Anthropic rejects
    # budget_tokens == max_tokens).
    assert _anthropic_thinking(True, 1024) is None


@pytest.mark.parametrize(
    "finish_reason,content,expected",
    [
        ("max_tokens", "some content", "✂"),  # Anthropic truncation signal
        ("max_tokens", None, "⏱"),
        ("length", "some content", "✂"),  # OpenAI truncation signal
        ("length", None, "⏱"),
        ("stop", "some content", "✓"),
        ("stop", "", "∅"),
    ],
)
def test_status_icon_recognizes_anthropic_truncation(finish_reason, content, expected):
    assert _status_icon(finish_reason, content) == expected


def test_parse_llm_json_plain_object():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_llm_json_strips_code_fences():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_llm_json_extracts_object_from_prose():
    text = 'Here is the result: {"a": 1, "b": [1, 2]} hope that helps.'
    assert parse_llm_json(text) == {"a": 1, "b": [1, 2]}


def test_parse_llm_json_returns_none_when_unparseable():
    assert parse_llm_json("") is None
    assert parse_llm_json("no json here") is None
    assert parse_llm_json("[1, 2, 3]") is None
