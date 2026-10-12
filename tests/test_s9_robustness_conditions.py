"""Tests for configurable condition handling in Stage 9 robustness."""
from __future__ import annotations

from stages.s9_robustness import (
    _extract_conditions,
    _load_condition_config,
    _normalize_condition,
)


class FakeConfig:
    def __init__(self, conditions=None):
        self.co_occurring_conditions = conditions or []
        self.research_config: dict | None = None


def test_load_condition_config_from_config():
    cfg = FakeConfig(["Depression", "Anxiety", "PTSD"])
    patterns, ranking = _load_condition_config(cfg)
    assert ranking == ["Depression", "Anxiety", "PTSD"]
    assert patterns == {
        "depression": "Depression",
        "anxiety": "Anxiety",
        "ptsd": "PTSD",
    }


def test_load_condition_config_neuroscience_pack_uses_legacy_ranking():
    cfg = FakeConfig([])
    cfg.research_config = {"defaults": {"pattern_pack": "neuroscience"}}
    patterns, ranking = _load_condition_config(cfg)
    assert "ADHD" in ranking
    assert patterns["adhd"] == "ADHD"


def test_normalize_condition_uses_provided_patterns():
    patterns = {"depression": "Depression", "anxiety": "Anxiety"}
    assert _normalize_condition("Major depression", patterns) == "Depression"
    assert _normalize_condition("Anxiety disorder", patterns) == "Anxiety"
    assert _normalize_condition("Autism", patterns) is None


def test_extract_conditions_from_paper():
    patterns = {"adhd": "ADHD", "anxiety": "Anxiety"}
    paper = {
        "conditions": ["ADHD", "Generalized Anxiety"],
        "title": "Study on attention deficit",
        "abstract": "Participants with anxiety were assessed.",
    }
    found = _extract_conditions(paper, patterns)
    assert set(found) == {"ADHD", "Anxiety"}
