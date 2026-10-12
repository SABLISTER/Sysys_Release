"""Domain-genericity for hard_mode graph suggestions and scoring keystone axis.

Regression tests: hard_mode must not inject hardcoded domain vocabulary
("neuroscience" suffix) or hardcode domain axis IDs ("integration") into
logic; both are config-driven.
"""
from __future__ import annotations

from hard_mode.graph import build_keyword_suggestions
from hard_mode.scoring import derive_overlap_and_tier


def _papers(n: int = 6):
    return [
        {
            "title": f"Study {i} on yield response",
            "abstract": "Crop yield drought tolerance field trial results.",
            "human": {"reviewer_label": "relevant"},
        }
        for i in range(n)
    ]


def _hm_base(**overrides):
    hm = {
        "graph": {"min_reviewed_for_graph": 5, "top_keyword_suggestions": 12},
        "query_families": [{"label": "drought", "semantic_terms": ["drought"]}],
        "overlap_rules": {"axis_ids_for_overlap": ["axis_a", "axis_b"]},
        "tier_rules": {
            "tier1_min_axes_strong": 1,
            "axis_strong_threshold": 0.55,
            "tier1_min_integration": 0.5,
            "tier1_min_confidence": 0.45,
            "tier2_min_confidence": 0.3,
        },
    }
    for key, value in overrides.items():
        hm[key] = value
    return hm


def test_graph_suggestions_have_no_domain_suffix_by_default():
    sug = build_keyword_suggestions(_papers(), _hm_base())
    assert sug, "expected suggestions from 6 relevant papers"
    for s in sug:
        assert "neuroscience" not in s["suggestion"].lower()
        assert s["suggestion"].startswith('"')
        assert s["suggestion"].endswith('"')


def test_graph_suggestions_use_configured_suffix():
    hm = _hm_base(graph={"query_suffix": "drought"})
    sug = build_keyword_suggestions(_papers(), hm)
    assert sug
    for s in sug:
        assert s["suggestion"].endswith('drought')


def test_tier1_gating_uses_configured_keystone_axis():
    # A non-ASD config whose keystone axis is "yield_link": strong score on
    # that axis must be able to reach tier 1 (previously impossible because
    # the code hardcoded the ASD "integration" axis).
    hm = _hm_base(
        tier_rules={
            "tier1_min_axes_strong": 1,
            "axis_strong_threshold": 0.55,
            "tier1_min_integration": 0.5,
            "tier1_min_confidence": 0.45,
            "tier2_min_confidence": 0.3,
            "tier1_keystone_axis_id": "yield_link",
        }
    )
    machine = {
        "axis_scores": {"yield_link": 0.9, "axis_b": 0.9},
        "integration": 0.9,  # value the scorer would store for the keystone axis
        "confidence": 0.9,
    }
    assert derive_overlap_and_tier(hm, machine)["priority_tier"] == 1


def test_default_keystone_axis_preserves_legacy_behavior():
    # Without explicit configuration the legacy "integration" gating applies.
    hm = _hm_base()
    weak = {"axis_scores": {"axis_a": 0.9}, "integration": 0.0, "confidence": 0.9}
    assert derive_overlap_and_tier(hm, weak)["priority_tier"] != 1
    strong = {"axis_scores": {"axis_a": 0.9}, "integration": 0.9, "confidence": 0.9}
    assert derive_overlap_and_tier(hm, strong)["priority_tier"] == 1
