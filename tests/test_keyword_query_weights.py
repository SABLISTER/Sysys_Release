"""Tests for Search-tab query term components and keyword-analysis weights."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from analysis.scoring import (
    build_hypothesis_components,
    build_search_term_components,
    build_weighted_boolean_query,
    build_weighted_component_query_terms,
    update_component_weights_from_keyword_analysis,
)
from core.models import (
    AnalysisSession,
    Article,
    EvidenceDirection,
    EvidenceLink,
    HypothesisComponent,
)


def test_build_search_term_components_keeps_phrases_and_words_in_order():
    components = build_search_term_components(
        "Does excitatory inhibitory balance in autism affect white matter tract topology?"
    )

    labels = [c.label for c in components]
    assert "excitatory inhibitory balance" in labels
    assert "white matter tract topology" in labels
    assert "excitatory inhibitory" in labels
    assert "matter tract topology" in labels
    assert "autism" in labels
    assert "topology" in labels
    assert "balance autism" not in labels
    assert [c.weight for c in components] == [1.0] * len(components)


def test_build_hypothesis_components_includes_subject_domains_and_query_terms():
    components = build_hypothesis_components(
        research_config={
            "search": {
                "base_terms": ["autism", "ASD", "autism spectrum disorder"],
                "domains": [
                    {
                        "name": "E/I imbalance + criticality",
                        "terms": ["E/I imbalance", "neural criticality"],
                        "require_also": ["GABAergic"],
                    }
                ],
            }
        },
        query="How does autism affect white matter tract topology?",
    )

    labels = [component.label for component in components]
    assert "Subject: autism / ASD / autism spectrum disorder" in labels
    assert "E/I imbalance + criticality" in labels
    assert "white" in labels
    assert "topology" in labels
    subject = next(component for component in components if component.id.startswith("subject_"))
    assert subject.keywords == ["autism", "ASD", "autism spectrum disorder"]


def test_update_component_weights_rewards_useful_terms_and_suppresses_bad_terms():
    useful = HypothesisComponent(
        id="term_autism",
        label="autism",
        description="Search term from question",
        keywords=["autism"],
    )
    absent = HypothesisComponent(
        id="term_mouse",
        label="mouse",
        description="Search term from question",
        keywords=["mouse"],
    )
    harmful = HypothesisComponent(
        id="term_placebo",
        label="placebo",
        description="Search term from question",
        keywords=["placebo"],
    )

    session = AnalysisSession(hypothesis_components=[useful, absent, harmful])
    # Only membership (article_id in session.articles) is exercised below, so
    # bare placeholders stand in for real Article instances.
    session.articles = cast("dict[str, Article]", {"a1": object(), "a2": object()})
    session.evidence_links = [
        EvidenceLink(
            article_id="a1",
            component_id="term_autism",
            direction=EvidenceDirection.SUPPORTS,
            relevance_score=0.9,
            confidence=0.9,
            matched_keywords=["autism"],
        ),
        EvidenceLink(
            article_id="a2",
            component_id="term_autism",
            direction=EvidenceDirection.TANGENTIAL,
            relevance_score=0.6,
            confidence=0.7,
            matched_keywords=["autism"],
        ),
        EvidenceLink(
            article_id="a1",
            component_id="term_mouse",
            direction=EvidenceDirection.NEUTRAL,
            relevance_score=0.0,
            confidence=0.0,
            matched_keywords=[],
        ),
        EvidenceLink(
            article_id="a2",
            component_id="term_mouse",
            direction=EvidenceDirection.NEUTRAL,
            relevance_score=0.0,
            confidence=0.0,
            matched_keywords=[],
        ),
        EvidenceLink(
            article_id="a1",
            component_id="term_placebo",
            direction=EvidenceDirection.CONTRADICTS,
            relevance_score=0.8,
            confidence=0.8,
            matched_keywords=["placebo"],
        ),
        EvidenceLink(
            article_id="a2",
            component_id="term_placebo",
            direction=EvidenceDirection.CONTRADICTS,
            relevance_score=0.7,
            confidence=0.7,
            matched_keywords=["placebo"],
        ),
    ]

    stats = update_component_weights_from_keyword_analysis(session)

    assert useful.weight > 1.0
    assert absent.weight < 1.0
    assert harmful.weight < absent.weight
    assert stats["term_autism"]["coverage"] == 1.0
    assert "Keyword analysis:" in useful.description


def test_build_weighted_component_query_terms_orders_by_weight_and_skips_zeroes():
    components = [
        HypothesisComponent("term_noise", "noise", "", ["noise"], weight=0.0),
        HypothesisComponent("term_autism", "autism", "", ["autism"], weight=2.2),
        HypothesisComponent("term_cortex", "cortex", "", ["cortex"], weight=0.8),
    ]

    terms = build_weighted_component_query_terms(components)

    assert terms == ["autism", "cortex"]


def test_build_weighted_component_query_terms_uses_keywords_before_ui_labels():
    components = [
        HypothesisComponent(
            "subject_autism",
            "Subject: autism / ASD",
            "",
            ["autism", "ASD"],
            weight=1.5,
        )
    ]

    terms = build_weighted_component_query_terms(components)

    assert terms == ["autism", "ASD"]


def test_build_weighted_boolean_query_groups_terms_by_weight():
    components = [
        HypothesisComponent("core", "Core", "", ["autism"], weight=1.6),
        HypothesisComponent("mechanism", "Mechanism", "", ["E/I imbalance"], weight=1.4),
        HypothesisComponent("support", "Support", "", ["white matter"], weight=0.9),
        HypothesisComponent("low", "Low", "", ["placebo"], weight=0.2),
    ]

    query = build_weighted_boolean_query(components)

    assert "autism AND \"E/I imbalance\"" in query
    assert '("white matter")' in query
    assert "placebo" not in query
