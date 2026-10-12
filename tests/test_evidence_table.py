from __future__ import annotations

import json
from types import SimpleNamespace

from core.models import (
    Article,
    EvidenceDirection,
    EvidenceLink,
    HypothesisComponent,
)
from process.evidence_table import (
    build_claim_evidence_rows,
    build_session_evidence_rows,
    load_evidence_rows,
)


def test_claim_artifact_rows_preserve_provenance_and_excerpt():
    rows = build_claim_evidence_rows([{
        "title": "A useful study",
        "doi": "10.1000/example",
        "pmid": "12345",
        "year": 2024,
        "authors": ["A. Author", "B. Author"],
        "source": "pubmed",
        "relevance_score": 5,
        "claims": {
            "supports_primary_question": True,
            "key_quote": "The measured effect supported the hypothesis.",
            "mechanisms": ["inhibition"],
            "brain_regions": ["cortex"],
        },
    }])

    assert len(rows) == 1
    assert rows[0]["direction"] == "supports"
    assert rows[0]["score"] == 5.0
    assert rows[0]["excerpt"] == "The measured effect supported the hypothesis."
    assert rows[0]["doi"] == "10.1000/example"
    assert rows[0]["mechanisms"] == ["inhibition"]


def test_live_analysis_rows_are_preferred_over_saved_claims(tmp_path):
    (tmp_path / "claims_filtered.json").write_text(
        json.dumps([{"title": "Saved paper", "claims": {}}]),
        encoding="utf-8",
    )
    article = Article(id="a1", title="Live paper", doi="10.1/live")
    component = HypothesisComponent(
        id="c1", label="Circuit balance", description="Circuit balance"
    )
    link = EvidenceLink(
        article_id="a1",
        component_id="c1",
        direction=EvidenceDirection.SUPPORTS,
        relevance_score=0.8,
        key_excerpts=["Live evidence excerpt."],
        matched_keywords=["balance"],
    )

    rows, source = load_evidence_rows(
        SimpleNamespace(claims_dir=tmp_path),
        session=SimpleNamespace(evidence_links=[link]),
        articles={"a1": article},
        components=[component],
    )

    assert source == "live keyword analysis"
    assert len(rows) == 1
    assert rows[0]["title"] == "Live paper"
    assert rows[0]["component"] == "Circuit balance"
    assert rows[0]["matched_keywords"] == ["balance"]


def test_saved_filtered_claims_are_loaded_when_session_has_no_links(tmp_path):
    claims_path = tmp_path / "claims_filtered.json"
    claims_path.write_text(
        json.dumps([{
            "title": "Filtered paper",
            "relevance_confidence": 0.91,
            "claims": {"findings": ["A reviewable finding."]},
        }]),
        encoding="utf-8",
    )

    rows, source = load_evidence_rows(
        SimpleNamespace(claims_dir=tmp_path),
        session=SimpleNamespace(evidence_links=[]),
    )

    assert source == str(claims_path)
    assert len(rows) == 1
    assert rows[0]["title"] == "Filtered paper"
    assert rows[0]["score"] == 0.91
    assert rows[0]["excerpt"] == "A reviewable finding."


def test_session_row_builder_accepts_dataclass_models():
    article = Article(id="a1", title="Article")
    link = EvidenceLink(article_id="a1", component_id="c1")

    rows = build_session_evidence_rows(
        [link],
        articles={"a1": article},
        components=[HypothesisComponent(
            id="c1", label="Component", description="Component"
        )],
    )

    assert rows[0]["direction"] == "neutral"
    assert rows[0]["component"] == "Component"
