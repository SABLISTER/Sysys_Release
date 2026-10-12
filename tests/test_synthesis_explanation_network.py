import json
from types import SimpleNamespace

from process.synthesizer import (
    build_evidence_network,
    build_hypothesis_explanation,
    build_synthesis_source_citations,
    render_hypothesis_explanation_markdown,
    render_synthesis_source_citations_markdown,
)


def _sample_clusters():
    papers = [
        {
            "id": "p1",
            "title": "Mechanism A improves outcome B",
            "authors": ["Ada Smith", "Ben Jones"],
            "year": 2024,
            "doi": "10.1000/mechanism-a",
            "relevance_score": 5,
            "citations": 12,
            "claims": {
                "findings": ["Mechanism A is associated with outcome B"],
                "mechanisms": ["mechanism A"],
                "conditions": ["population C"],
                "brain_regions": ["region D"],
                "methodology": "randomized controlled trial",
                "supports_primary_question": True,
                "key_quote": "Mechanism A was associated with outcome B.",
            },
        },
        {
            "id": "p2",
            "title": "Mechanism A replication",
            "authors": ["Cara Lee"],
            "year": 2023,
            "pmid": "123456",
            "relevance_score": 4,
            "direction": "supports",
            "claims": {
                "findings": ["Mechanism A replicated in a second cohort"],
                "mechanisms": ["mechanism A"],
                "conditions": ["population C"],
                "methodology": "cohort observational",
                "supports_primary_question": True,
            },
        },
    ]
    return [
        {
            "cluster_id": 0,
            "papers": papers,
            "paper_count": len(papers),
            "mechanisms": ["mechanism A"],
            "conditions": ["population C"],
            "brain_regions": ["region D"],
            "primary_support_count": 2,
        }
    ]


def test_build_hypothesis_explanation_is_deterministic_and_traceable():
    explanation = build_hypothesis_explanation(
        _sample_clusters(),
        [{"cluster_id": 0, "narrative": "Mechanism A converges across studies."}],
        "The review supports mechanism A as a consensus hypothesis.",
        research_config={"natural_language_question": "Does mechanism A explain outcome B?"},
    )

    assert explanation["research_question"] == "Does mechanism A explain outcome B?"
    assert explanation["total_papers"] == 2
    assert explanation["supporting_count"] == 2
    assert explanation["clusters"][0]["top_papers"][0]["title"] == "Mechanism A improves outcome B"
    assert explanation["top_mechanisms"][0] == {"term": "mechanism A", "count": 2}
    assert "GRADE output was not available" in " ".join(explanation["caveats"])


def test_render_hypothesis_explanation_markdown_includes_core_sections():
    explanation = build_hypothesis_explanation(
        _sample_clusters(),
        [],
        "The review supports mechanism A as a consensus hypothesis.",
        research_config={"natural_language_question": "Does mechanism A explain outcome B?"},
    )
    markdown = render_hypothesis_explanation_markdown(explanation)

    assert "# Why This Hypothesis?" in markdown
    assert "## Evidence Basis" in markdown
    assert "Mechanism A improves outcome B" in markdown
    assert "## Deterministic Caveats" in markdown


def test_build_evidence_network_uses_general_schema_and_stable_ids():
    network = build_evidence_network(_sample_clusters())

    assert network["kind"] == "general_evidence_network"
    assert "nodes" in network
    assert "edges" in network

    node_ids = {node["id"] for node in network["nodes"]}
    assert "cluster:0" in node_ids
    assert "mechanism:mechanism_a" in node_ids
    assert "condition:population_c" in node_ids

    edge_types = {edge["type"] for edge in network["edges"]}
    assert "belongs_to" in edge_types
    assert "mentions" in edge_types
    assert "supports" in edge_types


def test_synthesis_source_citations_are_traceable_and_renderable():
    citations = build_synthesis_source_citations(_sample_clusters())

    assert citations["kind"] == "synthesis_source_citations"
    assert citations["source_count"] == 2
    assert citations["clusters"][0]["source_citation_keys"] == ["smith2024", "lee2023"]
    assert citations["sources"][0]["doi"] == "10.1000/mechanism-a"
    assert citations["sources"][1]["pmid"] == "123456"

    markdown = render_synthesis_source_citations_markdown(citations)

    assert "# Sources Used for Synthesis" in markdown
    assert "[smith2024] Mechanism A improves outcome B" in markdown
    assert "doi:10.1000/mechanism-a" in markdown


def test_stage6_writes_synthesis_source_citation_artifacts(tmp_path, monkeypatch):
    from stages import s6_synthesize
    import process.synthesizer as synthesizer

    papers = _sample_clusters()[0]["papers"]
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps(papers), encoding="utf-8")

    cfg = SimpleNamespace(
        llm_model="test-model",
        llm_provider="test",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        clusters_dir=tmp_path / "clusters",
        research_config={"natural_language_question": "Does mechanism A explain outcome B?"},
    )

    def ensure_dirs():
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.clusters_dir.mkdir(parents=True, exist_ok=True)

    cfg.ensure_dirs = ensure_dirs

    monkeypatch.setattr(
        synthesizer,
        "cluster_claims",
        lambda papers, **kwargs: [{
            "cluster_id": 0,
            "papers": papers,
            "paper_count": len(papers),
            "mechanisms": ["mechanism A"],
            "conditions": ["population C"],
            "brain_regions": ["region D"],
            "primary_support_count": 2,
        }],
    )
    monkeypatch.setattr(
        synthesizer,
        "generate_cluster_narrative",
        lambda **kwargs: {
            "cluster_id": 0,
            "narrative": "Mechanism A converges across the cited papers.",
            "paper_count": 2,
            "mechanisms": ["mechanism A"],
            "conditions": ["population C"],
            "brain_regions": ["region D"],
            "primary_support_count": 2,
        },
    )
    monkeypatch.setattr(
        synthesizer,
        "generate_abstract_draft",
        lambda **kwargs: "Structured abstract using the synthesis sources.",
    )

    result = s6_synthesize.run(config=cfg, claims_path=str(claims_path), n_clusters=1)

    source_json = tmp_path / "clusters" / "synthesis_sources.json"
    source_md = tmp_path / "clusters" / "synthesis_sources.md"
    source_bib = tmp_path / "clusters" / "synthesis_sources.bib"

    assert result["synthesis_sources_path"] == str(source_json)
    assert source_json.exists()
    assert source_md.exists()
    assert source_bib.exists()
    assert json.loads(source_json.read_text(encoding="utf-8"))["source_count"] == 2
    assert "[smith2024]" in source_md.read_text(encoding="utf-8")
    assert "@article{smith2024," in source_bib.read_text(encoding="utf-8")
