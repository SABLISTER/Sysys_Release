"""Focused checks for the review issues tracked in docs/issue-tracker.md."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from analysis.scoring import build_hypothesis_components
from acquire.deduplicator import deduplicate_papers
from core.config import Config
from hard_mode.dedupe import dedupe_hard_mode
from hard_mode.paths import hard_mode_run_dir
from process.coordinate_extractor import BrainCoordinate, convert_tal_to_mni
from process.export import export_report_bundle
from process.prisma import collect_prisma_data


def _prisma_config(tmp_path: Path) -> Config:
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    (cfg.data_dir / "fulltext").mkdir(parents=True, exist_ok=True)
    return cfg


def test_prisma_counts_unretrieved_fulltext(tmp_path):
    cfg = _prisma_config(tmp_path)
    log = cfg.data_dir / "fulltext" / "download_log.json"
    log.write_text(
        '[{"status": "ok", "file": "a.txt"}, {"status": "failed"}, {"status": "no-text"}]',
        encoding="utf-8",
    )

    data = collect_prisma_data(cfg)

    assert data.reports_sought_fulltext == 3
    assert data.reports_not_retrieved == 2
    assert data.reports_assessed_eligibility == 1


def test_prisma_not_retrieved_stays_zero_without_a_download_log(tmp_path):
    cfg = _prisma_config(tmp_path)
    data = collect_prisma_data(cfg)
    assert data.reports_not_retrieved == 0


def test_export_bundle_keeps_both_abstracts(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    (cfg.output_dir / "synthesis_abstract.md").write_text("pipeline abstract\n", encoding="utf-8")
    (cfg.clusters_dir / "abstract_draft.md").write_text("cluster draft\n", encoding="utf-8")

    bundle = export_report_bundle(cfg, tmp_path / "bundle")

    assert (bundle / "synthesis_abstract.md").read_text(encoding="utf-8") == "pipeline abstract\n"
    assert (bundle / "abstract_draft.md").read_text(encoding="utf-8") == "cluster draft\n"


def test_tal2mni_applies_lancaster_translation():
    converted = convert_tal_to_mni([BrainCoordinate(x=0, y=0, z=0, space="Talairach")])
    assert converted[0].space == "MNI"
    assert converted[0].x == 1.0
    assert converted[0].y == 1.5
    assert converted[0].z == -4.7

    unchanged = convert_tal_to_mni([BrainCoordinate(x=4, y=5, z=6, space="MNI")])
    assert (unchanged[0].x, unchanged[0].y, unchanged[0].z) == (4, 5, 6)


def test_cooccurring_condition_keywords_are_not_hardcoded_to_autism():
    components = build_hypothesis_components(co_occurring_conditions=["ADHD"])
    condition = next(component for component in components if component.label == "ADHD")
    assert condition.keywords == ["ADHD"]
    assert "autism" not in condition.keywords


def test_cooccurring_conditions_carry_no_domain_vocabulary_into_matching_or_search():
    from analysis.scoring import build_weighted_boolean_query, score_article_against_component
    from core.models import Article

    components = build_hypothesis_components(co_occurring_conditions=["drought", "soil salinity"])
    conditions = [c for c in components if c.id.startswith("condition_")]
    # Only the configured term itself identifies a condition component.
    assert [c.keywords for c in conditions] == [["drought"], ["soil salinity"]]

    # A generic relationship word must not credit every condition.
    article = Article(id="a1", title="", abstract="Comorbid and co-occurring factors were studied.")
    for component in conditions:
        assert score_article_against_component(article, component).matched_keywords == []

    # ...nor be injected into the provider search query.
    query = build_weighted_boolean_query(components).lower()
    assert "comorbid" not in query and "co-occurring" not in query
    assert "drought" in query


def test_fuzzy_dedup_still_merges_blocked_near_duplicates():
    papers = [
        {
            "title": "Cortical thickness in temporal lobe epilepsy",
            "year": 2020,
            "authors": ["Smith J"],
            "abstract": "One",
            "source": "pubmed",
        },
        {
            "title": "Cortical thickness in temporal lobe epilepsy",
            "year": 2020,
            "authors": ["Smith J"],
            "doi": "10.1000/thickness",
            "abstract": "One",
            "source": "openalex",
        },
        {
            "title": "Monetary policy and inflation expectations",
            "year": 2020,
            "authors": ["Smith J"],
            "abstract": "Other",
            "source": "s2",
        },
    ]
    deduped = deduplicate_papers(papers)
    titles = {paper["title"] for paper in deduped}
    assert len(deduped) == 2
    assert "10.1000/thickness" in {paper.get("doi") for paper in deduped}
    assert "Monetary policy and inflation expectations" in titles


def test_hard_mode_merge_keeps_nested_scores():
    papers = [
        {
            "doi": "10.1000/nested",
            "title": "Nested scores",
            "machine": {"axis_scores": {"relevance": 0.2}, "note": "first"},
        },
        {
            "doi": "10.1000/nested",
            "title": "Nested scores",
            "machine": {"axis_scores": {"quality": 0.8}, "extra": "second"},
        },
    ]
    merged = dedupe_hard_mode(papers)
    assert len(merged) == 1
    scores = merged[0]["machine"]["axis_scores"]
    assert scores["relevance"] == 0.2
    assert scores["quality"] == 0.8
    assert merged[0]["machine"]["note"] == "first"
    assert merged[0]["machine"]["extra"] == "second"


def test_run_id_rejects_path_traversal(tmp_path):
    cfg = Config(project_root=tmp_path)
    with pytest.raises(ValueError):
        hard_mode_run_dir(cfg, "../outside")
    safe = hard_mode_run_dir(cfg, "run-01")
    assert safe.name == "run-01"


def test_provider_failure_does_not_abort_the_family(tmp_path, monkeypatch):
    import hard_mode.ingestion as ingestion

    class Boom:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            raise RuntimeError("provider down")

        async def close(self):
            return None

    class Ok:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return [{"title": "Kept paper", "abstract": "text"}]

        async def close(self):
            return None

    monkeypatch.setattr(ingestion, "SemanticScholarClient", Boom)
    monkeypatch.setattr(ingestion, "OpenAlexClient", Ok)
    monkeypatch.setattr(ingestion, "_normalize_oa_paper", lambda paper: paper)

    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()
    hm = {
        "query_families": [
            {
                "id": "family",
                "label": "Family",
                "providers": ["s2", "openalex"],
                "semantic_terms": ["cortex"],
            }
        ]
    }

    papers = asyncio.run(ingestion.run_family_retrieval(cfg, hm, "run-01"))
    assert any(paper.get("title") == "Kept paper" for paper in papers)


def test_acquisition_queries_providers_together(tmp_path, monkeypatch):
    import acquire.orchestrator as orch

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return []

        async def close(self):
            return None

        async def get_citations(self, *args, **kwargs):
            return []

        async def get_references(self, *args, **kwargs):
            return []

    for name in (
        "ElicitClient",
        "SemanticScholarClient",
        "OpenAlexClient",
        "PubMedClient",
        "WebOfScienceClient",
        "ScopusClient",
    ):
        monkeypatch.setattr(orch, name, Stub)

    seen: dict[str, int] = {}
    real_gather = asyncio.gather

    async def spy(*args, **kwargs):
        seen["count"] = len(args)
        return await real_gather(*args, **kwargs)

    monkeypatch.setattr(orch.asyncio, "gather", spy)
    cfg = Config(project_root=tmp_path)
    cfg.elicit_api_key = "token"
    cfg.wos_api_key = "token"
    cfg.scopus_api_key = "token"
    cfg.citation_snowball_enabled = False

    asyncio.run(
        orch.run_acquisition(
            cfg,
            search_config={"base_terms": ["cortex"], "domains": []},
            natural_language_query="cortical thickness",
        )
    )
    assert seen["count"] == 6


def _acquisition_config(tmp_path, **keys):
    cfg = Config(project_root=tmp_path)
    cfg.elicit_api_key = keys.get("elicit_api_key", "")
    cfg.wos_api_key = keys.get("wos_api_key", "")
    cfg.scopus_api_key = keys.get("scopus_api_key", "")
    cfg.citation_snowball_enabled = False
    return cfg


def _stub_client(search_impl):
    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        async def search(self, *args, **kwargs):
            return await search_impl(*args, **kwargs)

        async def close(self):
            return None

    return Stub


def _run_with_stubs(monkeypatch, cfg, search_impl):
    import acquire.orchestrator as orch

    stub = _stub_client(search_impl)
    for name in (
        "ElicitClient",
        "SemanticScholarClient",
        "OpenAlexClient",
        "PubMedClient",
        "WebOfScienceClient",
        "ScopusClient",
    ):
        monkeypatch.setattr(orch, name, stub)
    return asyncio.run(
        orch.run_acquisition(
            cfg,
            search_config={"base_terms": ["cortex"], "domains": []},
            natural_language_query="",
        )
    )


def test_acquisition_raises_when_every_attempted_provider_fails(tmp_path, monkeypatch):
    async def boom(*args, **kwargs):
        raise ConnectionError("provider down")

    cfg = _acquisition_config(tmp_path)
    with pytest.raises(RuntimeError, match="All 3 acquisition providers failed"):
        _run_with_stubs(monkeypatch, cfg, boom)


def test_acquisition_allows_legitimate_zero_results(tmp_path, monkeypatch):
    async def empty(*args, **kwargs):
        return []

    cfg = _acquisition_config(tmp_path)
    corpus = _run_with_stubs(monkeypatch, cfg, empty)
    assert corpus == []


def test_acquisition_tolerates_partial_provider_failure(tmp_path, monkeypatch):
    import acquire.orchestrator as orch

    async def boom(*args, **kwargs):
        raise ConnectionError("s2 down")

    async def empty(*args, **kwargs):
        return []

    monkeypatch.setattr(orch, "SemanticScholarClient", _stub_client(boom))
    for name in ("OpenAlexClient", "PubMedClient"):
        monkeypatch.setattr(orch, name, _stub_client(empty))

    cfg = _acquisition_config(tmp_path)
    corpus = asyncio.run(
        orch.run_acquisition(
            cfg,
            search_config={"base_terms": ["cortex"], "domains": []},
            natural_language_query="",
        )
    )
    assert corpus == []
