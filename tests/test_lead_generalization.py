from pathlib import Path

import numpy as np

from core.config import Config
from core.research_components import build_hypothesis_components_from_config
from hard_mode.load_config import load_hard_mode_config
from process.claim_extractor import build_extraction_prompt
from process.queries import (
    get_natural_language_query,
    get_pubmed_queries,
    get_s2_queries,
)
from process import synthesizer
from process.synthesizer import ABSTRACT_PROMPT, NARRATIVE_PROMPT


ROOT = Path(__file__).resolve().parent.parent


def _joined(values) -> str:
    return " ".join(str(value) for value in values).lower()


def test_active_config_loads_and_uses_asd_ei_search():
    cfg = Config.load_config(project_root=ROOT)

    assert cfg.run_name == "asd-ei-imbalance"
    search = cfg.get_search_config()
    assert "autism" in _joined(search["base_terms"])
    assert any("domain" in str(d) or "name" in str(d) for d in search.get("domains", []))


def test_active_queries_use_asd_ei_terms():
    cfg = Config.load_config(project_root=ROOT)
    search = cfg.get_search_config()
    queries = [*get_s2_queries(search), *get_pubmed_queries(search)]
    joined = _joined(queries)

    assert "autism" in joined
    assert "excitatory inhibitory" in joined or "e/i" in joined or "gaba" in joined
    assert "criticality" in joined or "network" in joined


def test_natural_language_query_uses_relevance_research_question():
    cfg = Config.load_config(project_root=ROOT)

    query = get_natural_language_query(cfg.research_config)

    assert len(query) > 20
    assert "autism" in query.lower() or "e/i" in query.lower() or "excitatory" in query.lower()


def test_hard_mode_derives_asd_query_families_and_axes():
    hard_mode = load_hard_mode_config(project_root=ROOT)

    assert hard_mode["name"] == "asd-ei-imbalance"
    assert len(hard_mode["query_families"]) == len(hard_mode["search"]["domains"])
    first_family = hard_mode["query_families"][0]
    assert any(t in first_family["semantic_terms"] for t in ("autism", "ASD", "autism spectrum"))
    assert [axis["id"] for axis in hard_mode["axes"][:4]] == [
        "ei_imbalance",
        "asd_population",
        "criticality_network",
        "comorbidity_link",
    ]


def test_extraction_prompt_is_configured_for_asd_ei():
    cfg = Config.load_config(project_root=ROOT)

    prompt = build_extraction_prompt(
        "E/I imbalance in autism paper",
        "GABAergic dysfunction observed in ASD participants.",
        cfg.research_config,
    )

    assert "supports_primary_question" in prompt
    assert "lead" not in prompt.lower() or "child cognitive outcomes" not in prompt


def test_synthesis_prompts_are_configured_for_asd_ei():
    cfg = Config.load_config(project_root=ROOT)
    research_question = get_natural_language_query(cfg.research_config)

    narrative = NARRATIVE_PROMPT.format(
        research_question=research_question,
        paper_count=2,
        mechanisms="GABAergic dysfunction",
        conditions="ASD",
        entities="cortex",
        primary_support_pct=75,
        paper_summaries="- Example: Findings: E/I imbalance found in ASD",
    )
    abstract = ABSTRACT_PROMPT.format(
        research_question=research_question,
        total_papers=2,
        cluster_summaries="Cluster 1: E/I imbalance in autism",
    )

    assert "configured research question" in narrative
    assert research_question in narrative
    assert research_question in abstract


def test_cluster_claims_reads_primary_support_from_nested_claims(monkeypatch):
    papers = [
        {
            "title": "Executive function in children",
            "abstract": "Children completed executive function tasks.",
            "claims": {
                "mechanisms": ["executive function"],
                "conditions": ["children"],
                "supports_primary_question": True,
            },
        },
        {
            "title": "Adult memory study",
            "abstract": "Adults completed memory tasks.",
            "claims": {"mechanisms": ["memory"]},
        },
    ]

    monkeypatch.setattr(
        synthesizer,
        "_get_embeddings",
        lambda papers, model_name="all-MiniLM-L6-v2": np.ones((len(papers), 2)),
    )

    clusters = synthesizer.cluster_claims(papers, n_clusters=1)

    assert clusters[0]["primary_support_count"] == 1
    assert "executive function" in clusters[0]["mechanisms"]


def test_component_builder_uses_asd_ei_terms():
    cfg = Config.load_config(project_root=ROOT)

    components = build_hypothesis_components_from_config(cfg)
    labels = [component.label for component in components]
    keywords = _joined(keyword for component in components for keyword in component.keywords)

    assert "Research Question" in labels
    assert any(
        kw in keywords
        for kw in ("autism", "asd", "e/i", "excitatory", "inhibitory", "gaba", "criticality")
    )


def test_archived_autism_template_uses_current_search_schema():
    cfg = Config.load_config(
        config_path=ROOT / "templates" / "autism_ei_balance.yaml",
        project_root=ROOT,
    )

    search = cfg.get_search_config()

    assert search["base_terms"][0] == "autism"
    assert "domains" in search
    assert "base_term_s2" not in search


def test_abct_ai_digital_mental_health_template_loads_focused_defaults():
    cfg = Config.load_config(
        config_path=ROOT / "templates" / "abct_ai_digital_mental_health.yaml",
        project_root=ROOT,
    )

    assert cfg.run_name == "abct-ai-digital-mental-health-brief"
    assert cfg.research_config["committee_brief"]["meeting_date"] == "2026-06-30"
    assert cfg.get_synthesis_config()["audience"] == "ABCT Technology Committee"

    search = cfg.get_search_config()
    queries = [*get_s2_queries(search), *get_pubmed_queries(search)]
    joined = _joined(queries)

    assert "digital mental health" in joined
    assert "large language model" in joined
    assert "clinical decision support" in joined
    assert "algorithmic bias" in joined
    assert "mental health" in joined


def test_abct_tech_ai_infosheet_template_loads_june_22_defaults():
    cfg = Config.load_config(
        config_path=ROOT / "templates" / "abct_tech_ai_infosheet.yaml",
        project_root=ROOT,
    )

    assert cfg.run_name == "abct-tech-ai-infosheet"
    assert cfg.research_config["abct_tech_ai_infosheet"]["meeting_date"] == "2026-06-22"
    assert cfg.research_config["committee_brief"]["meeting_date"] == "2026-06-22"
    assert cfg.get_synthesis_config()["audience"] == "ABCT Technology Committee AI Subcommittee"

    search = cfg.get_search_config()
    queries = [*get_s2_queries(search), *get_pubmed_queries(search)]
    joined = _joined(queries)

    assert "digital mental health" in joined
    assert "informed consent" in joined
    assert "clinician oversight" in joined
    assert "crisis escalation" in joined
    assert "digital divide" in joined


def test_extraction_schema_is_config_driven():
    from process.claim_extractor import (
        _field_specs,
        _parse_extraction_response,
        build_extraction_prompt,
    )

    rc = {"extraction": {"fields": [
        {"name": "yield_effect", "type": "string",
         "description": "effect on crop yield"},
        {"name": "supports_drought_hypothesis", "type": "boolean",
         "description": "whether findings support the drought hypothesis"},
        {"name": "design", "type": "string", "description": "study design"},
    ]}}
    specs = _field_specs(rc)
    assert [name for name, _, _ in specs] == [
        "yield_effect", "supports_drought_hypothesis", "design"]

    prompt = build_extraction_prompt("Drought trial", "Yield fell.", rc)
    assert "yield_effect" in prompt
    assert "supports_criticality" not in prompt

    result = _parse_extraction_response(
        '{"yield_effect": "decreased", "supports_drought_hypothesis": true,'
        ' "design": "field experiment"}',
        specs,
    )
    assert result["yield_effect"] == "decreased"
    assert result["supports_drought_hypothesis"] is True
    assert result["design"] == "field experiment"
    assert "supports_criticality" not in result


def test_extraction_schema_falls_back_to_legacy_without_config():
    from process.claim_extractor import _LEGACY_FIELDS, _field_specs

    assert _field_specs(None) == _LEGACY_FIELDS
    assert _field_specs({}) == _LEGACY_FIELDS
    assert _field_specs({"extraction": {"fields": []}}) == _LEGACY_FIELDS
    names = [name for name, _, _ in _field_specs(None)]
    # The no-config schema is topic-neutral: no criticality or brain-region fields.
    assert "key_entities" in names and "supports_primary_question" in names
    assert not {"supports_criticality", "brain_regions", "comorbidity_link"} & set(names)
