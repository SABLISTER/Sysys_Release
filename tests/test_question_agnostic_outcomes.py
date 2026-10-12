"""Robustness/verification read real Stage 3 output for any research question."""
from __future__ import annotations

import json
from types import SimpleNamespace

from core.claims import claim_flag, claim_list, claim_value
from stages import s9_robustness, s10_verify


def _nested(title: str, supports: bool, conditions: list[str] | None = None) -> dict:
    """The shape Stage 3/5 actually write: extracted fields under ``claims``."""
    return {"title": title, "abstract": "", "relevance_score": 4,
            "claims": {"supports_primary_question": supports,
                       "conditions": conditions or [], "findings": [title]}}


def test_claim_helpers_read_nested_top_level_and_legacy_names():
    paper = {"claims": {"brain_regions": ["PFC"], "supports_primary_question": "yes"}}
    assert claim_list(paper, "key_entities") == ["PFC"]
    assert claim_flag(paper) is True
    assert claim_value({"key_entities": ["x"]}, "key_entities") == ["x"]
    assert claim_flag({}) is False


def test_stage9_support_rate_counts_nested_claims():
    # Regression: Stage 9 read only top-level fields, so real runs scored 0%.
    papers = [_nested("a", True), _nested("b", True), _nested("c", False)]
    result = s9_robustness.sensitivity_analysis(papers)
    assert result["thresholds"][1]["pct_supporting"] == round(200 / 3, 2)


def test_stage9_conditions_match_whole_terms_from_nested_claims():
    patterns = {"rett": "Rett"}
    assert s9_robustness._extract_conditions(
        {"title": "Barrett esophagus", "abstract": ""}, patterns) == []
    assert s9_robustness._extract_conditions(
        _nested("x", True, ["Rett syndrome"]), patterns) == ["Rett"]


def test_stage10_counts_configured_conditions_without_crashing():
    # Regression: condition regex lists reached a set() and raised TypeError.
    cfg = SimpleNamespace(research_config={}, co_occurring_conditions=["drought", "frost"])
    papers = [_nested("drought yields", True), _nested("frost damage", False)]
    result = s10_verify.verify(papers, "We found 2 conditions; 50% support the question.",
                               config=cfg)
    checks = {c["claim"]: c for c in result["checks"]}
    assert checks["2 conditions"]["status"] == "PASS"
    assert checks["50%"]["status"] == "PASS"
    assert "support (supports_primary_question)" in checks["50%"]["note"]


def test_stage9_run_reports_configured_outcome(tmp_path):
    from core.config import Config
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"robustness": {"outcome_field": "reports_gain"}}))
    cfg = Config.load_config(path, project_root=tmp_path)
    source = tmp_path / "claims.json"
    source.write_text(json.dumps([
        {"title": "t", "abstract": "", "relevance_score": 3, "claims": {"reports_gain": True}},
    ]))
    result = s9_robustness.run(config=cfg, claims_path=str(source), bootstrap_iterations=4,
                               split_iterations=4, permutation_iterations=4)
    assert result["outcome_field"] == "reports_gain"
    assert result["sensitivity"]["thresholds"][1]["pct_supporting"] == 100.0
    report = open(result["report_path"]).read()
    assert "`reports_gain`" in report and "riticality" not in report


def test_extraction_prompt_without_config_is_topic_neutral():
    from process.claim_extractor import build_extraction_prompt
    from process.synthesizer import NARRATIVE_PROMPT

    prompt = build_extraction_prompt("Title", "Abstract text.")
    assert "key_entities" in prompt
    for word in ("brain", "criticality", "comorbid"):
        assert word not in prompt.lower()
    assert "brain" not in NARRATIVE_PROMPT.lower()


def test_heuristic_merge_keeps_llm_values_stored_under_legacy_name():
    from stages.s3_extract import _merge_heuristic_into_claims

    llm_claims = {"brain_regions": ["amygdala"], "findings": ["x"]}
    merged = _merge_heuristic_into_claims(llm_claims, {"key_entities": ["other"]})
    assert claim_list(merged, "key_entities") == ["amygdala"]


def test_evidence_direction_ignores_topic_specific_flags():
    from process.evidence_table import _paper_direction

    assert _paper_direction({"claims": {"supports_criticality": True}}) == "neutral"
    assert _paper_direction({"claims": {"supports_primary_question": True}}) == "supports"


def test_interrater_compares_nested_original_claims_and_old_checkpoints():
    from stages.s8_interrater import _record_values

    assert _record_values({"original_key_entities": ["a"]}, "original", "key_entities") == ["a"]
    assert _record_values({"second_brain_regions": ["b"]}, "second", "key_entities") == ["b"]


def test_compare_runs_counts_configured_outcome_from_nested_claims():
    from pathlib import Path
    from stages.s11_compare_runs import _run_summary

    records = [_nested("a", True), _nested("b", False), {"title": "c"}]
    summary = _run_summary(Path("run.json"), records)
    assert (summary["support_count"], summary["support_total"]) == (1, 2)
    assert summary["outcome_field"] == "supports_primary_question"


def _cfg(tmp_path, data: dict):
    import yaml
    from core.config import Config

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return Config.load_config(path, project_root=tmp_path)


def test_default_method_rules_have_no_neuroimaging_and_pack_rules_come_first(tmp_path):
    from stages.s4_validate import _heuristic_classify

    pet_study = {"title": "Pet ownership and loneliness: a cohort study"}
    general = _cfg(tmp_path, {})
    assert _heuristic_classify(pet_study, general)["study_method_type"] == "cohort_observational"
    assert _heuristic_classify({"title": "fMRI of reward"}, general)["study_method_type"] == "unknown"

    neuro = _cfg(tmp_path, {"defaults": {"pattern_pack": "neuroscience"}})
    assert _heuristic_classify({"title": "fMRI of reward"}, neuro)["study_method_type"] == "imaging_ml_classifier"
    # General rules still apply under the pack.
    assert _heuristic_classify(pet_study, neuro)["study_method_type"] == "cohort_observational"


def test_blank_template_and_committee_defaults_are_topic_neutral():
    from pathlib import Path
    from process.committee_brief import DEFAULT_COMMITTEE_BRIEF, DEFAULT_INFOSHEET

    root = Path(__file__).resolve().parents[1]
    blank = (root / "templates" / "blank_template.yaml").read_text().lower()
    defaults = repr((DEFAULT_COMMITTEE_BRIEF, DEFAULT_INFOSHEET)).lower()
    for word in ("brain", "criticality", "comorbid", "abct", "mental health", "clinical"):
        assert word not in blank
        assert word not in defaults


def test_hard_mode_suggestions_do_not_append_a_field_name():
    from hard_mode.graph import build_keyword_suggestions

    papers = [{"title": "salinity tolerance", "abstract": "", "human": {"reviewer_label": "relevant"}}] * 6
    suggestions = build_keyword_suggestions(papers, {"graph": {"min_reviewed_for_graph": 5}})
    assert suggestions and all(s["suggestion"] == f'"{s["term"]}"' for s in suggestions)


def test_sample_sizes_recognize_non_biomedical_units():
    from process.statistical_extractor import parse_subjects

    found = parse_subjects("We surveyed 412 households across 38 schools.")
    labels = {(s.get("n"), s.get("label")) for s in found}
    assert (412, "households") in labels and (38, "schools") in labels
