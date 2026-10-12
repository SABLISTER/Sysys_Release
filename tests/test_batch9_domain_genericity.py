"""Non-neuroscience domain vocabulary, stage execution, and persistence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import yaml

from core.config import Config
from core.domain_config import domain_config_snapshot
from process.domain_patterns import extraction_patterns, method_rules
from process.graph_analysis import (
    build_cooccurrence_edges, extract_entities, load_patterns, network_artifact_path,
)
from stages import s3_extract, s4_validate, s9_robustness, s10_verify


DOMAIN = {
    "co_occurring_conditions": ["drought"],
    "graph_analysis": {"entities": {"Crop yield": [r"\byield\b"]}},
    "extraction": {"heuristics": {
        "mechanisms": ["water stress"], "methodology_hints": ["field experiment"],
    }},
    "method_review": {"heuristic_rules": [{
        "study_method_type": "mixed_methods", "any_terms": ["field experiment"],
        "all_terms": ["interviews"], "confidence": 0.65,
    }]},
}


def _config(tmp_path, data=None):
    path = tmp_path / "domain.yaml"
    path.write_text(yaml.safe_dump(DOMAIN if data is None else data), encoding="utf-8")
    cfg = Config.load_config(path, project_root=tmp_path)
    cfg.llm_provider = ""  # All functional checks are offline.
    return cfg


def test_neutral_defaults_do_not_inject_neuroscience():
    text = "Autism ADHD GABA criticality amygdala network"
    assert all(not values for values in load_patterns(None).values())
    claims = s3_extract._heuristic_extract({"title": text, "abstract": ""})
    for key in ("conditions", "mechanisms", "key_entities"):
        assert claims[key] == []
    assert "supports_criticality" not in claims  # only a pack with criticality_terms emits it
    assert not claims["network_effects_described"]


def test_neuroscience_pack_is_explicit_and_independent(tmp_path):
    cfg = _config(tmp_path, {"defaults": {"pattern_pack": "neuroscience"}})
    patterns = load_patterns(cfg)
    entities = extract_entities("anxiety and amygdala", patterns)
    assert entities["conditions"] == ["Anxiety"]
    assert entities["structural_regions"] == ["Amygdala"]
    patterns["conditions"].clear()
    assert load_patterns(cfg)["conditions"]
    claims = s3_extract._heuristic_extract({"title": "ASD GABA amygdala criticality"}, cfg)
    assert "autism spectrum disorder" in claims["conditions"]
    assert "GABA" in claims["mechanisms"]
    assert claims["supports_criticality"]


def test_partial_custom_graph_has_no_neuroscience_fallback(tmp_path):
    cfg = _config(tmp_path)
    patterns = load_patterns(cfg)
    assert patterns["structural_regions"] == patterns["functional_networks"] == {}
    edges = build_cooccurrence_edges([{"title": "Drought reduces yield. ADHD amygdala"}], patterns)
    assert edges == [{"source": "drought", "target": "Crop yield", "weight": 1,
                      "source_type": "condition", "target_type": "entity"}]


def test_explicit_empty_overrides_disable_pack_categories(tmp_path):
    cfg = _config(tmp_path, {
        "defaults": {"pattern_pack": "neuroscience"},
        "co_occurring_conditions": ["anxiety"],
        "graph_analysis": {"conditions": {}, "structural_regions": {}},
        "extraction": {"heuristics": {"mechanisms": [], "brain_regions": []}},
        "method_review": {"heuristic_rules": []},
    })
    assert load_patterns(cfg)["conditions"] == {}
    claims = s3_extract._heuristic_extract({"title": "anxiety GABA amygdala"}, cfg)
    # The legacy "brain_regions" heuristics key still overrides key_entities.
    assert claims["conditions"] == claims["mechanisms"] == claims["key_entities"] == []
    assert s4_validate._heuristic_classify({"title": "randomized controlled"}, cfg)["study_method_type"] == "unknown"


@pytest.mark.parametrize("bad", [[], {"Drought": "drought"}, {"Drought": [42]}, {"Drought": ["["]}])
def test_invalid_graph_patterns_rejected_with_config_path(tmp_path, bad):
    cfg = _config(tmp_path, {"graph_analysis": {"conditions": bad}})
    with pytest.raises(ValueError, match="graph_analysis.conditions"):
        load_patterns(cfg)
    with pytest.raises(ValueError, match="graph_analysis.conditions"):
        extraction_patterns(cfg)


def test_condition_literals_escape_punctuation_and_match_whole_terms(tmp_path):
    cfg = _config(tmp_path, {"co_occurring_conditions": ["C++", "a.b", "rat"]})
    matched = extract_entities("C++ and a.b; concentration axb", load_patterns(cfg))
    assert matched["conditions"] == ["C++", "a.b"]
    cfg = _config(tmp_path, {"graph_analysis": {"conditions": {"Spaced": [" drought "]}}})
    assert extract_entities("drought", load_patterns(cfg))["conditions"] == []
    assert extract_entities(" drought ", load_patterns(cfg))["conditions"] == ["Spaced"]


@pytest.mark.parametrize("text,expected", [
    ("concentration strategy competition", "unknown"),
    ("rat experiment", "preclinical_animal"),
    ("randomised controlled trial", "randomized_controlled_trial"),
])
def test_method_defaults_use_whole_terms(text, expected):
    assert s4_validate._heuristic_classify({"title": text})["study_method_type"] == expected


@pytest.mark.parametrize("change", [
    {"confidence": 1.1}, {"confidence": float("nan")},
    {"study_method_type": "invented"}, {"any_terms": "field experiment"},
    {"any_terms": [], "all_terms": []},
])
def test_invalid_method_rules_rejected(tmp_path, change):
    data = deepcopy(DOMAIN)
    data["method_review"]["heuristic_rules"][0].update(change)
    with pytest.raises(ValueError, match="method_review.heuristic_rules"):
        method_rules(_config(tmp_path, data))


def test_non_neuroscience_stages_write_configured_results(tmp_path):
    cfg = _config(tmp_path)
    papers = [{"paper_id": "field-1", "title": "Drought and yield",
               "abstract": "A field experiment and interviews. We found water stress decreased yield."}]
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    source = cfg.claims_dir / "relevant.json"
    source.write_text(json.dumps(papers))
    s3_extract.run(cfg)
    claims_path = cfg.claims_dir / "claims.json"
    claims = json.loads(claims_path.read_text())[0]["claims"]
    assert claims["conditions"] == ["drought"]
    assert claims["mechanisms"] == ["water stress"]
    assert claims["key_entities"] == []
    assert claims["methodology"] == "field experiment"

    result = s4_validate.run(cfg, input_path=claims_path)
    profile = json.loads(Path(result["output_path"]).read_text())[0]["method_profile"]
    assert profile["study_method_type"] == "mixed_methods"
    assert profile["confidence"] == 0.65
    assert s4_validate._heuristic_classify({"title": "field experiment"}, cfg)["study_method_type"] == "unknown"

    result = s9_robustness.run(config=cfg, claims_path=str(claims_path),
                             bootstrap_iterations=4, split_iterations=4, permutation_iterations=4)
    network = json.loads(Path(result["network_path"]).read_text())
    assert Path(result["network_path"]).name == "condition_networks.json"
    assert network["edges"][0]["target"] == "Crop yield"
    assert network["edges"][0]["target_type"] == "entity"
    assert "Condition–Entity" in Path(result["network_report_path"]).read_text()
    assert not (Path(result["network_path"]).parent / "autism_condition_networks.json").exists()


def test_project_save_and_mainwindow_reopen_restore_domain_config(tmp_path):
    from gui.project_manager import ProjectManager
    from gui.main_window import MainWindow
    from gui.original_question import OriginalQuestionState

    cfg = _config(tmp_path)
    ProjectManager.save_project(tmp_path, cfg, {})
    restored = Config(project_root=tmp_path)
    window = SimpleNamespace(
        pipeline_config=restored, _original_question_state=OriginalQuestionState(),
        _set_project_path=lambda _: None, _populate_recent_menu=lambda: None,
        _restore_fulltext_project_config=lambda _: False,
        _refresh_pipeline_progress_from_disk=lambda _: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a: None),
    )
    MainWindow._open_project_path(cast(MainWindow, window), tmp_path)
    assert domain_config_snapshot(restored) == domain_config_snapshot(cfg)
    assert load_patterns(restored) == load_patterns(cfg)
    assert method_rules(restored) == method_rules(cfg)
    assert extraction_patterns(restored) == extraction_patterns(cfg)


def test_settings_export_and_reload_preserve_domain_config(tmp_path):
    from gui.dialogs import SettingsDialog
    from tests.test_config_save_no_keys import _fake_dialog

    cfg = _config(tmp_path)
    dialog = _fake_dialog(tmp_path)
    dialog.main_window.pipeline_config = cfg
    output = tmp_path / "exported.json"
    with patch("gui.dialogs.QFileDialog.getSaveFileName", return_value=(str(output), "")):
        SettingsDialog._on_save_config(cast(SettingsDialog, dialog))
    reloaded = Config.load_config(output, project_root=tmp_path)
    assert domain_config_snapshot(reloaded) == domain_config_snapshot(cfg)


def test_network_reader_prefers_new_file_and_falls_back_to_legacy(tmp_path):
    legacy = tmp_path / "autism_condition_networks.json"
    legacy.write_text("{}")
    assert network_artifact_path(tmp_path) == legacy
    current = tmp_path / "condition_networks.json"
    current.write_text("{}")
    assert network_artifact_path(tmp_path) == current


def test_legacy_module_import_remains_usable():
    from process.autism_graph_analysis import load_patterns as old_load
    assert old_load is load_patterns


@pytest.mark.parametrize("failed", [False, True])
def test_stage3_llm_supplement_and_fallback_use_custom_vocabulary(tmp_path, monkeypatch, failed):
    from process import claim_extractor

    cfg = _config(tmp_path)
    cfg.llm_provider = "ollama"
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    paper = {"title": "Drought yield water stress", "abstract": ""}
    (cfg.claims_dir / "relevant.json").write_text(json.dumps([paper]))
    monkeypatch.setattr(claim_extractor, "extract_all_claims", lambda *a, **kw: (
        [{**paper, "claims": {"error": "offline" if failed else None}}], False,
    ))
    s3_extract.run(cfg)
    claims = json.loads((cfg.claims_dir / "claims.json").read_text())[0]["claims"]
    assert claims["mechanisms"] == ["water stress"]
    assert claims["conditions"] == ["drought"]
    assert claims["extraction_method"] == ("heuristic_fallback" if failed else "llm+heuristic")


def test_stage4_llm_failure_uses_configured_rules(tmp_path, monkeypatch):
    from process import llm

    cfg = _config(tmp_path)
    cfg.llm_provider = "ollama"
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    (cfg.claims_dir / "relevant.json").write_text(json.dumps([
        {"title": "Agricultural review", "abstract": "field experiment with interviews"},
    ]))
    def unavailable(**kwargs):
        raise RuntimeError("offline test")
    monkeypatch.setattr(llm, "chat", unavailable)
    result = s4_validate.run(cfg)
    profile = json.loads(Path(result["output_path"]).read_text())[0]["method_profile"]
    assert profile["study_method_type"] == "mixed_methods"
    assert profile["confidence"] == 0.65


def test_invalid_stage3_patterns_preserve_existing_output(tmp_path):
    cfg = _config(tmp_path, {"extraction": {"heuristics": {"conditions": {"bad": ["["]}}}})
    cfg.claims_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.claims_dir / "claims.json"
    output.write_text("previous output")
    with pytest.raises(ValueError, match="extraction.heuristics.conditions.bad"):
        s3_extract.run(cfg)
    assert output.read_text() == "previous output"


def test_shipped_agricultural_template_loads_and_matches(tmp_path):
    template = Path(__file__).resolve().parents[1] / "templates" / "crop_drought.yaml"
    cfg = Config.load_config(template, project_root=tmp_path)
    claims = s3_extract._heuristic_extract({"title": "Drought yield water stress"}, cfg)
    assert claims["conditions"] == ["drought"]
    assert claims["mechanisms"] == ["water stress"]
    assert len(build_cooccurrence_edges([{"title": "drought yield"}], load_patterns(cfg))) == 1


def test_unconfigured_stage9_does_not_restore_neuroscience_defaults(tmp_path):
    cfg = _config(tmp_path, {})
    assert s9_robustness._load_condition_config(cfg) == ({}, [])
    source = tmp_path / "claims.json"
    source.write_text(json.dumps([{"title": "drought yield", "abstract": ""}]))
    result = s9_robustness.run(config=cfg, claims_path=str(source),
                             bootstrap_iterations=4, split_iterations=4, permutation_iterations=4)
    assert result["network_analysis"]["skipped"]
    assert result["condition_sensitivity"]["skipped"]
    assert "adhd_comorbidity" not in result
    assert "skipped" in Path(result["network_report_path"]).read_text()


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_settings_load_can_clear_old_domain_conditions(tmp_path, suffix):
    from gui.dialogs import SettingsDialog

    cfg = _config(tmp_path)
    cfg.co_occurring_conditions = ["ADHD"]
    source = tmp_path / f"neutral{suffix}"
    data = {"co_occurring_conditions": [], "graph_analysis": {"conditions": {}}}
    source.write_text(json.dumps(data) if suffix == ".json" else yaml.safe_dump(data))
    dialog = SimpleNamespace(
        main_window=SimpleNamespace(pipeline_config=cfg, _log=lambda *a: None),
        _has_value=SettingsDialog._has_value, _populate_from_config=lambda _: None,
    )
    with patch("gui.dialogs.QFileDialog.getOpenFileName", return_value=(str(source), "")), \
         patch("gui.settings_persistence.restore_saved_settings"), \
         patch("gui.dialogs.QMessageBox.warning") as warning:
        SettingsDialog._on_load_config(cast(SettingsDialog, dialog))
    warning.assert_not_called()
    assert cfg.co_occurring_conditions == []
    assert load_patterns(cfg)["conditions"] == {}


def test_stage9_outcome_has_no_keyword_fallback_without_config():
    paper = {"findings": ["E/I imbalance and GABAergic excitation drive criticality"]}
    # Without configured terms, domain-flavored findings must NOT count.
    assert s9_robustness.DEFAULT_OUTCOME.supports(paper) is False
    # Explicitly configured terms still drive the keyword scan.
    crit = s9_robustness.RobustnessOutcome("supports_criticality", ("GABAergic",))
    assert crit.supports(paper) is True
    assert s9_robustness.RobustnessOutcome("supports_criticality", ("dopamine",)).supports(paper) is False
    # The extracted boolean field is always honored first, nested or not.
    assert crit.supports({"supports_criticality": False, "findings": ["GABAergic"]}) is False
    assert crit.supports({"claims": {"supports_criticality": True}}) is True
    assert s9_robustness._load_outcome(None) == s9_robustness.DEFAULT_OUTCOME


def test_stage9_outcome_terms_come_from_config_or_pack_only(tmp_path):
    assert s9_robustness._load_outcome(_config(tmp_path, {})).fallback_terms == ()
    # Selecting the pack alone does not change the outcome away from the question.
    neuro = _config(tmp_path, {"defaults": {"pattern_pack": "neuroscience"}})
    assert s9_robustness._load_outcome(neuro) == s9_robustness.DEFAULT_OUTCOME
    # Opting into the legacy criticality outcome uses the pack's vocabulary.
    legacy = _config(tmp_path, {"defaults": {"pattern_pack": "neuroscience"},
                                "robustness": {"outcome_field": "supports_criticality"}})
    terms = s9_robustness._load_outcome(legacy).fallback_terms
    assert "GABA" in terms and "criticality" in terms
    custom = _config(tmp_path, {"robustness": {"outcome_field": "reports_yield_gain",
                                               "outcome_terms": ["yield gain"]}})
    assert s9_robustness._load_outcome(custom) == s9_robustness.RobustnessOutcome(
        "reports_yield_gain", ("yield gain",))


def test_stage10_verify_terms_empty_without_config():
    mechanisms, outcome, mech_context, support_context, conditions = s10_verify._verify_terms(None)
    assert (mechanisms, mech_context, conditions) == ([], [], {})
    assert outcome == s9_robustness.DEFAULT_OUTCOME
    # Only generic support wording is recognized; no domain words.
    assert support_context == ["support", "inform"]
