"""Tests for deterministic statistical evidence extraction."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from process.statistical_extractor import (
    apply_llm_context_reviews,
    extract_article,
    extract_statistical_evidence,
    metrics_for_meta_analysis,
    parse_llm_context_response,
    repair_decimal_spaced_result_numbers,
)
from process.meta_analysis import load_effect_sizes
from stages import s2c_statistical_extract


def test_extract_article_captures_complete_t_test_with_subjects():
    paper = {
        "title": "Treatment effects in adults",
        "doi": "10.1/example",
        "abstract": (
            "The trial included 84 adults. Treatment vs control improved scores "
            "with t(82) = 2.41, p = .018, Cohen's d = 0.52, 95% CI 0.10 to 0.94."
        ),
    }

    record = extract_article(paper)

    assert record["article_id"] == "10.1/example"
    assert len(record["results"]) >= 2
    t_result = next(r for r in record["results"] if r["test_type"] == "t_test")
    assert t_result["statistic"] == 2.41
    assert t_result["p_values"][0]["value"] == 0.018
    assert t_result["sample_sizes"][0]["n"] == 84
    assert "Treatment vs control" in t_result["context"]
    assert t_result["location"]["text_source"] == "abstract"


def test_extract_article_handles_pdf_spaced_decimals():
    paper = {
        "title": "PDF spacing artifacts",
        "abstract": (
            "The sample included 42 children. Scores were associated with the "
            "outcome (r = 0. 48, p < 0. 001). A model coefficient was "
            "negative (coefficient = \u22120. 02, p = 0. 04)."
        ),
    }

    record = extract_article(paper)

    correlation = next(r for r in record["results"] if r["test_type"] == "correlation")
    regression = next(r for r in record["results"] if r["test_type"] == "regression_coefficient")
    assert correlation["statistic"] == 0.48
    assert correlation["p_values"][0]["value"] == 0.001
    assert regression["statistic"] == -0.02
    assert any(p_value["value"] == 0.04 for p_value in regression["p_values"])


def test_repair_decimal_spaced_result_numbers_for_legacy_review_rows():
    legacy_result = {
        "test_type": "correlation",
        "test_text": "r = 0",
        "statistic": 0.0,
        "p_values": [{"operator": "=", "value": 0.0, "text": "p = 0"}],
        "context": "The study reported a reliable association (r = 0. 59, p = 0. 003).",
    }

    repaired = repair_decimal_spaced_result_numbers(legacy_result)

    assert repaired["statistic"] == 0.59
    assert repaired["p_values"][0]["value"] == 0.003
    assert legacy_result["statistic"] == 0.0


def test_incomplete_statistical_result_requires_human_review_context():
    paper = {
        "title": "Incomplete result",
        "abstract": "Participants improved after treatment, F(1, 32) = 4.9.",
    }

    record = extract_article(paper)

    assert record["needs_human_review"] is True
    assert record["human_review"]
    assert "Participants improved" in record["human_review"][0]["context"]
    result = record["results"][0]
    assert "p_value_or_ci" in result["missing_fields"]
    assert "effect_size" in result["missing_fields"]


def test_no_parseable_statistic_flags_likely_result_paragraph():
    paper = {
        "title": "Narrative result",
        "abstract": "Results showed a significant increase in symptoms among patients.",
    }

    record = extract_article(paper)

    assert not record["results"]
    assert record["human_review"][0]["reason"] in {
        "result_context_without_parseable_statistic",
        "no_statistical_result_found",
    }
    assert "significant increase" in record["human_review"][0]["context"]


def test_metrics_for_meta_analysis_flattens_effect_numbers():
    papers = [
        {
            "title": "Effect paper",
            "abstract": (
                "The sample included 50 participants. Group A versus Group B "
                "showed d = 0.42, p < .05, 95% CI 0.02, 0.82."
            ),
        }
    ]
    extracted = extract_statistical_evidence(papers)

    metrics = metrics_for_meta_analysis(extracted["records"])

    assert len(metrics) == 1
    assert metrics[0]["effect_sizes"] == [0.42]
    assert metrics[0]["sample_sizes"] == [50]
    assert metrics[0]["confidence_intervals"] == [[0.02, 0.82]]
    assert metrics[0]["p_values"] == [0.05]


def test_stage_writes_reviewable_artifacts(tmp_path: Path):
    claims_dir = tmp_path / "data" / "claims"
    corpus_dir = tmp_path / "data" / "corpus"
    data_dir = tmp_path / "data"
    claims_dir.mkdir(parents=True)
    corpus_dir.mkdir(parents=True)
    relevant_path = claims_dir / "relevant.json"
    relevant_path.write_text(
        json.dumps([
            {
                "title": "Writable effect",
                "abstract": "Included 40 patients; r = 0.31, p = .04.",
            }
        ]),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        claims_dir=claims_dir,
        corpus_dir=corpus_dir,
        data_dir=data_dir,
    )

    result = s2c_statistical_extract.run(config=cfg)

    output_path = Path(result["output_path"])
    review_path = Path(result["review_path"])
    meta_analysis_input_path = Path(result["meta_analysis_input_path"])
    metrics_path = Path(result["metrics_path"])
    assert output_path.exists()
    assert review_path.exists()
    assert meta_analysis_input_path.exists()
    assert meta_analysis_input_path.name == "statistical_meta_analysis_input.json"
    assert metrics_path.exists()
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["results"] == 1
    assert artifact["records"][0]["results"][0]["context"]


def test_meta_analysis_loader_reads_stage2c_input_by_default(tmp_path: Path):
    claims_dir = tmp_path / "data" / "claims"
    output_dir = tmp_path / "output"
    claims_dir.mkdir(parents=True)
    output_dir.mkdir()
    (claims_dir / "statistical_meta_analysis_input.json").write_text(
        json.dumps([
            {
                "article_id": "a1",
                "title": "Meta ready",
                "effect_sizes": [0.5],
                "sample_sizes": [100],
                "confidence_intervals": [[0.1, 0.9]],
            }
        ]),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        claims_dir=claims_dir,
        data_dir=tmp_path / "data",
        output_dir=output_dir,
    )

    studies = load_effect_sizes(cfg)

    assert len(studies) == 1
    assert studies[0].article_id == "a1"
    assert studies[0].sample_size == 100
    assert studies[0].effect_size != 0.0


def test_meta_analysis_loader_requires_explicit_stage7_selection(tmp_path: Path):
    data_dir = tmp_path / "data"
    claims_dir = data_dir / "claims"
    output_dir = tmp_path / "output"
    metrics_dir = data_dir / "fulltext"
    claims_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    output_dir.mkdir()
    (metrics_dir / "auto_extracted_metrics.json").write_text(
        json.dumps([
            {
                "doi": "10.1/stage7",
                "title": "Stage 7 nested",
                "metrics": {
                    "effect_sizes": [{"value": 0.5}],
                    "sample_sizes": [{"value": 100}],
                    "confidence_intervals": [{"lower": 0.1, "upper": 0.9}],
                },
            }
        ]),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(claims_dir=claims_dir, data_dir=data_dir, output_dir=output_dir)

    assert load_effect_sizes(cfg) == []

    studies = load_effect_sizes(cfg, input_source="stage7")

    assert len(studies) == 1
    assert studies[0].article_id == "10.1/stage7"
    assert studies[0].sample_size == 100
    assert studies[0].effect_size != 0.0


def test_parse_llm_context_response_normalizes_review_json():
    response = """
    ```json
    {
      "represents_study_data": true,
      "usable_for_meta_analysis": true,
      "data_role": "primary_result",
      "population": "adults",
      "outcome": "symptom score",
      "comparison": "treatment vs control",
      "effect_direction": "positive",
      "corrected_fields": {
        "test_type": "correlation",
        "test_text": "r = 0.40, p = 0.02",
        "statistic": 0.40,
        "effect_sizes": [0.4],
        "sample_sizes": [80],
        "p_values": [0.02]
      },
      "confidence": 0.91,
      "reason": "The context reports a study result.",
      "needs_human_review": false
    }
    ```
    """

    parsed = parse_llm_context_response(response)

    assert parsed["represents_study_data"] is True
    assert parsed["usable_for_meta_analysis"] is True
    assert parsed["corrected_fields"]["statistic"] == 0.4
    assert parsed["corrected_fields"]["effect_sizes"][0]["value"] == 0.4
    assert parsed["corrected_fields"]["sample_sizes"][0]["n"] == 80
    assert parsed["corrected_fields"]["p_values"][0]["value"] == 0.02
    assert parsed["confidence"] == 0.91


def test_llm_corrected_fields_can_replace_false_deterministic_metric():
    records = [{
        "article_id": "paper-1",
        "title": "Correctable local statistic",
        "doi": "",
        "pmid": "",
        "results": [{
            "test_type": "correlation",
            "test_text": "r = 0",
            "statistic": 0.0,
            "effect_sizes": [],
            "sample_sizes": [],
            "confidence_intervals": [],
            "p_values": [{"operator": "=", "value": 0.0, "text": "p = 0"}],
            "location": {"text_source": "abstract", "paragraph_index": 0},
            "needs_human_review": True,
            "llm_context_review": {
                "represents_study_data": False,
                "usable_for_meta_analysis": True,
                "data_role": "primary_result",
                "corrected_fields": {
                    "test_type": "correlation",
                    "test_text": "r = 0.48, p = .03",
                    "statistic": 0.48,
                    "effect_sizes": [{"type": "r", "value": 0.48, "text": "r = 0.48"}],
                    "sample_sizes": [{"n": 60, "label": "children", "text": "60 children"}],
                    "confidence_intervals": [],
                    "p_values": [{"operator": "=", "value": 0.03, "text": "p = .03"}],
                },
                "needs_human_review": True,
            },
        }],
        "human_review": [],
        "needs_human_review": True,
    }]

    metrics = metrics_for_meta_analysis(records)

    assert metrics[0]["effect_sizes"] == [0.48]
    assert metrics[0]["sample_sizes"] == [60]
    assert metrics[0]["p_values"] == [0.03]
    assert metrics[0]["statistical_tests"][0]["statistic"] == 0.48
    assert metrics[0]["statistical_tests"][0]["llm_correction_applied"] is True


def test_llm_context_review_rejected_result_is_not_meta_metric():
    papers = [
        {
            "title": "Background statistic",
            "abstract": (
                "Prior work reported d = 0.62, p = .03, 95% CI 0.10, 1.14. "
                "Our study only discusses this result."
            ),
        }
    ]
    extracted = extract_statistical_evidence(papers)

    def fake_chat(*args, **kwargs):
        return json.dumps({
            "represents_study_data": False,
            "usable_for_meta_analysis": False,
            "data_role": "background_or_citation",
            "population": "",
            "outcome": "",
            "comparison": "",
            "effect_direction": "unclear",
            "corrected_fields": {},
            "confidence": 0.88,
            "reason": "The context frames the number as prior work.",
            "needs_human_review": True,
        })

    counts = apply_llm_context_reviews(
        extracted["records"],
        model="fake-model",
        chat_fn=fake_chat,
    )
    metrics = metrics_for_meta_analysis(extracted["records"])

    assert counts["llm_rejected"] >= 1
    assert extracted["records"][0]["results"][0]["llm_context_review"]["data_role"] == "background_or_citation"
    assert metrics == []
