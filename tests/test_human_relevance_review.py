"""Tests for the human relevance calibration queue."""
from __future__ import annotations

from process.human_relevance_review import (
    build_review_queue,
    compare_human_to_model,
    recommended_sample_size,
)


def test_recommended_sample_size_scales_from_twenty_to_fifty():
    assert recommended_sample_size(10) == 10
    assert recommended_sample_size(100) == 20
    assert recommended_sample_size(300) == 30
    assert recommended_sample_size(900) == 40
    assert recommended_sample_size(2000) == 50


def test_build_review_queue_keeps_reviewable_fields():
    papers = [
        {
            "doi": f"10.1/{index}",
            "title": f"Paper {index}",
            "year": 2020 + index,
            "abstract": "This is a child-development cognition abstract.",
        }
        for index in range(25)
    ]

    queue = build_review_queue(papers, seed=7, source_path="corpus.json")

    assert queue["corpus_size"] == 25
    assert queue["sample_size"] == 20
    assert queue["source_path"] == "corpus.json"
    assert queue["records"][0]["article_id"]
    assert "abstract" in queue["records"][0]
    assert queue["records"][0]["human_score"] is None
    assert queue["records"][0]["human_verified"] is False


def test_compare_human_to_model_reports_agreement_metrics():
    review_records = [
        {
            "article_id": "a",
            "title": "A",
            "human_score": 5,
            "human_verified": True,
        },
        {"article_id": "b", "title": "B", "human_score": 1},
        {"article_id": "c", "title": "C", "human_score": None},
    ]
    scored = [
        {"id": "a", "title": "A", "relevance_score": 4},
        {"id": "b", "title": "B", "relevance_score": 1},
        {"id": "c", "title": "C", "relevance_score": 5},
    ]

    summary = compare_human_to_model(review_records, scored, threshold=3)

    assert summary["reviewed"] == 2
    assert summary["human_verified"] == 1
    assert summary["matched"] == 2
    assert summary["comparisons"][0]["human_verified"] is True
    assert summary["exact_agreement_pct"] == 50.0
    assert summary["threshold_agreement_pct"] == 100.0
    assert -1.0 <= summary["score_weighted_kappa"] <= 1.0
    assert summary["threshold_kappa"] == 1.0


def test_compare_uses_a_human_corrected_model_score_without_overwriting_it():
    review_records = [{
        "article_id": "a",
        "title": "A",
        "human_score": 4,
        "model_score": 4,
        "model_score_human_corrected": True,
        "human_verified": True,
    }]
    scored = [{"id": "a", "title": "A", "relevance_score": 1}]

    summary = compare_human_to_model(review_records, scored, threshold=3)

    comparison = summary["comparisons"][0]
    assert comparison["model_score"] == 4
    assert comparison["model_score_source"] == "human_correction"
    assert summary["threshold_agreement_pct"] == 100.0
