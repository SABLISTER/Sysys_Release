"""Tests for per-component GRADE filtering in process.grade."""
from __future__ import annotations

import pytest

from process.grade import (
    DomainRating,
    GradeConcern,
    assess_imprecision,
    assess_inconsistency,
)


def test_assess_inconsistency_filters_records_by_component_studies():
    interrater_records = [
        {"article_id": "a1", "original_score": 5, "second_score": 5},
        {"article_id": "a2", "original_score": 5, "second_score": 3},
        {"article_id": "a3", "original_score": 5, "second_score": 1},
    ]
    robustness_data = {}

    # Component A includes only a1 (perfect agreement) and a2 (moderate)
    result = assess_inconsistency(interrater_records, robustness_data, ["a1", "a2"])
    assert isinstance(result, DomainRating)
    assert result.domain == "inconsistency"
    assert result.data_points["n_component_records"] == 2
    assert result.data_points["mean_agreement"] is not None


def test_assess_inconsistency_no_records_gives_serious_concern():
    result = assess_inconsistency([], {}, ["x1"])
    assert result.concern == GradeConcern.SERIOUS
    assert result.downgrade_levels == 1


def test_assess_imprecision_reads_nested_bootstrap_ci():
    robustness_data = {
        "bootstrap": {
            "ci_lower": 0.1,
            "ci_upper": 0.7,
        },
    }
    result = assess_imprecision(robustness_data, n_studies=10, component_id="c1")
    assert result.domain == "imprecision"
    assert result.data_points["ci_width"] == pytest.approx(0.6)
    assert result.data_points["component_id"] == "c1"


def test_assess_imprecision_falls_back_when_no_bootstrap_data():
    result = assess_imprecision({}, n_studies=2, component_id="c2")
    assert result.concern == GradeConcern.VERY_SERIOUS
    assert result.data_points["ci_width"] is None
